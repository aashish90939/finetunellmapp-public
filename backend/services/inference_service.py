from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from transformers import TextStreamer

from app_core.config import TrainConfig
from app_core.model import (
    get_peft_model_for_inference,
    format_instruction,
    load_base_model,
    load_tokenizer,
)

_CACHE: dict[str, tuple] = {}


def _resolve_config(run_output_dir: Path, cfg_overrides: dict[str, Any] | None) -> TrainConfig:
    if cfg_overrides:
        cfg = TrainConfig(**cfg_overrides)
    else:
        cfg_path = run_output_dir / "config.json"
        cfg = TrainConfig.load(cfg_path) if cfg_path.exists() else TrainConfig(model_name=str(run_output_dir))

    if not cfg.model_name:
        cfg.model_name = str(run_output_dir)
    return cfg


def load_adapter_for_run(
    run_id: str,
    run_output_dir: Path,
    cfg_overrides: dict[str, Any] | None,
    plain_model: bool,
):
    cfg = _resolve_config(run_output_dir, cfg_overrides)
    if cfg.load_in_4bit and not torch.cuda.is_available():
        # Avoid bitsandbytes requirement on CPU-only hosts
        cfg.load_in_4bit = False

    if plain_model or not (run_output_dir.exists() and run_output_dir.is_dir()):
        tokenizer = load_tokenizer(cfg)
        model = load_base_model(cfg, for_training=False)
    else:
        adapter_dir = run_output_dir / "adapter"
        if not adapter_dir.exists():
            fallback = run_output_dir / "adapter_final"
            if fallback.exists():
                adapter_dir = fallback
        if not adapter_dir.exists():
            raise FileNotFoundError(f"No adapter found for run {run_id} in {run_output_dir}")
        model, tokenizer = get_peft_model_for_inference(cfg, adapter_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return cfg, model, tokenizer, device, streamer


def generate(
    run_id: str,
    run_output_dir: Path,
    input_text: str,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    cfg_overrides: dict[str, Any] | None = None,
    plain_model: bool = False,
) -> str:
    key = run_id or str(run_output_dir)
    if key not in _CACHE:
        _CACHE[key] = load_adapter_for_run(run_id, run_output_dir, cfg_overrides, plain_model)
    cfg, model, tokenizer, device, streamer = _CACHE[key]

    if max_new_tokens is None:
        max_new_tokens = getattr(cfg, "gen_max_new_tokens", 512)
    if temperature is None:
        temperature = getattr(cfg, "gen_temperature", 0.7)
    if top_p is None:
        top_p = getattr(cfg, "gen_top_p", 0.9)

    prompt = format_instruction(cfg, input_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0.0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return output_text.strip()
