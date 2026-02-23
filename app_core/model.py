from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

from .config import TrainConfig
from .prompts import create_prompt_builder


def get_bnb_config(cfg: TrainConfig) -> BitsAndBytesConfig | None:
    """
    Build bitsandbytes quantization config for 4-bit QLoRA.
    """
    if not cfg.load_in_4bit:
        return None

    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return bnb_config


def load_tokenizer(cfg: TrainConfig):
    """
    Load tokenizer and set pad token correctly.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(cfg: TrainConfig, for_training: bool = True):
    """
    Load the base model, possibly in 4-bit with bitsandbytes.
    """
    device_map = "auto"
    torch_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

    bnb_config = get_bnb_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=device_map,
        torch_dtype=torch_dtype if not cfg.load_in_4bit else None,
        quantization_config=bnb_config,
    )

    # use_cache must be False for gradient checkpointing during training
    if for_training:
        model.config.use_cache = False
        model.train()
    else:
        model.config.use_cache = True
        model.eval()

    return model


def get_peft_model_for_training(cfg: TrainConfig):
    """
    Build the 4-bit base model + LoRA adapter for training.

    IMPORTANT: we intentionally DO NOT call prepare_model_for_kbit_training
    here to avoid the large temporary fp32 allocation that was causing CUDA OOM.
    """
    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg, for_training=True)

    # Freeze all base params (LoRA will add trainable adapters on top)
    for param in base_model.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing + input grads for QLoRA-style training
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_target_modules),
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def get_peft_model_for_inference(cfg: TrainConfig, adapter_dir: str | Path):
    """
    Load base model + trained LoRA adapter for inference.
    """
    tokenizer = load_tokenizer(cfg)
    base_model = load_base_model(cfg, for_training=False)

    adapter_dir = Path(adapter_dir)
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # For 4-bit QLoRA we keep it as a PEFT model (no merge_and_unload).
    model.eval()
    return model, tokenizer


def format_instruction(cfg: TrainConfig, input_text: str) -> str:
    """
    Build an inference prompt using the same style as training.
    """
    builder = create_prompt_builder(cfg)
    pieces = builder.build(src=input_text, tgt="", eos_token="")
    return pieces.prompt_str
