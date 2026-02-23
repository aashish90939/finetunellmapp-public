from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TrainConfig:
    # Paths
    data_path: str = "data/data.csv"  # CSV with columns: input_text, output_text
    output_dir: str = "outputs"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # How to build prompts:
    #   "auto"    -> infer from model_name (Mistral vs LLaMA vs generic)
    #   "mistral" -> force Mistral-style [INST] ... [/INST]
    #   "llama"   -> force LLaMA-style chat template
    #   "plain"   -> simple system + user text
    prompt_style: str = "auto"

    # System prompt for the chat template
    system_prompt: str = (
        "You are a helpful assistant. "
        "Follow the task instructions exactly and return concise, accurate output."
    )

    # General training
    seed: int = 42
    num_train_epochs: int = 10
    max_train_samples: int | None = None  # for debugging; set to None for full data
    max_eval_samples: int | None = None

    # Sequence lengths
    max_source_length: int = 3072
    max_target_length: int = 1024

    # Optimizer / scheduler
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Batch sizes (per device)
    # We FIX them to a safe, notebook-like setting to avoid OOM
    per_device_train_batch_size: int | None = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1

    # Logging / saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3

    # LoRA (lighter than before to save memory)
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # "bfloat16" or "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Generation (for eval / deploy)
    gen_max_new_tokens: int = 2048
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9

    # Auto batch size search (DISABLED to avoid OOM surprises)
    auto_batch_search: bool = False
    auto_batch_candidates: tuple[int, ...] = (8, 4, 2, 1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
