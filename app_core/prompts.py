from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import TrainConfig


@dataclass
class PromptPieces:
    """
    Simple container: raw text for prompt (input) and target (output).
    """
    prompt_str: str
    target_str: str


class BasePromptBuilder:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def build(self, src: str, tgt: str, eos_token: str) -> PromptPieces:
        """
        Return (prompt_str, target_str) for one example.
        Must be implemented in subclasses.
        """
        raise NotImplementedError


class MistralPromptBuilder(BasePromptBuilder):
    """
    Matches your current Mistral-style [INST] ... [/INST] format.
    """

    def build(self, src: str, tgt: str, eos_token: str) -> PromptPieces:
        system_prompt = self.cfg.system_prompt
        user_prompt = f"<input-text>{src}</input-text>\n"

        # 1) Instruction-style prompt
        prompt_str = f"[INST] {system_prompt} {user_prompt} [/INST]"

        # 2) Target (answer)
        target_str = f" \n{tgt}{eos_token}"

        return PromptPieces(prompt_str=prompt_str, target_str=target_str)


class LlamaChatPromptBuilder(BasePromptBuilder):
    """
    Example: LLaMA-style chat template using <<SYS>>.
    Adapt as needed for your exact LLaMA model.
    """

    def build(self, src: str, tgt: str, eos_token: str) -> PromptPieces:
        system_prompt = self.cfg.system_prompt
        user_prompt = f"<input-text>{src}</input-text>\n"

        prompt_str = (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_prompt} [/INST]"
        )
        target_str = f" \n{tgt}{eos_token}"

        return PromptPieces(prompt_str=prompt_str, target_str=target_str)


class PlainPromptBuilder(BasePromptBuilder):
    """
    Fallback: simple system + user prompt, no special chat template.
    """

    def build(self, src: str, tgt: str, eos_token: str) -> PromptPieces:
        system_prompt = self.cfg.system_prompt
        user_prompt = f"<input-text>{src}</input-text>\n"

        prompt_str = f"{system_prompt}\n\n{user_prompt}"
        target_str = f" \n{tgt}{eos_token}"

        return PromptPieces(prompt_str=prompt_str, target_str=target_str)


def create_prompt_builder(cfg: TrainConfig) -> BasePromptBuilder:
    """
    Factory: chooses prompt builder based on cfg.prompt_style and cfg.model_name.

    - If prompt_style == "auto": infer from model_name.
    - Else: use the style explicitly set in config.
    """
    style = (cfg.prompt_style or "").lower()
    model_name = cfg.model_name.lower()

    if style == "auto":
        # Infer from model name
        if "mistral" in model_name:
            return MistralPromptBuilder(cfg)
        if "llama" in model_name:
            return LlamaChatPromptBuilder(cfg)
        return PlainPromptBuilder(cfg)

    if style == "mistral":
        return MistralPromptBuilder(cfg)
    if style == "llama":
        return LlamaChatPromptBuilder(cfg)

    return PlainPromptBuilder(cfg)
