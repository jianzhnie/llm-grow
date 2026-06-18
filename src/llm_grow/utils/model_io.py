"""Model IO utilities: load, save, and merge expanded models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


def load_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> nn.Module:
    """加载 HuggingFace 因果语言模型。"""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    return model


def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


def save_model(
    model: nn.Module,
    output_dir: str | Path,
    tokenizer=None,
    safe_serialization: bool = True,
) -> None:
    """保存扩增后的模型和 tokenizer 到指定目录。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)


def verify_state_dict_keys(
    original: nn.Module,
    expanded: nn.Module,
) -> tuple[set[str], set[str]]:
    """比较原始和扩增模型的 state_dict keys。

    Returns:
        (keys_only_in_expanded, keys_only_in_original)
    """
    orig_keys = set(original.state_dict().keys())
    exp_keys = set(expanded.state_dict().keys())
    new_keys = exp_keys - orig_keys
    missing_keys = orig_keys - exp_keys
    logger.info("New keys in expanded: %d", len(new_keys))
    logger.info("Missing keys (orig→exp): %d", len(missing_keys))
    return new_keys, missing_keys
