# models/lora_adapter.py

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


def _load_lora_config(config_path: Union[str, Path]) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_lora_config(
    config_path: Union[str, Path] = "config/lora_config.yaml",
) -> LoraConfig:
    """
    Create a LoraConfig object from YAML.
    """
    cfg = _load_lora_config(config_path)

    lora_cfg = cfg.get("lora", {})
    model_cfg = cfg.get("model", {})

    # target_modules dari config, atau default Q/V proj
    target_modules = model_cfg.get("target_modules", ["q_proj", "v_proj"])

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=lora_cfg.get("task_type", "FEATURE_EXTRACTION"),
    )
    return lora_config


def attach_lora_to_clip(
    model: CLIPModel,
    lora_config: LoraConfig,
) -> CLIPModel:
    """
    Wrap CLIPModel with LoRA adapters using PEFT.
    """
    peft_model = get_peft_model(model, lora_config)
    # Optional: print jumlah trainable params
    peft_model.print_trainable_parameters()
    return peft_model
