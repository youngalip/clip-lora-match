# models/clip_model.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import yaml
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


def _load_clip_config(config_path: Union[str, Path]) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"CLIP config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_device(device_str: Optional[str] = None) -> torch.device:
    if device_str is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def _get_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if dtype_str.lower() == "float16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def load_clip_model(
    config_path: Union[str, Path] = "config/clip_config.yaml",
    use_lora: bool = False,
    lora_weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Load CLIP model + processor sesuai config.
    Optionally attach LoRA weights untuk inference.
    """
    config = _load_clip_config(config_path)

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "openai/clip-vit-base-patch32")
    device_cfg = model_cfg.get("device")
    dtype_cfg = model_cfg.get("dtype", "float16")

    device = _get_device(device_cfg)
    dtype = _get_dtype(dtype_cfg, device)

    # Load base CLIP
    model = CLIPModel.from_pretrained(model_name)
    model.to(device=device, dtype=dtype)

    processor = CLIPProcessor.from_pretrained(model_name)

    # LoRA inference (opsional)
    if use_lora:
        paths_cfg = config.get("paths", {})
        if lora_weights_path is None:
            lora_weights_path = paths_cfg.get("lora_weights_dir")

        if lora_weights_path is None:
            print("[clip_model] use_lora=True tetapi lora_weights_path tidak diset, lanjut tanpa LoRA.")
        else:
            lora_path = Path(lora_weights_path)
            if not lora_path.exists():
                print(f"[clip_model] LoRA weights tidak ditemukan di: {lora_path}, lanjut tanpa LoRA.")
            else:
                print(f"[clip_model] Loading LoRA weights from: {lora_path}")
                model = PeftModel.from_pretrained(model, str(lora_path))
                # Untuk inference, kamu bisa pilih mau merge atau tidak.
                # Di sini kita biarkan sebagai PeftModel supaya fleksibel untuk fine-tuning lanjutan.
                model.to(device=device, dtype=dtype)

    model.eval()
    return model, processor, device
