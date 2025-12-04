# models/clip_model.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import yaml
from PIL import Image  # <-- tambahkan ini
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

    print(f"[clip_model] Loading CLIP model '{model_name}' on device: {device} (dtype={dtype})")

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
                model.to(device=device, dtype=dtype)

    model.eval()
    return model, processor, device


# =========================
#  Tambahan: helper inference
# =========================

def encode_image(
    image_path: Union[str, Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode satu gambar menjadi embedding CLIP ter-normalisasi.

    Return:
        embedding shape (d,) di CPU (torch.float32)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Samakan dtype dengan model
    model_dtype = next(model.parameters()).dtype
    pixel_values = pixel_values.to(dtype=model_dtype)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.squeeze(0).to("cpu", torch.float32)


def encode_text(
    text: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode satu kalimat teks menjadi embedding CLIP ter-normalisasi.

    Return:
        embedding shape (d,) di CPU (torch.float32)
    """
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.squeeze(0).to("cpu", torch.float32)
