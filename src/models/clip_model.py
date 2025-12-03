# models/clip_model.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union, Optional

import yaml
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


def _load_clip_config(config_path: Union[str, Path]) -> dict:
    path = Path(config_path)
    if not path.exists():
        # fallback kalau config belum ada
        return {
            "model": {
                "name": "openai/clip-vit-base-patch32",
            }
        }
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_clip_model(
    config_path: Union[str, Path] = "config/clip_config.yaml",
    use_lora: bool = False,
    lora_weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """
    Load CLIP model + processor.

    - config_path       : YAML untuk nama model (optional).
    - use_lora          : True jika ingin load LoRA hasil training.
    - lora_weights_path : folder checkpoint LoRA (models/saved/clip-lora/epoch_X).

    Return:
        model      : CLIPModel (bisa base-only atau sudah dibungkus PEFT LoRA)
        processor  : CLIPProcessor
        device     : torch.device (cuda/cpu)
    """
    cfg = _load_clip_config(config_path)
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "openai/clip-vit-base-patch32")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip_model] Loading CLIP model '{model_name}' on device: {device}")

    base_model = CLIPModel.from_pretrained(model_name)

    if use_lora:
        if lora_weights_path is None:
            raise ValueError("use_lora=True tapi lora_weights_path=None")
        lora_weights_path = Path(lora_weights_path)
        if not lora_weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found at {lora_weights_path}")

        print(f"[clip_model] Loading LoRA weights from: {lora_weights_path}")
        # Bungkus base_model dengan PeftModel yang sudah disimpan
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
    else:
        model = base_model

    model.to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor, device


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
