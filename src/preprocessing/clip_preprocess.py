# src/preprocessing/clip_preprocess.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from PIL import Image
from transformers import CLIPProcessor


class ClipPreprocessor:
    """
    Wrapper untuk CLIPProcessor dengan:
    - resize & normalize gambar
    - tokenisasi teks dengan padding tetap (max_length)
    """

    def __init__(self, config_path: Union[str, Path] = "config/clip_config.yaml") -> None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        model_cfg = cfg.get("model", {})
        model_name = model_cfg.get("name", "openai/clip-vit-base-patch32")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Panjang maksimum tokenizer CLIP (biasanya 77)
        self.max_length = self.processor.tokenizer.model_max_length

    def preprocess_image(self, image: Image.Image) -> Any:
        """
        Hanya preprocess gambar.
        Return: pixel_values shape (1, 3, H, W)
        """
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        return inputs["pixel_values"]

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Hanya preprocess teks dengan padding max_length,
        supaya panjang token konsisten antar sample.
        """
        encodings = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encodings

    def preprocess_pair(self, image: Image.Image, text: str) -> Dict[str, Any]:
        """
        Preprocess pasangan image + text dalam satu call.

        Return:
            {
                "pixel_values": (1, 3, H, W),
                "input_ids": (1, L),
                "attention_mask": (1, L),
            }
        """

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",           # <-- KUNCI
            truncation=True,
            max_length=self.max_length,
        )
        return inputs
