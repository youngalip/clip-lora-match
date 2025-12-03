# datasets/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from src.preprocessing.clip_preprocess import ClipPreprocessor
from src.preprocessing.augment import ImageAugmenter, default_augmenter


class ClipPairDataset(Dataset):
    """
    Dataset untuk training CLIP+LoRA.
    Membaca CSV dengan kolom minimal:
        - image_path : path relatif/absolut ke file gambar
        - text       : deskripsi teks untuk gambar tsb

    Opsional: kolom lain (item_id, category, dst) akan diabaikan.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root_dir: str | Path = ".",
        use_augmentation: bool = False,
        augmenter: Optional[ImageAugmenter] = None,
        clip_config_path: str | Path = "config/clip_config.yaml",
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "image_path" not in self.df.columns or "text" not in self.df.columns:
            raise ValueError("CSV must contain 'image_path' and 'text' columns.")

        self.image_root_dir = Path(image_root_dir)
        self.use_augmentation = use_augmentation
        self.augmenter = augmenter or (default_augmenter() if use_augmentation else None)

        # Preprocessor CLIP
        self.preprocessor = ClipPreprocessor(clip_config_path)

    def __len__(self) -> int:
        return len(self.df)

    def _get_image_full_path(self, rel_path: str) -> Path:
        path = Path(rel_path)
        if path.is_absolute():
            return path
        return self.image_root_dir / path

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img_path = self._get_image_full_path(row["image_path"])
        text = str(row["text"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Augment (hanya kalau mode training)
        if self.use_augmentation and self.augmenter is not None:
            img = self.augmenter.augment(img)

        # Preprocess pasangan (image + text)
        inputs = self.preprocessor.preprocess_pair(image=img, text=text)
        # inputs: pixel_values (1,3,H,W), input_ids (1,L), attention_mask (1,L)

        pixel_values = inputs["pixel_values"].squeeze(0)        # (3, H, W)
        input_ids = inputs["input_ids"].squeeze(0)              # (L,)
        attention_mask = inputs["attention_mask"].squeeze(0)    # (L,)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # optional info
            "image_path": str(img_path),
            "raw_text": text,
        }
