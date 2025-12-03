# src/preprocessing/augment.py

from __future__ import annotations

import random
from typing import Optional
from PIL import Image, ImageEnhance


class ImageAugmenter:
    """
    Simple image augmentation utility.
    Using only Pillow, so no extra dependency like torchvision required.

    Typical usage:
        aug = ImageAugmenter()
        img_aug = aug.augment(img)
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
        brightness_factor_range: tuple[float, float] = (0.8, 1.2),
        contrast_factor_range: tuple[float, float] = (0.8, 1.2),
        rotate_prob: float = 0.3,
        max_rotate_deg: int = 15,
    ) -> None:
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness_factor_range = brightness_factor_range
        self.contrast_factor_range = contrast_factor_range
        self.rotate_prob = rotate_prob
        self.max_rotate_deg = max_rotate_deg

    def augment(self, image: Image.Image) -> Image.Image:
        """
        Apply a random combination of simple augmentations.

        Args:
            image: PIL.Image in RGB

        Returns:
            Augmented PIL.Image
        """
        img = image.convert("RGB")

        # Random horizontal flip
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            img = img.rotate(angle, expand=True)

        # Random brightness & contrast
        if random.random() < self.color_jitter_prob:
            # Brightness
            b_min, b_max = self.brightness_factor_range
            brightness_factor = random.uniform(b_min, b_max)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

            # Contrast
            c_min, c_max = self.contrast_factor_range
            contrast_factor = random.uniform(c_min, c_max)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        return img


def default_augmenter() -> ImageAugmenter:
    """
    Helper to quickly get a default augmenter.
    """
    return ImageAugmenter()
