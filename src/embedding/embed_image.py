# src/embedding/embed_image.py

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def _load_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def embed_image(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Union[str, Path, Image.Image],
    device: Union[str, torch.device] = "cpu",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute embedding for a single image.

    Returns:
        torch.Tensor shape (d,) if normalize=True, else (d,)
    """
    model.eval()
    pil_img = _load_image(image)

    inputs = processor(
        images=pil_img,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)

    # shape: (1, d) -> (d,)
    image_features = image_features.squeeze(0)

    if normalize:
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=False)

    return image_features.detach().cpu()


def embed_images_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Union[str, Path, Image.Image]],
    device: Union[str, torch.device] = "cpu",
    normalize: bool = True,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Compute embeddings for a list of images in batches.

    Returns:
        torch.Tensor shape (N, d)
    """
    model.eval()
    device = torch.device(device)

    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch_imgs = [ _load_image(im) for im in images[i : i + batch_size] ]

        inputs = processor(
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
        )

        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)

        if normalize:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu())

    if not all_embeddings:
        return torch.empty(0)

    return torch.cat(all_embeddings, dim=0)
