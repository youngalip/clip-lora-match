# src/embedding/embed_text.py

from __future__ import annotations

from typing import List, Union

import torch
from transformers import CLIPProcessor, CLIPModel


def embed_text(
    model: CLIPModel,
    processor: CLIPProcessor,
    text: Union[str, List[str]],
    device: Union[str, torch.device] = "cpu",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute embedding for text (single string or list of strings).

    Returns:
        if input is str:   shape (d,)
        if input is list: shape (N, d)
    """
    model.eval()
    device = torch.device(device)

    if isinstance(text, str):
        texts = [text]
        single_input = True
    else:
        texts = text
        single_input = False

    encodings = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if normalize:
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    text_features = text_features.cpu()

    if single_input:
        return text_features.squeeze(0)

    return text_features
