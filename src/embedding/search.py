# src/embedding/search.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch

from models.clip_model import encode_text, CLIPModel, CLIPProcessor


@dataclass
class SearchResult:
    """Satu hasil pencarian."""
    index: int           # index di dalam index embeddings
    score: float         # cosine similarity
    image_path: str
    text: str


class TextSearchIndex:
    """
    Index pencarian berbasis embedding teks CLIP+LoRA.

    Cara pakai (high-level):

        index = TextSearchIndex("data/index/fashion_text_index.pt")
        results = index.search_by_text("black leather bag", model, processor, device, top_k=5)
    """

    def __init__(self, index_path: Union[str, Path]):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        print(f"[TextSearchIndex] Loading index from: {index_path}")
        obj = torch.load(index_path, map_location="cpu")

        self.embeddings: torch.Tensor = obj["embeddings"]  # (N, d)
        self.image_paths: list[str] = obj["image_path"]
        self.texts: list[str] = obj["text"]

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (N, d)")

        self.num_items, self.dim = self.embeddings.shape
        print(f"[TextSearchIndex] Loaded {self.num_items} items with dim={self.dim}")

        # Pastikan normalized
        self.embeddings = self.embeddings / self.embeddings.norm(dim=-1, keepdim=True)

    def search_with_embedding(
        self,
        query_emb: torch.Tensor,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Pencarian dengan embedding query (sudah di-encode di luar).
        query_emb: shape (d,) atau (1, d), float32 CPU.
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.unsqueeze(0)  # (1, d)
        if query_emb.shape[-1] != self.dim:
            raise ValueError(
                f"query_emb dim {query_emb.shape[-1]} != index dim {self.dim}"
            )

        # Normalize query juga
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

        # Dot product dengan semua embedding index
        sims = torch.matmul(query_emb, self.embeddings.T).squeeze(0)  # (N,)

        k = min(top_k, self.num_items)
        scores, indices = torch.topk(sims, k=k, largest=True, sorted=True)

        results: List[SearchResult] = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            results.append(
                SearchResult(
                    index=idx,
                    score=float(score),
                    image_path=self.image_paths[idx],
                    text=self.texts[idx],
                )
            )

        return results

    def search_by_text(
        self,
        query: str,
        model: CLIPModel,
        processor: CLIPProcessor,
        device: torch.device,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Langsung pencarian dari string query:
        - encode teks dengan CLIP+LoRA
        - hitung similarity dengan index
        """
        query_emb = encode_text(query, model, processor, device)
        return self.search_with_embedding(query_emb, top_k=top_k)
