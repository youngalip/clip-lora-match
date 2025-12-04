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
    def __init__(self, index_path: Path):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        obj = torch.load(index_path, map_location="cpu")

        # Embedding wajib ada
        embs = obj.get("embeddings")
        if embs is None:
            raise ValueError("Index file does not contain 'embeddings'")

        self.embeddings: torch.Tensor = embs.float()
        if self.embeddings.dim() == 1:
            self.embeddings = self.embeddings.unsqueeze(0)

        # image_paths bisa disimpan dengan key berbeda
        images = obj.get("image_paths")
        if images is None:
            images = obj.get("image_path")
        if images is None:
            images = []

        self.image_paths: list[str] = list(images)

        # texts juga bisa beda key
        texts = obj.get("texts")
        if texts is None:
            texts = obj.get("text")
        if texts is None:
            texts = []

        self.texts: list[str] = list(texts)

        if self.embeddings.size(0) != len(self.image_paths):
            print(
                f"[TextSearchIndex] WARNING: embeddings rows ({self.embeddings.size(0)}) "
                f"!= len(image_paths) ({len(self.image_paths)})"
            )

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
    
    def search_by_image(
        self,
        image_path: Union[str, Path],
        model: CLIPModel,
        processor: CLIPProcessor,
        device: torch.device,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Pencarian berbasis gambar:
        - encode gambar dengan CLIP+LoRA
        - pakai embedding gambar untuk mencari di index (yang berisi embedding teks)

        image_path: path ke file gambar (relatif terhadap root project atau absolut).
        """
        from models.clip_model import encode_image  # import lokal untuk hindari circular

        query_emb = encode_image(image_path, model, processor, device)
        return self.search_with_embedding(query_emb, top_k=top_k)
