# src/embedding/similarity.py

from __future__ import annotations

from typing import Tuple

import torch


def cosine_similarity(
    query: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between query and candidates.

    Args:
        query:      (d,) or (1, d)
        candidates: (N, d)

    Returns:
        similarities: (N,)
    """
    if query.dim() == 1:
        query = query.unsqueeze(0)  # (1, d)

    # Normalize if not already normalized
    query = query / query.norm(p=2, dim=-1, keepdim=True)
    candidates = candidates / candidates.norm(p=2, dim=-1, keepdim=True)

    # (1, d) x (d, N) -> (1, N)
    sims = torch.matmul(query, candidates.T)  # (1, N)
    return sims.squeeze(0)  # (N,)


def top_k_similar(
    query: torch.Tensor,
    candidates: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find top-k most similar candidates to query.

    Args:
        query:      (d,) or (1, d)
        candidates: (N, d)
        k:          number of results

    Returns:
        (topk_values, topk_indices)
        where:
            topk_values  shape: (k,)
            topk_indices shape: (k,)
    """
    sims = cosine_similarity(query, candidates)  # (N,)
    k = min(k, sims.shape[0])
    topk_values, topk_indices = torch.topk(sims, k)
    return topk_values, topk_indices
