# scripts/evaluate.py

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.clip_model import load_clip_model, encode_text


# ============================================
# CONFIGURATION
# ============================================
RELEVANCE_THRESHOLD = 0.7  # Item dengan similarity >= 0.7 dianggap relevan
TOP_K_VALUES = [1, 5, 10]  # Evaluasi untuk top-1, top-5, top-10
NUM_VAL_SAMPLES = None     # None = pakai semua val data, atau set angka untuk testing


# ============================================
# UTILITY FUNCTIONS
# ============================================

def cosine_similarity(query_emb: torch.Tensor, index_emb: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity antara query dan semua embeddings di index.
    Args:
        query_emb: (d,) tensor
        index_emb: (N, d) tensor
    Returns:
        similarities: (N,) tensor
    """
    # Pastikan normalized
    query_emb = query_emb / query_emb.norm()
    index_emb = index_emb / index_emb.norm(dim=-1, keepdim=True)
    
    similarities = torch.matmul(index_emb, query_emb)  # (N,)
    return similarities


def calculate_recall_at_k(relevant_indices: List[int], top_k_indices: List[int], k: int) -> float:
    """
    Recall@K: Berapa banyak item relevan yang ditemukan di top-K?
    """
    top_k = top_k_indices[:k]
    found = sum(1 for idx in relevant_indices if idx in top_k)
    if len(relevant_indices) == 0:
        return 0.0
    return found / len(relevant_indices)


def calculate_precision_at_k(relevant_indices: List[int], top_k_indices: List[int], k: int) -> float:
    """
    Precision@K: Dari K hasil teratas, berapa yang relevan?
    """
    top_k = top_k_indices[:k]
    found = sum(1 for idx in top_k if idx in relevant_indices)
    return found / k if k > 0 else 0.0


def calculate_mrr(relevant_indices: List[int], top_k_indices: List[int]) -> float:
    """
    MRR (Mean Reciprocal Rank): 1 / rank of first relevant item
    """
    for rank, idx in enumerate(top_k_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


def calculate_average_precision(relevant_indices: List[int], top_k_indices: List[int]) -> float:
    """
    Average Precision untuk single query
    """
    if len(relevant_indices) == 0:
        return 0.0
    
    precisions = []
    num_relevant = 0
    
    for rank, idx in enumerate(top_k_indices, start=1):
        if idx in relevant_indices:
            num_relevant += 1
            precision = num_relevant / rank
            precisions.append(precision)
    
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(relevant_indices)


# ============================================
# MAIN EVALUATION FUNCTIONS
# ============================================

def build_index_from_train(
    train_csv: Path,
    model,
    processor,
    device: str
) -> Dict:
    """
    Build index dari train_fashion.csv
    Returns dict with keys: embeddings, texts, image_paths
    """
    print(f"\n[Evaluate] Building index from: {train_csv}")
    
    df = pd.read_csv(train_csv)
    if len(df) == 0:
        raise ValueError("Train CSV is empty")
    
    embeddings = []
    texts = df["text"].tolist()
    image_paths = df["image_path"].tolist()
    
    print(f"[Evaluate] Encoding {len(texts)} training texts...")
    for i, text in enumerate(tqdm(texts, desc="Encoding train")):
        emb = encode_text(text, model, processor, device)
        embeddings.append(emb)
    
    emb_tensor = torch.stack(embeddings, dim=0)  # (N, d)
    emb_tensor = emb_tensor / emb_tensor.norm(dim=-1, keepdim=True)  # normalize
    
    return {
        "embeddings": emb_tensor,
        "texts": texts,
        "image_paths": image_paths
    }


def evaluate_single_query(
    query_text: str,
    query_emb: torch.Tensor,
    index: Dict,
    threshold: float = 0.7
) -> Tuple[List[int], List[float], List[int]]:
    """
    Evaluate single query terhadap index.
    
    Returns:
        relevant_indices: indices of items with similarity >= threshold
        similarities: all similarity scores
        top_k_indices: indices sorted by similarity (descending)
    """
    index_emb = index["embeddings"]  # (N, d)
    
    # Compute similarities
    similarities = cosine_similarity(query_emb, index_emb)  # (N,)
    
    # Sort by similarity (descending)
    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    sorted_scores = similarities[sorted_indices].tolist()
    
    # Find relevant items (similarity >= threshold)
    relevant_indices = [idx for idx, score in zip(sorted_indices, sorted_scores) 
                       if score >= threshold]
    
    return relevant_indices, sorted_scores, sorted_indices


def evaluate_epoch(
    epoch_num: int,
    root_dir: Path,
    train_csv: Path,
    val_csv: Path,
    clip_config: Path,
    threshold: float = 0.7,
    top_k_values: List[int] = [1, 5, 10],
    num_samples: int = None
) -> Dict:
    """
    Evaluate single epoch.
    
    Returns:
        metrics dict with recall@k, precision@k, mrr, map, avg_time
    """
    lora_dir = root_dir / "models" / "saved" / "clip-lora" / f"epoch_{epoch_num}"
    
    if not lora_dir.exists():
        print(f"[Evaluate] Epoch {epoch_num} not found at: {lora_dir}")
        return None
    
    print(f"\n{'='*60}")
    print(f"EVALUATING EPOCH {epoch_num}")
    print(f"{'='*60}")
    print(f"LoRA dir: {lora_dir}")
    
    # Load model
    print("[Evaluate] Loading CLIP + LoRA model...")
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir
    )
    print(f"[Evaluate] Model loaded on device: {device}")
    
    # Build index from training data
    index = build_index_from_train(train_csv, model, processor, device)
    print(f"[Evaluate] Index built with {len(index['texts'])} items")
    
    # Load validation queries
    print(f"\n[Evaluate] Loading validation queries from: {val_csv}")
    val_df = pd.read_csv(val_csv)
    if num_samples:
        val_df = val_df.head(num_samples)
        print(f"[Evaluate] Using {num_samples} samples for quick evaluation")
    
    val_texts = val_df["text"].tolist()
    print(f"[Evaluate] Total validation queries: {len(val_texts)}")
    
    # Initialize metrics collectors
    all_recalls = {k: [] for k in top_k_values}
    all_precisions = {k: [] for k in top_k_values}
    all_mrr = []
    all_ap = []
    query_times = []
    
    # Evaluate each query
    print(f"\n[Evaluate] Running evaluation...")
    for query_text in tqdm(val_texts, desc=f"Epoch {epoch_num}"):
        start_time = time.time()
        
        # Encode query
        query_emb = encode_text(query_text, model, processor, device)
        
        # Search
        relevant_indices, similarities, top_k_indices = evaluate_single_query(
            query_text, query_emb, index, threshold
        )
        
        query_time = (time.time() - start_time) * 1000  # ms
        query_times.append(query_time)
        
        # Calculate metrics for this query
        for k in top_k_values:
            recall = calculate_recall_at_k(relevant_indices, top_k_indices, k)
            precision = calculate_precision_at_k(relevant_indices, top_k_indices, k)
            all_recalls[k].append(recall)
            all_precisions[k].append(precision)
        
        mrr = calculate_mrr(relevant_indices, top_k_indices)
        all_mrr.append(mrr)
        
        ap = calculate_average_precision(relevant_indices, top_k_indices)
        all_ap.append(ap)
    
    # Aggregate metrics
    metrics = {
        "epoch": epoch_num,
        "num_queries": len(val_texts),
        "threshold": threshold,
        "recall": {k: np.mean(all_recalls[k]) * 100 for k in top_k_values},
        "precision": {k: np.mean(all_precisions[k]) * 100 for k in top_k_values},
        "mrr": np.mean(all_mrr),
        "map": np.mean(all_ap),
        "avg_query_time_ms": np.mean(query_times),
    }
    
    return metrics


# ============================================
# MAIN
# ============================================

def main():
    # Setup paths
    root = Path(__file__).resolve().parents[1]
    train_csv = root / "data" / "text" / "train_fashion.csv"
    val_csv = root / "data" / "text" / "val_fashion.csv"
    clip_config = root / "config" / "clip_config.yaml"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Validate files exist
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv}")
    if not clip_config.exists():
        raise FileNotFoundError(f"CLIP config not found: {clip_config}")
    
    print("\n" + "="*60)
    print("FASHION DATASET - MODEL EVALUATION")
    print("="*60)
    print(f"Train data: {train_csv}")
    print(f"Val data: {val_csv}")
    print(f"Relevance threshold: {RELEVANCE_THRESHOLD}")
    print(f"Top-K values: {TOP_K_VALUES}")
    print("="*60)
    
    # Evaluate epochs 1-5
    all_results = []
    
    for epoch_num in range(1, 6):
        metrics = evaluate_epoch(
            epoch_num=epoch_num,
            root_dir=root,
            train_csv=train_csv,
            val_csv=val_csv,
            clip_config=clip_config,
            threshold=RELEVANCE_THRESHOLD,
            top_k_values=TOP_K_VALUES,
            num_samples=NUM_VAL_SAMPLES
        )
        
        if metrics:
            all_results.append(metrics)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch_num} SUMMARY")
            print(f"{'='*60}")
            print(f"Queries evaluated: {metrics['num_queries']}")
            print(f"\nRecall:")
            for k in TOP_K_VALUES:
                print(f"  @{k:2d}: {metrics['recall'][k]:6.2f}%")
            print(f"\nPrecision:")
            for k in TOP_K_VALUES:
                print(f"  @{k:2d}: {metrics['precision'][k]:6.2f}%")
            print(f"\nMRR:  {metrics['mrr']:.4f}")
            print(f"mAP:  {metrics['map']:.4f}")
            print(f"\nAvg Query Time: {metrics['avg_query_time_ms']:.2f}ms")
            print(f"{'='*60}\n")
    
    if len(all_results) == 0:
        print("\n[Evaluate] No epochs found to evaluate!")
        return
    
    # Save results to JSON
    json_path = results_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Evaluate] Results saved to: {json_path}")
    
    # Save human-readable summary
    summary_path = results_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("FASHION DATASET - EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Relevance Threshold: {RELEVANCE_THRESHOLD}\n")
        f.write(f"Evaluated Epochs: {[r['epoch'] for r in all_results]}\n\n")
        
        for metrics in all_results:
            f.write(f"\nEpoch {metrics['epoch']}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Queries: {metrics['num_queries']}\n\n")
            f.write("Recall:\n")
            for k in TOP_K_VALUES:
                f.write(f"  @{k:2d}: {metrics['recall'][k]:6.2f}%\n")
            f.write("\nPrecision:\n")
            for k in TOP_K_VALUES:
                f.write(f"  @{k:2d}: {metrics['precision'][k]:6.2f}%\n")
            f.write(f"\nMRR:  {metrics['mrr']:.4f}\n")
            f.write(f"mAP:  {metrics['map']:.4f}\n")
            f.write(f"Avg Query Time: {metrics['avg_query_time_ms']:.2f}ms\n")
            f.write("\n" + "="*60 + "\n")
    
    print(f"[Evaluate] Summary saved to: {summary_path}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()