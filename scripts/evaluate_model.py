# scripts/evaluate_model.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from models.clip_model import load_clip_model, encode_text, encode_image
from src.embedding.similarity import cosine_similarity


class CLIPEvaluator:
    """
    Evaluator untuk model CLIP+LoRA dengan berbagai metrik:
    - Retrieval metrics (Recall@K, MRR, mAP)
    - Zero-shot classification accuracy
    - Image-text matching accuracy
    """
    
    def __init__(
        self,
        clip_config_path: Path,
        lora_weights_path: Path = None,
        use_lora: bool = True,
    ):
        self.model, self.processor, self.device = load_clip_model(
            config_path=clip_config_path,
            use_lora=use_lora,
            lora_weights_path=lora_weights_path if use_lora else None,
        )
        print(f"[Evaluator] Model loaded on device: {self.device}")
    
    def compute_recall_at_k(
        self,
        similarities: torch.Tensor,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        Hitung Recall@K untuk image-to-text retrieval.
        
        Args:
            similarities: (N, N) matrix, baris=query, kolom=kandidat
            k_values: list nilai K untuk dihitung
        
        Returns:
            Dict dengan key "recall@1", "recall@5", dst.
        """
        N = similarities.shape[0]
        recalls = {}
        
        for k in k_values:
            correct = 0
            for i in range(N):
                # Top-K indices untuk query i
                topk_indices = torch.topk(similarities[i], k=min(k, N)).indices
                # Ground truth: diagonal (i harus match dengan i)
                if i in topk_indices:
                    correct += 1
            
            recall = correct / N
            recalls[f"recall@{k}"] = recall
        
        return recalls
    
    def compute_mrr(self, similarities: torch.Tensor) -> float:
        """
        Mean Reciprocal Rank (MRR).
        
        Args:
            similarities: (N, N) matrix
        
        Returns:
            MRR score
        """
        N = similarities.shape[0]
        reciprocal_ranks = []
        
        for i in range(N):
            # Sort descending
            sorted_indices = torch.argsort(similarities[i], descending=True)
            # Cari posisi ground truth (diagonal)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks)
    
    def compute_map(self, similarities: torch.Tensor) -> float:
        """
        Mean Average Precision (mAP).
        Untuk kasus image-text matching, setiap query hanya punya 1 ground truth.
        """
        N = similarities.shape[0]
        avg_precisions = []
        
        for i in range(N):
            sorted_indices = torch.argsort(similarities[i], descending=True)
            # Posisi ground truth
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            # AP = 1/rank (karena hanya 1 relevant item)
            avg_precisions.append(1.0 / rank)
        
        return np.mean(avg_precisions)
    
    def evaluate_retrieval(
        self,
        test_csv: Path,
        image_root_dir: Path,
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        Evaluasi retrieval performance pada test set.
        
        CSV harus punya kolom: image_path/image_file dan text/caption
        """
        df = pd.read_csv(test_csv)
        
        # Deteksi nama kolom yang digunakan
        image_col = None
        text_col = None
        
        for col in df.columns:
            if col.lower() in ['image_path', 'image_file', 'filename', 'image']:
                image_col = col
            if col.lower() in ['text', 'caption', 'description']:
                text_col = col
        
        if image_col is None or text_col is None:
            raise ValueError(f"CSV must have image and text columns. Found: {df.columns.tolist()}")
        
        print(f"[Evaluator] Using columns: image='{image_col}', text='{text_col}'")
        print(f"[Evaluator] Loading {len(df)} test samples...")
        
        # Encode semua gambar dan teks
        image_embeddings = []
        text_embeddings = []
        valid_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding"):
            img_filename = row[image_col]
            
            # Coba berbagai kemungkinan path
            possible_paths = [
                image_root_dir / img_filename,
                image_root_dir / Path(img_filename).name,
                Path(img_filename),
            ]
            
            img_path = None
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break
            
            if img_path is None:
                print(f"Warning: Image not found: {img_filename}, skipping...")
                continue
            
            try:
                img_emb = encode_image(img_path, self.model, self.processor, self.device)
                txt_emb = encode_text(row[text_col], self.model, self.processor, self.device)
                
                image_embeddings.append(img_emb)
                text_embeddings.append(txt_emb)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error encoding {img_filename}: {e}")
                continue
        
        if len(image_embeddings) == 0:
            raise ValueError("No valid samples found! Check image paths.")
        
        print(f"[Evaluator] Successfully loaded {len(image_embeddings)} samples")
        
        image_embeddings = torch.stack(image_embeddings)  # (N, d)
        text_embeddings = torch.stack(text_embeddings)    # (N, d)
        
        # Normalize
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Hitung similarity matrix (N, N)
        # Baris = image query, kolom = text candidates
        similarities = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Hitung metrik
        results = {}
        
        # Recall@K
        recalls = self.compute_recall_at_k(similarities, k_values)
        results.update(recalls)
        
        # MRR
        mrr = self.compute_mrr(similarities)
        results["mrr"] = mrr
        
        # mAP
        map_score = self.compute_map(similarities)
        results["map"] = map_score
        
        # Text-to-Image retrieval (transpose)
        similarities_t2i = similarities.T
        recalls_t2i = self.compute_recall_at_k(similarities_t2i, k_values)
        for k, v in recalls_t2i.items():
            results[f"t2i_{k}"] = v
        
        return results
    
    def evaluate_matching_accuracy(
        self,
        test_csv: Path,
        image_root_dir: Path,
    ) -> float:
        """
        Simple image-text matching accuracy.
        Untuk setiap image, apakah text yang benar mendapat similarity tertinggi?
        """
        df = pd.read_csv(test_csv)
        
        # Deteksi nama kolom
        image_col = None
        text_col = None
        
        for col in df.columns:
            if col.lower() in ['image_path', 'image_file', 'filename', 'image']:
                image_col = col
            if col.lower() in ['text', 'caption', 'description']:
                text_col = col
        
        image_embeddings = []
        text_embeddings = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding for matching"):
            img_filename = row[image_col]
            
            # Coba berbagai kemungkinan path
            possible_paths = [
                image_root_dir / img_filename,
                image_root_dir / Path(img_filename).name,
                Path(img_filename),
            ]
            
            img_path = None
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break
            
            if img_path is None:
                continue
            
            try:
                img_emb = encode_image(img_path, self.model, self.processor, self.device)
                txt_emb = encode_text(row[text_col], self.model, self.processor, self.device)
                
                image_embeddings.append(img_emb)
                text_embeddings.append(txt_emb)
            except:
                continue
        
        if len(image_embeddings) == 0:
            return 0.0
        
        image_embeddings = torch.stack(image_embeddings)
        text_embeddings = torch.stack(text_embeddings)
        
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        similarities = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Untuk setiap baris (image), cek apakah diagonal (correct text) punya score tertinggi
        correct = 0
        N = similarities.shape[0]
        
        for i in range(N):
            pred_idx = torch.argmax(similarities[i]).item()
            if pred_idx == i:
                correct += 1
        
        accuracy = correct / N if N > 0 else 0.0
        return accuracy


# ===== SCRIPT UTAMA =====

def run_full_evaluation():
    """
    Contoh menjalankan evaluasi lengkap.
    """
    # Config paths
    root_dir = Path(__file__).parent.parent
    clip_config = root_dir / "config/clip_config.yaml"
    
    # FIXED: Gunakan epoch 1 yang sudah ada
    lora_weights = root_dir / "models/saved/clip-lora/epoch_1"
    
    # FIXED: Gunakan file yang benar di data/text/
    test_csv = root_dir / "data/text/val_fashion.csv"  # Gunakan val sebagai test
    image_root = root_dir / "data/text/images"  # Sesuaikan dengan lokasi gambar
    
    # Check if paths exist
    if not test_csv.exists():
        print(f"❌ Error: Test CSV not found: {test_csv}")
        print(f"Please create or move your test CSV to this location")
        return
    
    # Load evaluator
    print(f"Loading model with LoRA weights from: {lora_weights}")
    evaluator = CLIPEvaluator(
        clip_config_path=clip_config,
        lora_weights_path=lora_weights,
        use_lora=True,
    )
    
    print("\n" + "="*60)
    print("EVALUASI 1: RETRIEVAL METRICS")
    print("="*60)
    
    try:
        retrieval_results = evaluator.evaluate_retrieval(
            test_csv=test_csv,
            image_root_dir=image_root,
            k_values=[1, 5, 10],
        )
        
        print("\nImage-to-Text Retrieval:")
        for metric, value in retrieval_results.items():
            if not metric.startswith("t2i_"):
                print(f"  {metric}: {value:.4f}")
        
        print("\nText-to-Image Retrieval:")
        for metric, value in retrieval_results.items():
            if metric.startswith("t2i_"):
                print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"❌ Error in retrieval evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("EVALUASI 2: MATCHING ACCURACY")
    print("="*60)
    
    try:
        matching_acc = evaluator.evaluate_matching_accuracy(
            test_csv=test_csv,
            image_root_dir=image_root,
        )
        print(f"Image-Text Matching Accuracy: {matching_acc:.4f}")
    except Exception as e:
        print(f"❌ Error in matching evaluation: {e}")
        matching_acc = 0.0
    
    # Save results
    output_file = root_dir / "results/evaluation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "retrieval": retrieval_results,
        "matching_accuracy": matching_acc,
    }
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    run_full_evaluation()