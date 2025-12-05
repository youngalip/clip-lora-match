# scripts/qualitative_evaluation.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import torch

from models.clip_model import load_clip_model, encode_text, encode_image
from src.embedding.similarity import cosine_similarity


class QualitativeEvaluator:
    """
    Evaluasi kualitatif untuk melihat hasil prediksi secara visual.
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
    
    def analyze_failure_cases(
        self,
        test_csv: Path,
        image_root_dir: Path,
        output_dir: Path,
        num_cases: int = 10,
    ):
        """
        Analisis kasus-kasus dimana model gagal (worst predictions).
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
        
        if image_col is None or text_col is None:
            print(f"❌ Required columns not found. Available: {df.columns.tolist()}")
            return
        
        # Encode all
        image_embeddings = []
        text_embeddings = []
        valid_indices = []
        valid_rows = []
        
        print(f"Processing {len(df)} samples...")
        
        for idx, row in df.iterrows():
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
                valid_indices.append(idx)
                valid_rows.append(row)
            except Exception as e:
                continue
        
        if len(image_embeddings) == 0:
            print("❌ No valid samples found!")
            return
        
        print(f"Successfully loaded {len(image_embeddings)} samples")
        
        image_embeddings = torch.stack(image_embeddings)
        text_embeddings = torch.stack(text_embeddings)
        
        # Normalize
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Similarity matrix
        similarities = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Find failure cases (where correct match has low rank)
        failure_scores = []
        
        for i in range(similarities.shape[0]):
            correct_score = similarities[i, i].item()
            sorted_indices = torch.argsort(similarities[i], descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            
            failure_scores.append({
                "index": i,
                "correct_score": correct_score,
                "rank": rank,
                "failure_score": rank - correct_score,  # Higher = worse
            })
        
        # Sort by failure score
        failure_scores.sort(key=lambda x: x["failure_score"], reverse=True)
        
        # Visualize worst cases
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_cases = min(num_cases, len(failure_scores))
        
        for case_idx, case in enumerate(failure_scores[:num_cases]):
            i = case["index"]
            row = valid_rows[i]
            
            # Get top-5 predictions
            topk_scores, topk_indices = torch.topk(similarities[i], k=min(5, len(valid_rows)))
            
            # Create visualization
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            
            # Query image
            img_filename = row[image_col]
            possible_paths = [
                image_root_dir / img_filename,
                image_root_dir / Path(img_filename).name,
            ]
            
            img_path = None
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break
            
            if img_path is None:
                continue
            
            img = Image.open(img_path)
            axes[0].imshow(img)
            axes[0].set_title(
                f"Query (Rank: {case['rank']})\nScore: {case['correct_score']:.3f}",
                fontweight="bold",
                color='red',
                fontsize=9,
            )
            axes[0].axis("off")
            
            # Top-5 predictions
            for j, (pred_idx, score) in enumerate(zip(topk_indices, topk_scores)):
                pred_idx = pred_idx.item()
                score = score.item()
                
                pred_row = valid_rows[pred_idx]
                pred_img_filename = pred_row[image_col]
                
                possible_paths = [
                    image_root_dir / pred_img_filename,
                    image_root_dir / Path(pred_img_filename).name,
                ]
                
                pred_img_path = None
                for p in possible_paths:
                    if p.exists():
                        pred_img_path = p
                        break
                
                if pred_img_path is None:
                    axes[j + 1].axis("off")
                    continue
                
                pred_img = Image.open(pred_img_path)
                axes[j + 1].imshow(pred_img)
                
                is_correct = (pred_idx == i)
                color = 'green' if is_correct else 'red'
                marker = "✓" if is_correct else "✗"
                
                axes[j + 1].set_title(
                    f"{marker} Pred {j+1}\nScore: {score:.3f}",
                    color=color,
                    fontsize=9,
                )
                axes[j + 1].axis("off")
            
            # Truncate text if too long
            query_text = row[text_col]
            if len(query_text) > 80:
                query_text = query_text[:77] + "..."
            
            plt.suptitle(
                f"Failure Case {case_idx + 1}: '{query_text}'",
                fontsize=11,
                fontweight="bold",
            )
            plt.tight_layout()
            
            output_path = output_dir / f"failure_case_{case_idx + 1}.png"
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
        
        print(f"\n✅ {num_cases} failure cases saved to: {output_dir}")
    
    def visualize_embedding_space(
        self,
        test_csv: Path,
        image_root_dir: Path,
        output_path: Path,
        method: str = "tsne",
    ):
        """
        Visualisasi embedding space dengan t-SNE atau UMAP.
        """
        from sklearn.manifold import TSNE
        
        df = pd.read_csv(test_csv)
        
        # Deteksi kolom
        image_col = None
        text_col = None
        
        for col in df.columns:
            if col.lower() in ['image_path', 'image_file', 'filename', 'image']:
                image_col = col
            if col.lower() in ['text', 'caption', 'description']:
                text_col = col
        
        # Encode all
        embeddings = []
        labels = []
        
        print("Encoding samples for embedding visualization...")
        
        for idx, row in df.iterrows():
            img_filename = row[image_col]
            
            possible_paths = [
                image_root_dir / img_filename,
                image_root_dir / Path(img_filename).name,
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
                embeddings.append(img_emb.cpu().numpy())
                
                # Extract label if available
                if "label" in row:
                    labels.append(row["label"])
                else:
                    # Use index as label
                    labels.append(f"sample_{idx}")
            except:
                continue
        
        if len(embeddings) == 0:
            print("❌ No valid samples for embedding visualization")
            return
        
        embeddings = np.array(embeddings)
        
        print(f"Applying {method.upper()} to {len(embeddings)} samples...")
        
        # Apply t-SNE
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        unique_labels = list(set(labels))
        
        if len(unique_labels) <= 20:  # Only use colors if reasonable number of classes
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = [l == label for l in labels]
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[color],
                    label=label,
                    alpha=0.6,
                    s=50,
                )
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Too many classes, just plot points
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                alpha=0.5,
                s=30,
            )
        
        ax.set_title("Embedding Space Visualization (t-SNE)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Embedding visualization saved to: {output_path}")


# ===== SCRIPT UTAMA =====

def run_qualitative_evaluation():
    """
    Jalankan evaluasi kualitatif.
    """
    root_dir = Path(__file__).parent.parent
    
    # Paths - FIXED
    clip_config = root_dir / "config/clip_config.yaml"
    lora_weights = root_dir / "models/saved/clip-lora/epoch_1"  # Epoch 1
    test_csv = root_dir / "data/text/val_fashion.csv"  # Gunakan val
    image_root = root_dir / "data/text/images"
    output_dir = root_dir / "results/qualitative"
    
    if not test_csv.exists():
        print(f"❌ Error: Test CSV not found: {test_csv}")
        return
    
    # Initialize evaluator
    print("Loading model for qualitative evaluation...")
    evaluator = QualitativeEvaluator(
        clip_config_path=clip_config,
        lora_weights_path=lora_weights,
        use_lora=True,
    )
    
    print("="*60)
    print("QUALITATIVE EVALUATION")
    print("="*60)
    
    # 1. Analyze failure cases
    print("\n[1/2] Analyzing failure cases...")
    try:
        evaluator.analyze_failure_cases(
            test_csv=test_csv,
            image_root_dir=image_root,
            output_dir=output_dir / "failure_cases",
            num_cases=10,
        )
    except Exception as e:
        print(f"❌ Error in failure analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Visualize embedding space
    print("\n[2/2] Visualizing embedding space...")
    try:
        evaluator.visualize_embedding_space(
            test_csv=test_csv,
            image_root_dir=image_root,
            output_path=output_dir / "embedding_space.png",
            method="tsne",
        )
    except Exception as e:
        print(f"❌ Error in embedding visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Qualitative evaluation completed!")


if __name__ == "__main__":
    run_qualitative_evaluation()