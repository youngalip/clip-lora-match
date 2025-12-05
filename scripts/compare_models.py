# scripts/compare_models.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scripts.evaluate_model import CLIPEvaluator


class ModelComparator:
    """
    Membandingkan performance model base CLIP vs CLIP+LoRA.
    """
    
    def __init__(self, clip_config_path: Path):
        self.clip_config = clip_config_path
        self.results = {}
    
    def evaluate_model(
        self,
        model_name: str,
        lora_weights_path: Path = None,
        test_csv: Path = None,
        image_root_dir: Path = None,
    ) -> Dict[str, float]:
        """
        Evaluasi satu model dan simpan hasilnya.
        """
        use_lora = lora_weights_path is not None
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        evaluator = CLIPEvaluator(
            clip_config_path=self.clip_config,
            lora_weights_path=lora_weights_path,
            use_lora=use_lora,
        )
        
        # Retrieval metrics
        try:
            retrieval_results = evaluator.evaluate_retrieval(
                test_csv=test_csv,
                image_root_dir=image_root_dir,
                k_values=[1, 5, 10],
            )
        except Exception as e:
            print(f"❌ Error in retrieval evaluation: {e}")
            retrieval_results = {
                "recall@1": 0.0,
                "recall@5": 0.0,
                "recall@10": 0.0,
                "mrr": 0.0,
                "map": 0.0,
            }
        
        # Matching accuracy
        try:
            matching_acc = evaluator.evaluate_matching_accuracy(
                test_csv=test_csv,
                image_root_dir=image_root_dir,
            )
        except Exception as e:
            print(f"❌ Error in matching evaluation: {e}")
            matching_acc = 0.0
        
        results = {
            **retrieval_results,
            "matching_accuracy": matching_acc,
        }
        
        self.results[model_name] = results
        
        print(f"\n✅ {model_name} evaluation completed!")
        return results
    
    def compare_all(
        self,
        base_model_name: str = "Base CLIP",
        lora_epochs: List[int] = [1],
        test_csv: Path = None,
        image_root_dir: Path = None,
        lora_dir: Path = None,
    ):
        """
        Evaluasi base model + berbagai epoch LoRA.
        """
        # 1. Evaluasi base model (tanpa LoRA)
        print("\n📊 Evaluating base CLIP model...")
        self.evaluate_model(
            model_name=base_model_name,
            lora_weights_path=None,
            test_csv=test_csv,
            image_root_dir=image_root_dir,
        )
        
        # 2. Evaluasi setiap epoch LoRA
        for epoch in lora_epochs:
            lora_path = lora_dir / f"epoch_{epoch}"
            
            if not lora_path.exists():
                print(f"⚠️  Warning: {lora_path} not found, skipping...")
                continue
            
            print(f"\n📊 Evaluating CLIP+LoRA (Epoch {epoch})...")
            self.evaluate_model(
                model_name=f"CLIP+LoRA (Epoch {epoch})",
                lora_weights_path=lora_path,
                test_csv=test_csv,
                image_root_dir=image_root_dir,
            )
        
        return self.results
    
    def save_results(self, output_path: Path):
        """Simpan hasil comparison ke JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    def plot_comparison(self, output_dir: Path):
        """
        Buat visualisasi perbandingan model.
        """
        if len(self.results) == 0:
            print("⚠️  No results to plot")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Bar plot untuk Recall@K
        self._plot_recall_comparison(output_dir / "recall_comparison.png")
        
        # 2. Heatmap perbandingan semua metrik
        self._plot_metrics_heatmap(output_dir / "metrics_heatmap.png")
        
        # 3. Radar chart jika ada lebih dari 1 model
        if len(self.results) > 1:
            self._plot_radar_chart(output_dir / "radar_comparison.png")
        
        print(f"\n📊 Plots saved to: {output_dir}")
    
    def _plot_recall_comparison(self, output_path: Path):
        """Plot Recall@K comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ["recall@1", "recall@5", "recall@10"]
        models = list(self.results.keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, model in enumerate(models):
            values = [self.results[model].get(m, 0) for m in metrics]
            ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)])
        
        ax.set_xlabel("Metric", fontsize=12, fontweight='bold')
        ax.set_ylabel("Score", fontsize=12, fontweight='bold')
        ax.set_title("Recall@K Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")
    
    def _plot_metrics_heatmap(self, output_path: Path):
        """Plot heatmap of all metrics."""
        # Prepare data
        models = list(self.results.keys())
        metrics = ["recall@1", "recall@5", "recall@10", "mrr", "map", "matching_accuracy"]
        
        data = []
        for model in models:
            row = [self.results[model].get(m, 0) for m in metrics]
            data.append(row)
        
        df = pd.DataFrame(data, index=models, columns=metrics)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "Score"},
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title("Metrics Comparison Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")
    
    def _plot_radar_chart(self, output_path: Path):
        """Plot radar chart untuk perbandingan multi-metrik."""
        from math import pi
        
        models = list(self.results.keys())
        metrics = ["recall@1", "recall@5", "recall@10", "mrr", "map", "matching_accuracy"]
        
        # Setup
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for i, model in enumerate(models):
            values = [self.results[model].get(m, 0) for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Multi-Metric Comparison (Radar Chart)", 
                     fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_path.name}")
    
    def print_summary_table(self):
        """Print tabel ringkasan hasil."""
        if len(self.results) == 0:
            print("⚠️  No results to display")
            return
        
        models = list(self.results.keys())
        metrics = ["recall@1", "recall@5", "recall@10", "mrr", "map", "matching_accuracy"]
        
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)
        
        # Header
        header = f"{'Model':<30}"
        for m in metrics:
            header += f"{m:>14}"
        print(header)
        print("-" * 100)
        
        # Rows
        for model in models:
            row = f"{model:<30}"
            for m in metrics:
                value = self.results[model].get(m, 0)
                row += f"{value:>14.4f}"
            print(row)
        
        print("="*100)
        
        # Find best model for each metric
        print("\n🏆 BEST MODELS:")
        for m in metrics:
            best_model = max(models, key=lambda x: self.results[x].get(m, 0))
            best_score = self.results[best_model].get(m, 0)
            print(f"  {m:<25} : {best_model:<30} ({best_score:.4f})")
        
        # Improvement analysis
        if len(models) > 1 and "Base CLIP" in models[0]:
            print("\n📈 IMPROVEMENT ANALYSIS:")
            base_results = self.results[models[0]]
            
            for model in models[1:]:
                print(f"\n  {model} vs Base CLIP:")
                for m in metrics:
                    base_score = base_results.get(m, 0)
                    model_score = self.results[model].get(m, 0)
                    improvement = ((model_score - base_score) / base_score * 100) if base_score > 0 else 0
                    symbol = "↑" if improvement > 0 else "↓"
                    print(f"    {m:<20} : {improvement:>+7.2f}% {symbol}")


# ===== SCRIPT UTAMA =====

def run_model_comparison():
    """
    Jalankan perbandingan lengkap antara base model dan LoRA.
    """
    root_dir = Path(__file__).parent.parent
    
    # Paths - FIXED untuk struktur Anda
    clip_config = root_dir / "config/clip_config.yaml"
    test_csv = root_dir / "data/text/val_fashion.csv"  # Gunakan val sebagai test
    image_root = root_dir / "data/text/images"
    lora_dir = root_dir / "models/saved/clip-lora"
    
    # Check paths
    if not test_csv.exists():
        print(f"❌ Error: Test CSV not found: {test_csv}")
        return
    
    print(f"Using test CSV: {test_csv}")
    print(f"Image root: {image_root}")
    
    # Initialize comparator
    comparator = ModelComparator(clip_config_path=clip_config)
    
    # Compare models - FIXED: Hanya epoch 1
    comparator.compare_all(
        base_model_name="Base CLIP (No LoRA)",
        lora_epochs=[1],  # Hanya epoch 1 yang sudah di-train
        test_csv=test_csv,
        image_root_dir=image_root,
        lora_dir=lora_dir,
    )
    
    # Print summary
    comparator.print_summary_table()
    
    # Save results
    results_dir = root_dir / "results"
    comparator.save_results(results_dir / "model_comparison.json")
    
    # Create visualizations
    comparator.plot_comparison(results_dir / "plots")
    
    print("\n✅ Model comparison completed!")


if __name__ == "__main__":
    run_model_comparison()