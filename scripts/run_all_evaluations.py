# scripts/run_all_evaluations.py

"""
Script utama untuk menjalankan semua evaluasi model CLIP+LoRA.

Usage:
    python scripts/run_all_evaluations.py --config config/evaluation_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from datetime import datetime

from scripts.evaluate_model import run_full_evaluation
from scripts.compare_models import run_model_comparison
from scripts.qualitative_evaluation import run_qualitative_evaluation


def load_evaluation_config(config_path: Path) -> dict:
    """Load evaluation configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_evaluation_report(results_dir: Path):
    """
    Buat laporan evaluasi dalam format markdown.
    """
    import json
    
    report_path = results_dir / "evaluation_report.md"
    
    # Load results
    comparison_file = results_dir / "model_comparison.json"
    
    if not comparison_file.exists():
        print(f"⚠️  Warning: {comparison_file} not found")
        return
    
    with open(comparison_file, "r") as f:
        comparison_results = json.load(f)
    
    # Create markdown report
    with open(report_path, "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        f.write("## 1. Model Comparison\n\n")
        
        # Table header
        f.write("| Model | Recall@1 | Recall@5 | Recall@10 | MRR | mAP | Matching Acc |\n")
        f.write("|-------|----------|----------|-----------|-----|-----|-------------|\n")
        
        # Table rows
        for model_name, results in comparison_results.items():
            f.write(f"| {model_name} ")
            f.write(f"| {results.get('recall@1', 0):.4f} ")
            f.write(f"| {results.get('recall@5', 0):.4f} ")
            f.write(f"| {results.get('recall@10', 0):.4f} ")
            f.write(f"| {results.get('mrr', 0):.4f} ")
            f.write(f"| {results.get('map', 0):.4f} ")
            f.write(f"| {results.get('matching_accuracy', 0):.4f} |\n")
        
        f.write("\n---\n\n")
        f.write("## 2. Best Models\n\n")
        
        # Find best for each metric
        metrics = ["recall@1", "recall@5", "recall@10", "mrr", "map", "matching_accuracy"]
        
        for metric in metrics:
            best_model = max(
                comparison_results.items(),
                key=lambda x: x[1].get(metric, 0)
            )
            f.write(f"- **{metric}**: {best_model[0]} ({best_model[1].get(metric, 0):.4f})\n")
        
        f.write("\n---\n\n")
        f.write("## 3. Visualizations\n\n")
        f.write("### Recall Comparison\n")
        f.write("![Recall Comparison](plots/recall_comparison.png)\n\n")
        
        f.write("### Metrics Heatmap\n")
        f.write("![Metrics Heatmap](plots/metrics_heatmap.png)\n\n")
        
        if (results_dir / "plots/radar_comparison.png").exists():
            f.write("### Radar Comparison\n")
            f.write("![Radar Comparison](plots/radar_comparison.png)\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Performance Analysis\n\n")
        
        # Auto-generate recommendations
        best_recall1 = max(comparison_results.items(), key=lambda x: x[1].get("recall@1", 0))
        
        if "LoRA" in best_recall1[0]:
            f.write("✅ **LoRA fine-tuning improves retrieval performance.**\n\n")
            f.write(f"- Best model: {best_recall1[0]}\n")
            f.write(f"- Recall@1: {best_recall1[1].get('recall@1', 0):.4f}\n\n")
            
            # Calculate improvement over base
            base_results = {k: v for k, v in comparison_results.items() if "Base" in k}
            if base_results:
                base_model = list(base_results.keys())[0]
                base_recall1 = comparison_results[base_model].get("recall@1", 0)
                improvement = ((best_recall1[1].get('recall@1', 0) - base_recall1) / base_recall1 * 100) if base_recall1 > 0 else 0
                f.write(f"**Improvement over base model:** +{improvement:.2f}%\n\n")
        else:
            f.write("⚠️ **Base CLIP performs better than LoRA fine-tuning.**\n\n")
            f.write("Consider:\n")
            f.write("- Adjusting LoRA hyperparameters (rank, alpha)\n")
            f.write("- Using more training data\n")
            f.write("- Training for more epochs\n")
            f.write("- Checking if data preprocessing is correct\n")
        
        f.write("\n---\n\n")
        f.write("## 5. Recommendations\n\n")
        
        # Check current epoch
        lora_models = [k for k in comparison_results.keys() if "LoRA" in k]
        if len(lora_models) == 1:
            f.write("### Training Status\n")
            f.write("- ⚠️ Only 1 epoch trained so far\n")
            f.write("- 📈 Recommendation: Train for more epochs (3-5) to see full potential\n")
            f.write("- 🔍 Monitor validation loss to avoid overfitting\n\n")
        
        f.write("### Next Steps\n")
        f.write("1. Continue training for more epochs\n")
        f.write("2. Experiment with different LoRA ranks (current: 8)\n")
        f.write("3. Try different learning rates\n")
        f.write("4. Collect more training data if available\n")
        f.write("5. Analyze failure cases to understand model weaknesses\n")
    
    print(f"\n📝 Evaluation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all model evaluations")
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluation_config.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation (faster)",
    )
    parser.add_argument(
        "--skip-qualitative",
        action="store_true",
        help="Skip qualitative evaluation",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip base and qualitative",
    )
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.skip_base = True
        args.skip_qualitative = True
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_evaluation_config(config_path)
        print(f"✅ Loaded config from: {config_path}")
    else:
        print(f"⚠️  Config not found: {config_path}, using defaults")
        config = {}
    
    root_dir = Path(__file__).parent.parent
    results_dir = root_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STARTING COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Skip base: {args.skip_base}")
    print(f"Skip qualitative: {args.skip_qualitative}")
    print("="*70)
    
    # 1. Full evaluation (single model) - Skip if requested
    if not args.skip_base:
        print("\n" + "="*70)
        print("STEP 1: Full Evaluation (LoRA Model)")
        print("="*70)
        try:
            run_full_evaluation()
        except Exception as e:
            print(f"❌ Error in full evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping Step 1 (Full Evaluation)")
    
    # 2. Model comparison (always run this)
    print("\n" + "="*70)
    print("STEP 2: Model Comparison (Base vs LoRA)")
    print("="*70)
    try:
        run_model_comparison()
    except Exception as e:
        print(f"❌ Error in model comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Qualitative evaluation
    if not args.skip_qualitative:
        print("\n" + "="*70)
        print("STEP 3: Qualitative Evaluation")
        print("="*70)
        try:
            run_qualitative_evaluation()
        except Exception as e:
            print(f"❌ Error in qualitative evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⏭️  Skipping Step 3 (Qualitative Evaluation)")
    
    # 4. Create report (always run if comparison exists)
    print("\n" + "="*70)
    print("STEP 4: Creating Evaluation Report")
    print("="*70)
    try:
        create_evaluation_report(results_dir)
    except Exception as e:
        print(f"❌ Error creating report: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ ALL EVALUATIONS COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    
    # Check which files were generated
    generated_files = []
    if (results_dir / "evaluation_results.json").exists():
        generated_files.append("  ✓ evaluation_results.json")
    if (results_dir / "model_comparison.json").exists():
        generated_files.append("  ✓ model_comparison.json")
    if (results_dir / "evaluation_report.md").exists():
        generated_files.append("  ✓ evaluation_report.md")
    if (results_dir / "plots").exists():
        generated_files.append("  ✓ plots/ directory")
    if (results_dir / "qualitative").exists():
        generated_files.append("  ✓ qualitative/ directory")
    
    if generated_files:
        print("\n".join(generated_files))
    else:
        print("  ⚠️ No output files generated (check errors above)")
    
    print("\n💡 Tips:")
    print("  - View evaluation_report.md for detailed analysis")
    print("  - Check plots/ for visualizations")
    print("  - Use --quick for faster evaluation (skips base model)")


if __name__ == "__main__":
    main()