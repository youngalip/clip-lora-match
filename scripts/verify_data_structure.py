"""
Script untuk verify struktur data sebelum evaluasi.

Usage:
    python scripts/verify_data_structure.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def verify_data_structure():
    """
    Verify bahwa semua file yang diperlukan ada dan strukturnya benar.
    """
    root_dir = Path(__file__).parent.parent
    
    print("="*70)
    print("VERIFYING DATA STRUCTURE")
    print("="*70)
    
    issues = []
    
    # 1. Check CSV files
    print("\n📁 Checking CSV files...")
    
    train_fashion_csv = root_dir / "data/text/train_fashion.csv"
    val_fashion_csv = root_dir / "data/text/val_fashion.csv"
    
    if train_fashion_csv.exists():
        print(f"✅ train.csv found: {train_fashion_csv}")
        df_train = pd.read_csv(train_fashion_csv)
        print(f"   - Rows: {len(df_train)}")
        print(f"   - Columns: {df_train.columns.tolist()}")
    else:
        print(f"❌ train.csv NOT FOUND: {train_fashion_csv}")
        issues.append("train.csv missing")
    
    if val_fashion_csv.exists():
        print(f"✅ val.csv found: {val_fashion_csv}")
        df_val = pd.read_csv(val_fashion_csv)
        print(f"   - Rows: {len(df_val)}")
        print(f"   - Columns: {df_val.columns.tolist()}")
        
        # Check required columns
        required_cols = ["image_path", "text"]
        missing_cols = [col for col in required_cols if col not in df_val.columns]
        
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
            issues.append(f"val.csv missing columns: {missing_cols}")
        else:
            print(f"   ✅ All required columns present")
            
            # Show sample
            print(f"\n   Sample data (first 3 rows):")
            print(df_val.head(3).to_string(index=False))
            
            # Check image paths
            print(f"\n   Checking image paths...")
            sample_path = df_val["image_path"].iloc[0]
            print(f"   Sample path: {sample_path}")
            
            # Try to find images
            if Path(sample_path).is_absolute():
                img_exists = Path(sample_path).exists()
                print(f"   Absolute path exists: {img_exists}")
                if not img_exists:
                    issues.append("Sample image not found (absolute path)")
            else:
                # Try relative to root
                img_path = root_dir / sample_path
                img_exists = img_path.exists()
                print(f"   Relative path exists: {img_exists}")
                print(f"   Full path: {img_path}")
                if not img_exists:
                    issues.append("Sample image not found (relative path)")
    else:
        print(f"❌ val.csv NOT FOUND: {val_fashion_csv}")
        issues.append("val.csv missing")
    
    # 2. Check LoRA weights
    print("\n⚙️  Checking LoRA weights...")
    
    lora_dir = root_dir / "models/saved/clip-lora"
    
    if not lora_dir.exists():
        print(f"⚠️  LoRA directory not found: {lora_dir}")
        print("   (This is OK if you haven't trained yet)")
    else:
        epochs_found = []
        for path in lora_dir.iterdir():
            if path.is_dir() and path.name.startswith("epoch_"):
                epoch_num = path.name.split("_")[1]
                adapter_file = path / "adapter_model.safetensors"
                
                if adapter_file.exists():
                    epochs_found.append(epoch_num)
                    print(f"✅ Epoch {epoch_num} found with weights")
                else:
                    print(f"⚠️  Epoch {epoch_num} found but missing adapter_model.safetensors")
        
        if not epochs_found:
            print("⚠️  No valid LoRA weights found")
            print("   (Evaluation will use base CLIP only)")
        else:
            print(f"\n   Available epochs: {sorted(epochs_found)}")
    
    # 3. Check config
    print("\n📝 Checking config files...")
    
    clip_config = root_dir / "config/clip_config.yaml"
    if clip_config.exists():
        print(f"✅ CLIP config found: {clip_config}")
    else:
        print(f"❌ CLIP config NOT FOUND: {clip_config}")
        issues.append("clip_config.yaml missing")
    
    # Summary
    print("\n" + "="*70)
    if not issues:
        print("✅ ALL CHECKS PASSED!")
        print("="*70)
        print("\nYou can now run:")
        print("  python -m scripts.run_all_evaluations")
    else:
        print("❌ ISSUES FOUND!")
        print("="*70)
        print("\nProblems:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nPlease fix these issues before running evaluation.")
    
    return len(issues) == 0


if __name__ == "__main__":
    verify_data_structure()