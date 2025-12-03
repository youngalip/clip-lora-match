# scripts/test_lora_inference.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from models.clip_model import (
    load_clip_model,
    encode_image,
    encode_text,
)


def main():
    # Root project = folder "ml-service"
    root = Path(__file__).resolve().parents[1]

    # Path ke CSV val yang sudah kita buat sebelumnya
    val_csv = root / "data" / "text" / "val_fashion.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")

    # Path ke LoRA checkpoint (hasil training)
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {lora_dir}")

    print(f"[test_lora] Using val CSV  : {val_csv}")
    print(f"[test_lora] Using LoRA dir: {lora_dir}")

    df = pd.read_csv(val_csv)
    if len(df) == 0:
        raise ValueError("Validation CSV is empty.")

    # Ambil beberapa contoh random dari validation set
    num_samples = min(3, len(df))
    sample_df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # Load CLIP + LoRA
    model, processor, device = load_clip_model(
        config_path=root / "config" / "clip_config.yaml",
        use_lora=True,
        lora_weights_path=lora_dir,
    )
    print(f"[test_lora] Model loaded on device: {device}")

    for idx, row in sample_df.iterrows():
        print("\n" + "=" * 80)
        print(f"[Sample {idx + 1}]")
        print(f"image_path : {row['image_path']}")
        print(f"true_text  : {row['text']}")

        image_path = root / row["image_path"]

        # Encode gambar
        img_emb = encode_image(image_path, model, processor, device)

        # Bangun kandidat teks: 1 teks benar + beberapa teks random lain
        other_df = df[df["image_path"] != row["image_path"]]
        num_distractors = min(4, len(other_df))
        distractors = []
        if num_distractors > 0:
            distractors = other_df.sample(n=num_distractors, random_state=idx)["text"].tolist()

        candidates = [row["text"]] + distractors

        # Encode semua teks kandidat
        text_embs = []
        for t in candidates:
            emb = encode_text(t, model, processor, device)
            text_embs.append(emb)

        text_embs = torch.stack(text_embs, dim=0)  # (K, d)

        # Hitung similarity (dot product == cosine, karena sudah dinormalisasi)
        img_emb_batch = img_emb.unsqueeze(0)  # (1, d)
        sims = torch.matmul(img_emb_batch, text_embs.T).squeeze(0)  # (K,)

        # Urutkan dari paling mirip ke paling kecil
        sorted_idx = torch.argsort(sims, descending=True)

        print("\nRanking teks berdasarkan similarity dengan gambar:")
        for rank, j in enumerate(sorted_idx.tolist(), start=1):
            score = sims[j].item()
            text_preview = candidates[j][:80].replace("\n", " ")
            tag = " <-- TRUE" if j == 0 else ""
            print(f"{rank}. score={score:.4f}{tag}")
            print(f"    {text_preview}")

    print("\n[test_lora] Done.")


if __name__ == "__main__":
    main()
