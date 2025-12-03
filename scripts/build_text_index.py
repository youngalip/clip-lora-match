# scripts/build_text_index.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from models.clip_model import load_clip_model, encode_text


def main():
    # Root project = folder "ml-service"
    root = Path(__file__).resolve().parents[1]

    # ==== KONFIGURASI SEDERHANA (bisa kamu ubah kalau mau) ====
    data_csv = root / "data" / "text" / "train_fashion.csv"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    clip_config = root / "config" / "clip_config.yaml"
    index_dir = root / "data" / "index"
    index_path = index_dir / "fashion_text_index.pt"
    # ===========================================================

    print(f"[build_index] Using data CSV : {data_csv}")
    print(f"[build_index] Using LoRA dir: {lora_dir}")

    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv}")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint dir not found: {lora_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_csv)
    if "text" not in df.columns or "image_path" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'image_path' columns.")
    if len(df) == 0:
        raise ValueError("CSV is empty.")

    print(f"[build_index] Number of rows: {len(df)}")

    # Load CLIP+LoRA untuk inference
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )
    print(f"[build_index] Model loaded on device: {device}")

    embeddings = []
    texts = df["text"].tolist()
    image_paths = df["image_path"].tolist()

    # Encode semua teks
    for i, text in enumerate(texts):
        emb = encode_text(text, model, processor, device)  # (d,) CPU float32
        embeddings.append(emb)

        if (i + 1) % 500 == 0 or (i + 1) == len(texts):
            print(f"[build_index] Encoded {i + 1}/{len(texts)} texts")

    emb_tensor = torch.stack(embeddings, dim=0)  # (N, d)

    # Pastikan normalized (sebenarnya encode_text sudah normalize, tapi kita perkuat lagi)
    emb_tensor = emb_tensor / emb_tensor.norm(dim=-1, keepdim=True)

    index_obj = {
        "embeddings": emb_tensor,              # (N, d) tensor
        "image_path": image_paths,            # list[str]
        "text": texts,                        # list[str]
    }

    torch.save(index_obj, index_path)
    print(f"[build_index] Saved index to: {index_path}")
    print(f"[build_index] Embedding shape: {emb_tensor.shape}")


if __name__ == "__main__":
    main()
