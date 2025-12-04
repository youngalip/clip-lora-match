# scripts/build_custom_index.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from models.clip_model import load_clip_model, encode_text


def main():
    root = Path(__file__).resolve().parents[1]

    data_csv = root / "data" / "custom" / "my_items.csv"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    clip_config = root / "config" / "clip_config.yaml"
    index_dir = root / "data" / "index"
    index_path = index_dir / "custom_items_index.pt"

    print(f"[build_custom_index] Using data CSV : {data_csv}")
    print(f"[build_custom_index] Using LoRA dir: {lora_dir}")

    if not data_csv.exists():
        raise FileNotFoundError(f"CSV not found: {data_csv}")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint dir not found: {lora_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)

    # Paksa kolom pertama jadi index (path gambar)
    df = pd.read_csv(data_csv, header=0, index_col=0)

    print("\n[build_custom_index] df.head():")
    print(df.head())

    if "image_path" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns 'image_path' and 'text' (deskripsi & lokasi).")
    if len(df) == 0:
        raise ValueError("CSV is empty.")

    print(f"\n[build_custom_index] Number of rows: {len(df)}")

    # Sekarang:
    # - df.index        = path gambar (data/custom/images/....jpg)
    # - df['image_path'] = deskripsi barang
    # - df['text']       = lokasi
    image_paths = df.index.astype(str).tolist()
    desc = df["image_path"].astype(str).tolist()
    loc = df["text"].astype(str).tolist()

    # Gabungkan deskripsi + lokasi jadi satu kalimat
    texts = [f"{d}, {l}" for d, l in zip(desc, loc)]

    print("\n[build_custom_index] Pairs FINAL yang akan di-encode:")
    for i, (ip, tx) in enumerate(zip(image_paths, texts)):
        print(f"  Row {i}: image_path='{ip}' | text='{tx}'")
    print()

    # Load CLIP + LoRA
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )
    print(f"[build_custom_index] Model loaded on device: {device}")

    embeddings = []
    for i, text in enumerate(texts):
        emb = encode_text(text, model, processor, device)
        embeddings.append(emb)
        print(f"[build_custom_index] Encoded {i + 1}/{len(texts)} texts")

    emb_tensor = torch.stack(embeddings, dim=0)
    emb_tensor = emb_tensor / emb_tensor.norm(dim=-1, keepdim=True)

    index_obj = {
        "embeddings": emb_tensor,
        "image_path": image_paths,
        "text": texts,
    }

    torch.save(index_obj, index_path)
    print(f"\n[build_custom_index] Saved index to: {index_path}")
    print(f"[build_custom_index] Embedding shape: {emb_tensor.shape}")


if __name__ == "__main__":
    main()
