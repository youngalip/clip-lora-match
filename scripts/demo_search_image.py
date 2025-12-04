# scripts/demo_search_image.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.clip_model import load_clip_model
from src.embedding.search import TextSearchIndex


def main():
    # Root project = folder "ml-service"
    root = Path(__file__).resolve().parents[1]

    # Path ke config, LoRA, index, dan val CSV
    clip_config = root / "config" / "clip_config.yaml"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    index_path = root / "data" / "index" / "fashion_text_index.pt"
    val_csv = root / "data" / "text" / "val_fashion.csv"

    # Load CLIP+LoRA
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )

    # Load index text
    index = TextSearchIndex(index_path)

    # Load val CSV (opsional, buat fitur 'sample')
    val_df = None
    if val_csv.exists():
        val_df = pd.read_csv(val_csv)
        if len(val_df) == 0:
            val_df = None

    print("\n[demo_search_image] Interactive image search demo")
    print("Ketik path gambar relatif dari root project, misal:")
    print("  data/external/fashion/images/42938.jpg")
    if val_df is not None:
        print("Atau ketik 'sample' untuk memakai gambar random dari val_fashion.csv.")
    print("Ketik 'exit' atau 'quit' untuk keluar.\n")

    while True:
        query = input("Image path / 'sample' / 'exit'> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("\n[demo_search_image] Bye!")
            break

        if query.lower() == "sample":
            if val_df is None:
                print("[demo_search_image] val_fashion.csv tidak tersedia atau kosong.")
                continue
            row = val_df.sample(n=1, random_state=None).iloc[0]
            rel_img_path = row["image_path"]
            img_path = root / rel_img_path
            true_text = row["text"]
            print(f"\n[sample] Menggunakan sample dari val_fashion:")
            print(f"    image_path: {rel_img_path}")
            print(f"    true_text : {true_text[:120]}")
        else:
            # Anggap user memberi path relatif dari root project
            img_path = (root / query).resolve()
            rel_img_path = img_path.relative_to(root) if img_path.exists() else query
            true_text = None

        if not img_path.exists():
            print(f"[demo_search_image] File gambar tidak ditemukan: {img_path}")
            continue

        # Cari top-5 hasil
        results = index.search_by_image(
            image_path=img_path,
            model=model,
            processor=processor,
            device=device,
            top_k=5,
        )

        print(f"\nTop-5 hasil untuk gambar: {rel_img_path}\n")
        if true_text is not None:
            print(f"[Info] Deskripsi ground-truth: {true_text[:200]}\n")

        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r.score:.4f}")
            print(f"   image: {r.image_path}")
            print(f"   text : {r.text[:150]}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
