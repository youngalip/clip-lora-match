# scripts/demo_search_text_custom.py

from __future__ import annotations

from pathlib import Path

from models.clip_model import load_clip_model
from src.embedding.search import TextSearchIndex


def main():
    root = Path(__file__).resolve().parents[1]

    clip_config = root / "config" / "clip_config.yaml"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    index_path = root / "data" / "index" / "custom_items_index.pt"

    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )

    index = TextSearchIndex(index_path)

    print("\n[demo_search_text_custom] Search barang REAL kamu")
    print("Ketik deskripsi, misal:")
    print("  'tas hitam eiger ditemukan di lab'")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("\n[demo_search_text_custom] Bye!")
            break

        results = index.search_by_text(
            query=query,
            model=model,
            processor=processor,
            device=device,
            top_k=5,
        )

        print(f"\nTop hasil untuk: \"{query}\":\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r.score:.4f}")
            print(f"   image: {r.image_path}")
            print(f"   text : {r.text}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
