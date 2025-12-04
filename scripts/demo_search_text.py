# scripts/demo_search_text.py

from __future__ import annotations

from pathlib import Path

from models.clip_model import load_clip_model
from src.embedding.search import TextSearchIndex


def main():
    # Root project = folder "ml-service"
    root = Path(__file__).resolve().parents[1]

    # Path ke config, LoRA, dan index
    clip_config = root / "config" / "clip_config.yaml"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    index_path = root / "data" / "index" / "fashion_text_index.pt"

    # Load CLIP+LoRA
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )

    # Load index text
    index = TextSearchIndex(index_path)

    print("\n[demo_search_text] Interactive text search demo")
    print("Ketik query (misal: 'black leather bag for men').")
    print("Ketik 'exit' atau 'quit' untuk keluar.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("\n[demo_search_text] Bye!")
            break

        # Cari top-5 hasil
        results = index.search_by_text(
            query=query,
            model=model,
            processor=processor,
            device=device,
            top_k=5,
        )

        print(f"\nTop-5 hasil untuk: \"{query}\":\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r.score:.4f}")
            print(f"   image: {r.image_path}")
            print(f"   text : {r.text[:120]}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
