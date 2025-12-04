# scripts/demo_search_image_custom.py

from __future__ import annotations

from pathlib import Path

from models.clip_model import load_clip_model
from src.embedding.search import TextSearchIndex


def main():
    # Root project = folder "ml-service"
    root = Path(__file__).resolve().parents[1]

    # Path ke config, LoRA, dan index custom
    clip_config = root / "config" / "clip_config.yaml"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    index_path = root / "data" / "index" / "custom_items_index.pt"

    # Load CLIP+LoRA
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )

    # Load index text custom
    index = TextSearchIndex(index_path)

    print("\n[demo_search_image_custom] Search BERDASARKAN GAMBAR (barang real kamu)")
    print("Ketik path gambar relatif dari root project, misal:")
    print("  data/custom/images/tas_hitam_fun_boy_aula_f.jpg")
    print("  data/custom/images/kaca_mata_pink_gk_1.jpg")
    print("Ketik 'exit' atau 'quit' untuk keluar.\n")

    while True:
        query = input("Image path / 'exit'> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("\n[demo_search_image_custom] Bye!")
            break

        # Anggap user memberi path relatif dari root project
        img_path = (root / query).resolve()

        if not img_path.exists():
            print(f"[demo_search_image_custom] File gambar tidak ditemukan: {img_path}")
            continue

        # Cari top-3 hasil (karena item cuma 3 😄)
        results = index.search_by_image(
            image_path=img_path,
            model=model,
            processor=processor,
            device=device,
            top_k=3,
        )

        rel_img = img_path.relative_to(root)
        print(f"\nTop hasil untuk gambar: {rel_img}\n")

        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r.score:.4f}")
            print(f"   image: {r.image_path}")
            print(f"   text : {r.text}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
