# scripts/demo_search_image_yolo_custom.py

from __future__ import annotations

from pathlib import Path

from models.clip_model import load_clip_model, encode_image
from src.embedding.search import TextSearchIndex
from src.preprocessing.yolo_cropper import YoloCropper


def main():
    root = Path(__file__).resolve().parents[1]

    # Path config & model
    clip_config = root / "config" / "clip_config.yaml"
    lora_dir = root / "models" / "saved" / "clip-lora" / "epoch_1"
    index_path = root / "data" / "index" / "custom_items_index.pt"

    # Folder untuk simpan hasil crop
    crop_dir = root / "data" / "custom" / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP+LoRA
    model, processor, device = load_clip_model(
        config_path=clip_config,
        use_lora=True,
        lora_weights_path=lora_dir,
    )

    # Load index text custom
    index = TextSearchIndex(index_path)

    # Siapkan YOLO cropper (pakai config kamu yang sekarang)
    yolo_config = root / "config" / "yolo_config.yaml"
    cropper = YoloCropper(config_path=yolo_config)

    print("\n[demo_search_image_yolo_custom] Search BERDASARKAN GAMBAR + YOLO crop")
    print("Ketik path gambar relatif dari root project, misal:")
    print("  data/custom/images/tas_hitam_fun_boy_aula_f.jpg")
    print("  data/custom/images/kaca_mata_pink_gk_1.jpg")
    print("Ketik 'exit' atau 'quit' untuk keluar.\n")

    while True:
        query = input("Image path / 'exit'> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("\n[demo_search_image_yolo_custom] Bye!")
            break

        orig_path = (root / query).resolve()

        if not orig_path.exists():
            print(f"[demo_search_image_yolo_custom] File gambar tidak ditemukan: {orig_path}")
            continue

        # ----- YOLO CROP -----
        try:
            crop_paths = cropper.crop_image(
                image_path=orig_path,
                save_dir=crop_dir,
            )
        except Exception as e:
            print(f"[demo_search_image_yolo_custom] Error saat YOLO crop: {e}")
            continue

        if not crop_paths:
            print("[demo_search_image_yolo_custom] Tidak ada crop yang dihasilkan.")
            continue

        # Ambil crop pertama sebagai objek utama
        crop_path = crop_paths[0]

        rel_orig = orig_path.relative_to(root)
        rel_crop = crop_path.relative_to(root)

        print("\n[YOLO CROP]")
        print(f"  Before: {rel_orig}")
        print(f"  After : {rel_crop}")
        print("  (Silakan buka manual kedua file ini untuk lihat perbedaannya.)\n")

        # ----- ENCODE CROP & SEARCH -----
        query_emb = encode_image(crop_path, model, processor, device)
        results = index.search_with_embedding(query_emb, top_k=3)

        print(f"Top hasil untuk gambar (setelah crop): {rel_crop}\n")

        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r.score:.4f}")
            print(f"   image: {r.image_path}")
            print(f"   text : {r.text}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
