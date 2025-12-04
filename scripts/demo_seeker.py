# scripts/demo_seeker.py

from __future__ import annotations

from pathlib import Path

from src.embedding.seeker_service import SeekerConfig, SeekerService


def main():
    root = Path(__file__).resolve().parents[1]

    config = SeekerConfig(
        root_dir=root,
        clip_config_path=root / "config" / "clip_config.yaml",
        lora_dir=root / "models" / "saved" / "clip-lora" / "epoch_1",
        index_path=root / "data" / "index" / "custom_items_index.pt",
        yolo_config_path=root / "config" / "yolo_config.yaml",         # <- YOLO ON
        yolo_crop_dir=root / "data" / "custom" / "query_crops",        # <- hasil crop query
    )

    service = SeekerService(config)

    print("\n[demo_seeker] Mode ORANG KEHILANGAN (search only, dengan YOLO untuk gambar)")
    print("Kamu bisa:")
    print("- Isi hanya teks query")
    print("- Isi hanya path gambar (relatif dari root project)")
    print("- Atau isi keduanya untuk kombinasi image+text")
    print("Ketik 'exit' di teks query untuk keluar.\n")

    while True:
        q_text = input("Teks deskripsi (boleh kosong, 'exit' untuk keluar)> ").strip()
        if q_text.lower() in {"exit", "quit"}:
            print("\n[demo_seeker] Bye!")
            break

        q_img = input(
            "Path gambar relatif (boleh kosong, misal: data/custom/images/tas_hitam_fun_boy_aula_f.jpg)> "
        ).strip()
        if q_img == "":
            q_img = None

        if (not q_text) and (q_img is None):
            print("[demo_seeker] Minimal isi teks atau gambar ya.\n")
            continue

        try:
            results = service.search_items(
                query_text=q_text if q_text else None,
                query_image_path=q_img,
                top_k=3,
            )
        except FileNotFoundError as e:
            print(f"[demo_seeker] Error: {e}\n")
            continue
        except Exception as e:
            print(f"[demo_seeker] Unexpected error: {e}\n")
            continue

        print("\n=== Hasil Pencarian ===")
        print(f"Query teks  : {q_text!r}" if q_text else "Query teks  : (none)")
        print(f"Query gambar: {q_img!r}" if q_img else "Query gambar: (none)")
        print()

        for i, r in enumerate(results, start=1):
            print(f"{i}. score={r.score:.4f}")
            print(f"   image: {r.image_path}")
            print(f"   text : {r.text}")
        print("=======================\n")


if __name__ == "__main__":
    main()
