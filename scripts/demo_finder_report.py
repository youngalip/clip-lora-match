# scripts/demo_finder_report.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from src.embedding.finder_service import FinderConfig, FinderService


def main():
    root = Path(__file__).resolve().parents[1]

    # Contoh: pakai gambar tas hitam funboy yang sudah kamu punya
    sample_image = root / "data" / "custom" / "images" / "tas_hitam_fun_boy_aula_f.jpg"

    config = FinderConfig(
        root_dir=root,
        clip_config_path=root / "config" / "clip_config.yaml",
        lora_dir=root / "models" / "saved" / "clip-lora" / "epoch_1",
        index_path=root / "data" / "index" / "custom_items_index.pt",
        upload_dir=root / "data" / "reported" / "images",
        yolo_config_path=root / "config" / "yolo_config.yaml",
    )

    service = FinderService(config)

    result = service.report_item(
        src_image_path=sample_image,
        description="Tas ransel hitam polos merk funboy",
        location="Aula Gedung F",
        reporter="Penemu Testing",
        found_at=datetime.now(),
    )

    print("\n[demo_finder_report] Report selesai:")
    print(result)


if __name__ == "__main__":
    main()
