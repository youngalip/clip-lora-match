# src/embedding/finder_service.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import shutil
import torch

from models.clip_model import load_clip_model, encode_text, encode_image
from src.preprocessing.yolo_cropper import YoloCropper

# DB import
from src.db.database import SessionLocal
from src.db.models import FoundItem


@dataclass
class FinderConfig:
    """
    Konfigurasi untuk sisi PENEMU (report item ditemukan).

    - root_dir         : root project (misal: .../ml-service)
    - clip_config_path : config/clip_config.yaml
    - lora_dir         : folder weights LoRA (misal: models/saved/clip-lora/epoch_1)
    - index_path       : index barang ditemukan (misal: data/index/custom_items_index.pt)
    - upload_dir       : folder tempat menyimpan gambar laporan (misal: data/reported/images)
    - yolo_config_path : config YOLO (optional, boleh None)
    """
    root_dir: Path
    clip_config_path: Path
    lora_dir: Path
    index_path: Path
    upload_dir: Path
    yolo_config_path: Optional[Path] = None


class FinderService:
    """
    Service untuk PENEMU BARANG:
    - simpan gambar laporan ke folder upload_dir
    - (opsional) YOLO crop untuk embedding
    - hitung embedding CLIP+LoRA
    - update index .pt
    - simpan metadata ke Postgres (tabel found_items)
    """

    def __init__(self, config: FinderConfig) -> None:
        self.config = config

        # Pastikan folder upload ada
        self.config.upload_dir.mkdir(parents=True, exist_ok=True)

        # Load CLIP+LoRA
        self.model, self.processor, self.device = load_clip_model(
            config_path=self.config.clip_config_path,
            use_lora=True,
            lora_weights_path=self.config.lora_dir,
        )

        # YOLO (opsional)
        self.yolo_cropper: Optional[YoloCropper] = None
        if self.config.yolo_config_path is not None and self.config.yolo_config_path.exists():
            print(f"[FinderService] Mengaktifkan YOLO dengan config: {self.config.yolo_config_path}")
            self.yolo_cropper = YoloCropper(config_path=self.config.yolo_config_path)
        else:
            print("[FinderService] YOLO dimatikan atau config tidak ditemukan, lanjut tanpa YOLO.")

    # --------- UTIL INDEX (.pt) ---------

    def _load_index(self) -> Dict[str, Any]:
        """
        Load index .pt kalau ada, kalau tidak return struktur kosong.
        """
        if self.config.index_path.exists():
            data = torch.load(self.config.index_path, map_location="cpu")
            return {
                "embeddings": data.get("embeddings"),
                "image_paths": data.get("image_paths", []),
                "texts": data.get("texts", []),
            }

        # index baru
        return {
            "embeddings": None,
            "image_paths": [],
            "texts": [],
        }

    def _save_index(self, embeddings: torch.Tensor, image_paths, texts) -> None:
        self.config.index_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings,
                "image_paths": list(image_paths),
                "texts": list(texts),
            },
            self.config.index_path,
        )
        print(f"[FinderService] Index updated and saved to: {self.config.index_path}")

    # --------- API UTAMA: REPORT ITEM ---------

    def report_item(
        self,
        src_image_path: Path,
        description: str,
        location: Optional[str] = None,
        reporter: Optional[str] = None,
        found_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Logika PENEMU BARANG:
        - src_image_path : path gambar yang diupload (sementara / dari UI)
        - description    : deskripsi barang
        - location       : lokasi ditemukan (opsional)
        - reporter       : nama pelapor (opsional)
        - found_at       : waktu ditemukan (opsional, default: sekarang)

        Returns dict berisi info item yang tersimpan.
        """
        src_image_path = src_image_path.resolve()
        if not src_image_path.exists():
            raise FileNotFoundError(f"Source image not found: {src_image_path}")

        if found_at is None:
            found_at = datetime.now()

        # 1) Simpan gambar ke folder upload_dir dengan nama yang rapi
        dest_name = src_image_path.name
        dest_path = (self.config.upload_dir / dest_name).resolve()
        if src_image_path != dest_path:
            shutil.copy2(src_image_path, dest_path)

        rel_image_path = dest_path.relative_to(self.config.root_dir)

        # 2) YOLO crop untuk keperluan embedding (kalau aktif)
        image_for_clip = dest_path
        if self.yolo_cropper is not None:
            try:
                crop_dir = self.config.upload_dir / "crops"
                crop_paths = self.yolo_cropper.crop_image(
                    image_path=dest_path,
                    save_dir=crop_dir,
                )
                if crop_paths:
                    image_for_clip = crop_paths[0]
                    print(
                        f"[FinderService][YOLO] using crop for embedding: "
                        f"{image_for_clip.relative_to(self.config.root_dir)}"
                    )
            except Exception as e:
                print(f"[FinderService][YOLO] error, fallback ke gambar asli: {e}")

        # 3) Hitung embedding TEXT (bisa tambahkan lokasi di deskripsi)
        full_text = description
        if location:
            full_text = f"{description}, ditemukan di {location}"

        text_emb = encode_text(full_text, self.model, self.processor, self.device)

        # Pastikan bentuknya [1, D], bukan [D]
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)

        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # 4) Update index .pt
        index = self._load_index()
        old_embs = index["embeddings"]
        image_paths = index["image_paths"]
        texts = index["texts"]

        if old_embs is None:
            new_embs = text_emb
        else:
            new_embs = torch.cat([old_embs, text_emb], dim=0)

        image_paths.append(str(rel_image_path).replace("\\", "/"))
        texts.append(full_text)

        self._save_index(new_embs, image_paths, texts)

        # 5) Insert row ke Postgres
        db = SessionLocal()
        try:
            db_item = FoundItem(
                image_path=str(rel_image_path).replace("\\", "/"),
                description=full_text,
                location=location,
                found_at=found_at,
                reporter=reporter,
            )
            db.add(db_item)
            db.commit()
            db.refresh(db_item)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        result = {
            "id": db_item.id,
            "image_path": db_item.image_path,
            "description": db_item.description,
            "location": db_item.location,
            "found_at": db_item.found_at.isoformat() if db_item.found_at else None,
            "reporter": db_item.reporter,
        }

        print(f"[FinderService] New item reported: {result}")
        return result
