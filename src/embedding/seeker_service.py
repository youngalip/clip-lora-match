# src/embedding/seeker_service.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from models.clip_model import load_clip_model, encode_text, encode_image
from src.embedding.search import TextSearchIndex, SearchResult
from src.preprocessing.yolo_cropper import YoloCropper


@dataclass
class SeekerConfig:
    """
    Konfigurasi untuk mode 'orang kehilangan' (seeker).

    - root_dir         : root project (misal: .../ml-service)
    - clip_config_path : config/clip_config.yaml
    - lora_dir         : folder weights LoRA (misal: models/saved/clip-lora/epoch_1)
    - index_path       : index barang ditemukan (misal: data/index/custom_items_index.pt)
    - yolo_config_path : config YOLO (optional, kalau None berarti YOLO dimatikan)
    - yolo_crop_dir    : folder untuk menyimpan hasil crop query (optional)
    """
    root_dir: Path
    clip_config_path: Path
    lora_dir: Path
    index_path: Path
    yolo_config_path: Optional[Path] = None
    yolo_crop_dir: Optional[Path] = None


class SeekerService:
    """
    Layanan pencarian untuk ORANG KEHILANGAN (search only).

    - TIDAK menyimpan apa pun ke database
    - Hanya membaca index barang yang SUDAH dilaporkan ditemukan
    - Bisa search dengan:
      - teks saja
      - gambar saja
      - teks + gambar (fusion sederhana)
    - Jika YOLO dikonfigurasi, query image akan di-crop dulu sebelum di-encode.
    """

    def __init__(self, config: SeekerConfig) -> None:
        self.config = config

        # Load CLIP+LoRA sekali di awal
        self.model, self.processor, self.device = load_clip_model(
            config_path=self.config.clip_config_path,
            use_lora=True,
            lora_weights_path=self.config.lora_dir,
        )

        # Load index barang ditemukan
        self.index = TextSearchIndex(self.config.index_path)

        # Setup YOLO cropper (opsional)
        self.yolo_cropper: Optional[YoloCropper] = None
        self.yolo_crop_dir: Optional[Path] = None

        if self.config.yolo_config_path is not None:
            yolo_cfg_path = self.config.yolo_config_path
            if not yolo_cfg_path.exists():
                print(f"[SeekerService] YOLO config tidak ditemukan: {yolo_cfg_path}, lanjut tanpa YOLO.")
            else:
                print(f"[SeekerService] Mengaktifkan YOLO cropper dengan config: {yolo_cfg_path}")
                self.yolo_cropper = YoloCropper(config_path=yolo_cfg_path)

                # Folder untuk simpan crop query
                if self.config.yolo_crop_dir is not None:
                    self.yolo_crop_dir = self.config.yolo_crop_dir
                else:
                    # default: simpan di data/custom/query_crops
                    self.yolo_crop_dir = (
                        self.config.root_dir / "data" / "custom" / "query_crops"
                    )
                self.yolo_crop_dir.mkdir(parents=True, exist_ok=True)

    def _build_query_embedding(
        self,
        query_text: Optional[str],
        query_image_path: Optional[Path],
        w_text: float = 0.5,
        w_image: float = 0.5,
    ) -> torch.Tensor:
        """
        Bangun embedding query dari:
        - hanya teks
        - hanya gambar
        - teks + gambar (fusion sederhana).
        Jika YOLO diaktifkan, query image akan di-crop dulu.
        """
        have_text = query_text is not None and query_text.strip() != ""
        have_image = query_image_path is not None

        if not have_text and not have_image:
            raise ValueError("Minimal harus ada query_text atau query_image_path.")

        embs = []

        # ---- TEXT PART ----
        if have_text:
            txt_emb = encode_text(
                query_text,
                self.model,
                self.processor,
                self.device,
            )
            embs.append((txt_emb, w_text))

        # ---- IMAGE PART (dengan YOLO jika ada) ----
        if have_image:
            img_path_for_clip = query_image_path

            # Kalau YOLO aktif, pakai crop
            if self.yolo_cropper is not None and self.yolo_crop_dir is not None:
                try:
                    crop_paths = self.yolo_cropper.crop_image(
                        image_path=query_image_path,
                        save_dir=self.yolo_crop_dir,
                    )
                    if crop_paths:
                        crop_path = crop_paths[0]
                        # Logging sederhana: before/after supaya kelihatan di CLI
                        rel_before = query_image_path.relative_to(self.config.root_dir)
                        rel_after = crop_path.relative_to(self.config.root_dir)
                        print("\n[SeekerService][YOLO CROP]")
                        print(f"  Before: {rel_before}")
                        print(f"  After : {rel_after}")
                        print()
                        img_path_for_clip = crop_path
                except Exception as e:
                    print(f"[SeekerService] YOLO crop error, fallback ke gambar asli: {e}")

            img_emb = encode_image(
                img_path_for_clip,
                self.model,
                self.processor,
                self.device,
            )
            embs.append((img_emb, w_image))

        # Kalau cuma satu modalitas → langsung pakai
        if len(embs) == 1:
            emb, _ = embs[0]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb

        # Kalau dua-duanya (text + image) → fusion sederhana
        weighted = sum(w * e for e, w in embs)
        weighted = weighted / weighted.norm(dim=-1, keepdim=True)
        return weighted

    def search_items(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Pencarian barang ditemukan berdasarkan:
        - teks saja
        - gambar saja (akan di-YOLO crop dulu kalau aktif)
        - teks + gambar sekaligus.
        """
        img_path: Optional[Path] = None
        if query_image_path:
            img_path = (self.config.root_dir / query_image_path).resolve()
            if not img_path.exists():
                raise FileNotFoundError(f"Query image not found: {img_path}")

        # Bangun embedding query gabungan
        query_emb = self._build_query_embedding(
            query_text=query_text,
            query_image_path=img_path,
        )

        self.index = TextSearchIndex(self.config.index_path)

        # Pakai index.search_with_embedding (low-level)
        return self.index.search_with_embedding(query_emb, top_k=top_k)
