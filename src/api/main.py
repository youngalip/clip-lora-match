# src/api/main.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, List

import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # ⬅️ NEW

from src.api.schemas import (
    ReportItemResponse,
    SearchResponse,
    SearchResultModel,
    FoundItemModel,  # ⬅️ NEW
)
from src.embedding.finder_service import FinderConfig, FinderService
from src.embedding.seeker_service import SeekerConfig, SeekerService
from src.embedding.search import SearchResult  # tipe hasil internal

from src.db.db import get_connection

# -----------------------------------------
# Setup root & config global
# -----------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

CLIP_CONFIG = ROOT_DIR / "config" / "clip_config.yaml"
LORA_DIR = ROOT_DIR / "models" / "saved" / "clip-lora" / "epoch_5"
INDEX_PATH = ROOT_DIR / "data" / "index" / "custom_items_index.pt"
YOLO_CONFIG = None
#YOLO_CONFIG = ROOT_DIR / "config" / "yolo_config.yaml"

REPORTED_IMAGES_DIR = ROOT_DIR / "data" / "reported" / "images"
REPORTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# Inisialisasi service sekali (dipakai semua request)
# -----------------------------------------

finder_config = FinderConfig(
    root_dir=ROOT_DIR,
    clip_config_path=CLIP_CONFIG,
    lora_dir=LORA_DIR,
    index_path=INDEX_PATH,
    upload_dir=REPORTED_IMAGES_DIR,
    yolo_config_path=YOLO_CONFIG,
)
finder_service = FinderService(finder_config)

seeker_config = SeekerConfig(
    root_dir=ROOT_DIR,
    clip_config_path=CLIP_CONFIG,
    lora_dir=LORA_DIR,
    index_path=INDEX_PATH,
    yolo_config_path=YOLO_CONFIG,
    yolo_crop_dir=ROOT_DIR / "data" / "custom" / "query_crops",
)
seeker_service = SeekerService(seeker_config)

# -----------------------------------------
# FastAPI app
# -----------------------------------------

app = FastAPI(
    title="Balikkin ML Service",
    version="0.1.0",
)

# CORS (kalau nanti front-end beda origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # bisa dibatasi nanti
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files: serve semua isi folder data/ di bawah /static
# contoh: data/custom/images/...  ->  /static/custom/images/...
app.mount(
    "/static",
    StaticFiles(directory=str(ROOT_DIR / "data")),
    name="static",
)


# -----------------------------------------
# Endpoint: health check
# -----------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------------------
# Endpoint: laporan barang ditemukan (penemu)
# -----------------------------------------
@app.post("/api/report", response_model=ReportItemResponse)
async def report_item(
    description: str = Form(...),
    location: Optional[str] = Form(None),
    reporter: Optional[str] = Form(None),
    found_at: Optional[str] = Form(None),  # ISO string opsional
    image: UploadFile = File(...),
):
    """
    Penemu mengupload:
    - image (file)
    - description (teks)
    - location, reporter, found_at (opsional)

    Menghasilkan:
    - row baru di Postgres
    - embedding baru di index .pt
    """
    # 1) Simpan file sementara di disk
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diupload harus gambar.")

    temp_dir = ROOT_DIR / "data" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / image.filename
    with temp_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    # 2) Parse found_at jika ada
    found_at_dt: Optional[datetime] = None
    if found_at:
        try:
            found_at_dt = datetime.fromisoformat(found_at)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Format found_at harus ISO 8601, misal: 2025-12-04T21:04:23",
            )

    # 3) Panggil FinderService
    try:
        result_dict = finder_service.report_item(
            src_image_path=temp_path,
            description=description,
            location=location,
            reporter=reporter,
            found_at=found_at_dt,
        )
    finally:
        # optional: hapus file temp
        if temp_path.exists():
            temp_path.unlink()

    # 4) Convert ke Pydantic response
    return ReportItemResponse(
        id=result_dict["id"],
        image_path=result_dict["image_path"],
        description=result_dict["description"],
        location=result_dict["location"],
        found_at=datetime.fromisoformat(result_dict["found_at"])
        if result_dict["found_at"]
        else None,
        reporter=result_dict["reporter"],
    )


# -----------------------------------------
# Endpoint: search (orang kehilangan)
# -----------------------------------------
@app.post("/api/search", response_model=SearchResponse)
async def search_items(
    description: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(5),
):
    """
    Orang kehilangan bisa:
    - hanya isi description
    - hanya upload image
    - atau keduanya (description + image)
    """

    # --- Normalisasi description: kosong -> None ---
    if description is not None:
        description = description.strip()
        if description == "":
            description = None

    # --- Normalisasi image: kalau di Swagger dikirim tapi tidak pilih file ---
    if image is not None and (image.filename is None or image.filename == ""):
        image = None

    # Minimal harus ada salah satu
    if description is None and image is None:
        raise HTTPException(
            status_code=400,
            detail="Minimal isi description atau upload image.",
        )

    query_image_rel_path: Optional[str] = None
    temp_path: Optional[Path] = None

    # 1) Simpan gambar query (kalau ada)
    if image is not None:
        if image.content_type is None or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File yang diupload harus gambar.")

        query_dir = ROOT_DIR / "data" / "tmp" / "queries"
        query_dir.mkdir(parents=True, exist_ok=True)
        temp_path = query_dir / image.filename

        with temp_path.open("wb") as f:
            shutil.copyfileobj(image.file, f)

        # path relatif dari ROOT_DIR, supaya cocok dengan SeekerService
        query_image_rel_path = str(temp_path.relative_to(ROOT_DIR)).replace("\\", "/")

    # 2) Panggil SeekerService
    try:
        results_internal: List[SearchResult] = seeker_service.search_items(
            query_text=description,
            query_image_path=query_image_rel_path,
            top_k=top_k,
        )
    except Exception as e:
        # Untuk debug internal, bisa print error
        print(f"[api/search] ERROR: {e}")
        raise HTTPException(status_code=500, detail="Internal search error")
    finally:
        # optional: hapus file query
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    # 3) Convert ke response model
    results_out = [
        SearchResultModel(
            score=r.score,
            image_path=r.image_path,
            text=r.text,
        )
        for r in results_internal
    ]

    return SearchResponse(
        query_text=description,
        query_image_path=query_image_rel_path,
        results=results_out,
    )


# -----------------------------------------
# Endpoint: list semua barang ditemukan (dari DB)
# -----------------------------------------
@app.get("/api/items", response_model=List[FoundItemModel])
def list_found_items():
    """
    List semua barang yang sudah dilaporkan penemu (tabel found_items).
    Dipakai untuk:
    - dashboard admin/petugas
    - cek isi database dari luar
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, image_path, description, location, found_at, reporter
            FROM found_items
            ORDER BY found_at DESC
            """
        )
        rows = cur.fetchall()
    except Exception as e:
        print(f"[api/items] DB error: {e}")
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

    return [
        FoundItemModel(
            id=row[0],
            image_path=row[1],
            description=row[2],
            location=row[3],
            found_at=row[4],
            reporter=row[5],
        )
        for row in rows
    ]
