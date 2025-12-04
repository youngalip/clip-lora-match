# src/db/models.py

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, Text, DateTime

from .database import Base


class FoundItem(Base):
    __tablename__ = "found_items"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(Text, nullable=False)      # path relatif gambar di disk
    description = Column(Text, nullable=False)     # deskripsi lengkap
    location = Column(Text, nullable=True)
    found_at = Column(DateTime, nullable=True)
    reporter = Column(Text, nullable=True)


def init_db():
    """
    Opsional: untuk membuat tabel berdasarkan model ini.
    (Tabel kamu sudah ada, tapi fungsi ini bisa dipanggil kalau mau sync skema lain.)
    """
    from .database import engine
    Base.metadata.create_all(bind=engine)
