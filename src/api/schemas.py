# src/api/schemas.py

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ReportItemResponse(BaseModel):
    id: int
    image_path: str
    description: str
    location: Optional[str] = None
    found_at: Optional[datetime] = None
    reporter: Optional[str] = None


class SearchResultModel(BaseModel):
    score: float
    image_path: str
    text: str


class SearchResponse(BaseModel):
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    results: List[SearchResultModel]

class FoundItemModel(BaseModel):
    id: int
    image_path: str
    description: str
    location: Optional[str] = None
    found_at: datetime
    reporter: Optional[str] = None