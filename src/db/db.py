# src/db/db.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import psycopg2
import yaml


def load_db_config(config_path: str = "config/db_config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"DB config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("postgres", cfg)  # fleksibel: postgres: {...} atau langsung


def get_connection(config_path: str = "config/db_config.yaml"):
    cfg = load_db_config(config_path)
    conn = psycopg2.connect(
        host=cfg.get("host", "localhost"),
        port=cfg.get("port", 5432),
        user=cfg["user"],
        password=cfg["password"],
        dbname=cfg["dbname"],
    )
    return conn
