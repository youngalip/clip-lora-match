# models/yolo_model.py

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from ultralytics import YOLO


def _load_yolo_config(config_path: Union[str, Path]) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yolo_model(
    config_path: Union[str, Path] = "config/yolo_config.yaml",
) -> YOLO:
    """
    Load YOLOv8 model sesuai config.
    """
    cfg = _load_yolo_config(config_path)
    model_cfg = cfg.get("model", {})

    weights_path = model_cfg.get("weights_path")
    name = model_cfg.get("name", "yolov8s")
    device = model_cfg.get("device", "cpu")

    if weights_path and Path(weights_path).exists():
        model = YOLO(weights_path)
    else:
        model = YOLO(name)

    model.to(device)
    return model
