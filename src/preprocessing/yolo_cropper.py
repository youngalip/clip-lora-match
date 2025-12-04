# src/preprocessing/yolo_cropper.py

from pathlib import Path
from typing import List, Optional, Union

import yaml
from ultralytics import YOLO
from PIL import Image

import torch

class YoloCropper:
    """
    Wrapper sederhana untuk YOLOv8 sebagai object cropper.
    Fokus: deteksi objek dan simpan hasil crop ke folder tertentu.
    """

    def __init__(self, config_path: Union[str, Path] = "config/yolo_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        model_cfg = self.config.get("model", {})
        weights_path = model_cfg.get("weights_path")
        device = model_cfg.get("device", "cpu")

        # Fallback otomatis kalau minta cuda tapi tidak tersedia
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[YoloCropper] CUDA tidak tersedia, fallback ke CPU.")
            device = "cpu"

        self.device = device

        # Load YOLO model (bisa pakai path .pt atau nama model seperti 'yolov8s')
        if weights_path and Path(weights_path).exists():
            self.model = YOLO(weights_path)
        else:
            # fallback ke model name (misal: "yolov8s")
            name = model_cfg.get("name", "yolov8s")
            self.model = YOLO(name)

        self.model.to(device)

        self.infer_cfg = self.config.get("inference", {})
        self.crop_cfg = self.config.get("crop", {})

        # Setup direktori simpan crop
        self.save_dir = Path(self.crop_cfg.get("save_dir", "data/cropped"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Pola nama file crop
        self.filename_pattern = self.crop_cfg.get(
            "filename_pattern", "{stem}_crop_{idx}.jpg"
        )

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"YOLO config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def crop_image(
        self,
        image_path: Union[str, Path],
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """
        Deteksi objek pada satu gambar dan simpan crop hasil deteksi.

        Returns:
            List[Path]: daftar path file crop yang dihasilkan.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if save_dir is None:
            save_dir = self.save_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        results = self.model.predict(
            source=str(image_path),
            conf=self.infer_cfg.get("conf_threshold", 0.25),
            iou=self.infer_cfg.get("iou_threshold", 0.45),
            max_det=self.infer_cfg.get("max_det", 5),
            classes=self.infer_cfg.get("classes", None),
            agnostic_nms=self.infer_cfg.get("agnostic_nms", False),
            imgsz=self.model.model.args.get("imgsz", 640),
            verbose=False,
        )

        # Biasanya satu gambar → satu Results
        res = results[0]
        boxes = res.boxes

        crop_paths: List[Path] = []

        # Load original image sekali saja
        orig_img = Image.open(image_path).convert("RGB")
        width, height = orig_img.size

        stem = image_path.stem

        for idx, box in enumerate(boxes):
            # xyxy format: [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            # Pastikan koordinat dalam range gambar
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            crop = orig_img.crop((x1, y1, x2, y2))

            filename = self.filename_pattern.format(stem=stem, idx=idx)
            crop_path = save_dir / filename
            crop.save(crop_path)
            crop_paths.append(crop_path)

        # Kalau nggak ada deteksi, opsional: simpan full image sebagai crop
        if not crop_paths:
            filename = self.filename_pattern.format(stem=stem, idx=0)
            crop_path = save_dir / filename
            orig_img.save(crop_path)
            crop_paths.append(crop_path)

        return crop_paths

    def crop_folder(
        self,
        input_dir: Union[str, Path],
        save_dir: Optional[Union[str, Path]] = None,
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Crop semua gambar dalam folder.

        Args:
            input_dir: folder berisi gambar.
            save_dir: folder output crop (default dari config).
            extensions: list ekstensi yang diterima (default: jpg/jpeg/png).

        Returns:
            List[Path]: semua path crop yang dihasilkan.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if save_dir is None:
            save_dir = self.save_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]

        all_crops: List[Path] = []

        for img_path in input_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                crop_paths = self.crop_image(img_path, save_dir=save_dir)
                all_crops.extend(crop_paths)

        return all_crops
