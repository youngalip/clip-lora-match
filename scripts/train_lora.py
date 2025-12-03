# scripts/train_lora.py

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
import yaml

from datasets.dataset import ClipPairDataset
from models.lora_adapter import create_lora_config, attach_lora_to_clip


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lora_training_config(config_path: str | Path = "config/lora_config.yaml") -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRA config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg.get("data", {})
    train_csv = data_cfg.get("train_csv")
    val_csv = data_cfg.get("val_csv")
    image_root_dir = data_cfg.get("image_root_dir", ".")

    if train_csv is None or val_csv is None:
        raise ValueError("Please set 'train_csv' and 'val_csv' in lora_config.yaml under 'data'.")

    train_dataset = ClipPairDataset(
        csv_path=train_csv,
        image_root_dir=image_root_dir,
        use_augmentation=True,
        clip_config_path="config/clip_config.yaml",
    )

    val_dataset = ClipPairDataset(
        csv_path=val_csv,
        image_root_dir=image_root_dir,
        use_augmentation=False,
        clip_config_path="config/clip_config.yaml",
    )

    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def compute_clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Implementasi sederhana CLIP contrastive loss (image<->text).
    image_features: (N, d)
    text_features : (N, d)
    """

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = (image_features @ text_features.T) / temperature  # (N, N)
    logits_per_text = logits_per_image.T

    batch_size = image_features.shape[0]
    targets = torch.arange(batch_size, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    loss = (loss_i + loss_t) / 2.0

    return loss


def train():
    cfg = load_lora_training_config("config/lora_config.yaml")
    train_cfg = cfg.get("training", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_lora] Using device: {device}")

    # Load base CLIP
    model_name = cfg.get("model", {}).get("base_model_name", "openai/clip-vit-base-patch32")
    print(f"[train_lora] Loading base CLIP model: {model_name}")
    base_model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    base_model.to(device)

    # Attach LoRA
    print("[train_lora] Creating LoRA config and attaching to model...")
    lora_config = create_lora_config("config/lora_config.yaml")
    model = attach_lora_to_clip(base_model, lora_config)

    # Hanya LoRA params yang trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train_lora] Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # DataLoader
    train_loader, val_loader = build_dataloaders(cfg)

    # Training hyperparams (cast biar aman)
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    num_epochs = int(train_cfg.get("num_epochs", 5))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    logging_steps = int(train_cfg.get("logging_steps", 50))

    temperature = float(train_cfg.get("temperature", 0.07))

    output_dir = Path(train_cfg.get("output_dir", "models/saved/clip-lora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Optional simple scheduler (linear warmup + decay)
    total_steps = num_epochs * math.ceil(len(train_loader) / grad_accum_steps)
    warmup_ratio = train_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        print(f"\n[train_lora] ===== Epoch {epoch + 1}/{num_epochs} =====")
        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # dtype mengikuti model
            model_dtype = next(model.parameters()).dtype
            pixel_values = pixel_values.to(dtype=model_dtype)

            # Forward pass
            image_features = model.get_image_features(pixel_values=pixel_values)
            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = compute_clip_contrastive_loss(image_features, text_features, temperature)
            loss = loss / grad_accum_steps
            loss.backward()

            running_loss += loss.item()

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    print(
                        f"[train_lora] Step {global_step}/{total_steps} "
                        f"LR={scheduler.get_last_lr()[0]:.2e} "
                        f"Loss={avg_loss:.4f}"
                    )
                    running_loss = 0.0

        # Simple validation (optional, bisa kamu kembangkan)
        model.eval()
        val_loss_total = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                pixel_values = pixel_values.to(dtype=model_dtype)

                image_features = model.get_image_features(pixel_values=pixel_values)
                text_features = model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = compute_clip_contrastive_loss(image_features, text_features, temperature)
                val_loss_total += loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            avg_val_loss = val_loss_total / num_val_batches
        else:
            avg_val_loss = float("nan")

        print(f"[train_lora] Epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")

        # Save checkpoint per-epoch
        epoch_dir = output_dir / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train_lora] Saving LoRA model to {epoch_dir} ...")
        model.save_pretrained(epoch_dir)

    print("[train_lora] Training finished.")


if __name__ == "__main__":
    train()
