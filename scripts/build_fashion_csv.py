from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Root project = folder "ml-service"
    root = Path(__file__).resolve().parents[1]

    # Sesuaikan kalau lokasi kamu beda
    styles_path = root / "data" / "external" / "fashion" / "styles.csv"
    images_dir = root / "data" / "external" / "fashion" / "images"
    out_dir = root / "data" / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build_fashion_csv] styles.csv: {styles_path}")
    print(f"[build_fashion_csv] images dir: {images_dir}")

    if not styles_path.exists():
        raise FileNotFoundError(f"styles.csv not found at {styles_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found at {images_dir}")

    # Baca styles.csv
    df = pd.read_csv(styles_path, on_bad_lines="skip")

    # Pastikan kolom penting ada
    required_cols = [
        "id",
        "productDisplayName",
        "baseColour",
        "articleType",
        "gender",
        "masterCategory",
        "subCategory",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in styles.csv")

    # Buang baris yang kolom pentingnya kosong
    df = df.dropna(subset=required_cols)

    # Buat path absolut ke gambar berdasarkan id
    def make_image_path(id_value: int | str) -> Path:
        return images_dir / f"{id_value}.jpg"

    df["image_abs"] = df["id"].apply(make_image_path)

    # Filter hanya baris yang file gambarnya benar-benar ada
    df = df[df["image_abs"].apply(lambda p: p.exists())].reset_index(drop=True)
    print(f"[build_fashion_csv] Rows with existing images: {len(df)}")

    # Bangun text deskripsi untuk CLIP
    def build_text(row) -> str:
        # contoh format, bisa kamu modif nanti
        return (
            f"{row['productDisplayName']}, "
            f"{row['baseColour']} {row['articleType']} for {row['gender']}, "
            f"category {row['masterCategory']}/{row['subCategory']}"
        )

    df["text"] = df.apply(build_text, axis=1)

    # Simpan image_path sebagai path relatif terhadap root project
    # misal: data/external/fashion/images/12345.jpg
    df["image_path"] = df["image_abs"].apply(
        lambda p: p.relative_to(root).as_posix()
    )

    final_df = df[["image_path", "text"]]

    # Split train/val (90% / 10%)
    train_df, val_df = train_test_split(
        final_df,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    train_path = out_dir / "train_fashion.csv"
    val_path = out_dir / "val_fashion.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[build_fashion_csv] Saved train: {train_path}  ({len(train_df)} rows)")
    print(f"[build_fashion_csv] Saved val  : {val_path}  ({len(val_df)} rows)")


if __name__ == "__main__":
    main()
