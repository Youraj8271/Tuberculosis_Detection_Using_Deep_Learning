# src/preprocess.py
"""
Creates CSV listing images and labels from data/raw folder.
Also creates train/val/test splits (CSV).
"""
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

def create_labels_csv(out_csv=DATA_DIR / "labels.csv"):
    rows = []
    if not RAW_DIR.exists():
        print(f"{RAW_DIR} not found. Put images in data/raw/<ClassName>/")
        return None
    for label_dir in RAW_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img in label_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                rows.append({"image": str(img.resolve()), "label": label})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved labels CSV to {out_csv}")
    return df

def create_splits(labels_csv=DATA_DIR / "labels.csv", out_dir=DATA_DIR, test_size=0.2, val_size=0.5, random_state=42):
    if not Path(labels_csv).exists():
        print("labels.csv not found. Run create_labels_csv() first.")
        return None
    df = pd.read_csv(labels_csv)
    train, temp = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=random_state)
    val, test = train_test_split(temp, test_size=val_size, stratify=temp["label"], random_state=random_state)
    train.to_csv(Path(out_dir) / "train.csv", index=False)
    val.to_csv(Path(out_dir) / "val.csv", index=False)
    test.to_csv(Path(out_dir) / "test.csv", index=False)
    print(f"Created train/val/test CSVs in {out_dir}")
    return train, val, test

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = create_labels_csv()
    if df is not None and not df.empty:
        create_splits()
