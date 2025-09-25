# src/train.py
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import get_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAVED_DIR = ROOT / "saved_models"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_tf, val_tf

def ensure_folder_split_from_raw():
    """
    If data/train and data/val do not exist but data/raw has class folders,
    create train/ and val/ folders by splitting raw images 80/20.
    """
    raw = DATA_DIR / "raw"
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    # Already exists
    if train_dir.exists() and val_dir.exists():
        return True

    if not raw.exists():
        print("❌ No raw folder found at data/raw")
        return False

    import shutil
    from sklearn.model_selection import train_test_split

    for cls in [p for p in raw.iterdir() if p.is_dir()]:
        imgs = [p for p in cls.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]]
        if not imgs:
            continue
        train_imgs, val_imgs = train_test_split(imgs, test_size=0.2, random_state=42)
        (train_dir / cls.name).mkdir(parents=True, exist_ok=True)
        (val_dir / cls.name).mkdir(parents=True, exist_ok=True)
        for p in train_imgs:
            shutil.copy(p, train_dir / cls.name / p.name)
        for p in val_imgs:
            shutil.copy(p, val_dir / cls.name / p.name)

    print("✅ Created data/train and data/val by splitting data/raw (80/20).")
    return True

def train_model(epochs=5, batch_size=16, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Using device:", device)

    ok = ensure_folder_split_from_raw()
    if not ok:
        return

    train_tf, val_tf = get_transforms()
    train_folder = DATA_DIR / "train"
    val_folder = DATA_DIR / "val"

    train_ds = ImageFolder(root=str(train_folder), transform=train_tf)
    val_ds = ImageFolder(root=str(val_folder), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(num_classes=len(train_ds.classes), pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / max(1, len(train_ds))

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = SAVED_DIR / "best_model.pth"
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes
            }, save_path)
            print(f"💾 Saved best model to {save_path}")

    print("✅ Training finished. Best val acc:", best_acc)

if __name__ == "__main__":
    train_model(epochs=5, batch_size=16)
