# src/utils.py
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    classes = ckpt.get("classes", None)
    return model, classes

def preprocess_image(image, size=(224,224)):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    return tf(img).unsqueeze(0)
