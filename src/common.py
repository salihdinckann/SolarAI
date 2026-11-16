"""Eğitim, Grad-CAM ve Streamlit uygulamasının ortak kullandığı parçalar."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]

# Sınıf sırası SABİT: modelin çıkış nöronları bu sıraya göre.
CLASSES = [
    "clean",
    "dusty",
    "bird_drop",
    "snow_covered",
    "physical_damage",
    "electrical_damage",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Ekranda/posterde gösterilecek Türkçe adlar
CLASS_TR = {
    "clean": "Temiz",
    "dusty": "Tozlu",
    "bird_drop": "Kuş pisliği",
    "snow_covered": "Kar",
    "physical_damage": "Fiziksel hasar",
    "electrical_damage": "Elektriksel hasar",
}

# ImageNet ortalama/std: önceden eğitilmiş ağırlıklar bu normalizasyonla eğitildi
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(name: str = "efficientnet_b0", pretrained: bool = True) -> torch.nn.Module:
    """ImageNet'te eğitilmiş EfficientNet'i alır, son katmanı 6 sınıfa göre değiştirir."""
    model = timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))
    # Yeni son katmanı sıfırla: başlangıçta tüm sınıflar eşit olası (kayıp = ln 6),
    # rastgele büyük ağırlıklarla "emin ama yanlış" başlamaz.
    head = model.get_classifier()
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return model


def train_transform(img_size: int) -> transforms.Compose:
    # Veri artırma: her epoch'ta görüntünün biraz farklı bir hali gösterilir -> ezber azalır
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(img_size: int) -> transforms.Compose:
    # Değerlendirmede rastgelelik yok: her seferinde aynı sonuç
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
