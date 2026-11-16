"""Eğitilmiş modelle tahmin ve Grad-CAM ısı haritası.

Streamlit uygulaması bu modülü kullanır:
    pred = Predictor("outputs/effnet_b0_v1/best.pt")
    probs = pred.predict(pil_image)                  # {"clean": 0.91, "dusty": 0.05, ...}
    heat  = pred.gradcam(pil_image)                  # 224x224, 0-1 arası
    img   = overlay(pred.preprocess_view(pil_image), heat)

Grad-CAM (Selvaraju ve ark., 2017): Son evrişim katmanının çıktılarını, seçilen sınıfın
skoruna göre gradyanlarıyla ağırlıklandırır. Sonuç, modelin o sınıfa karar verirken
görüntünün HANGİ BÖLGESİNE dayandığını gösteren kaba bir haritadır.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms

from common import CLASSES, build_model, eval_transform


class Predictor:
    def __init__(self, ckpt_path: str | Path, device: str | None = None, cam_layer: str = "bn2"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if ckpt["classes"] != CLASSES:
            raise ValueError(f"Checkpoint sınıf sırası farklı: {ckpt['classes']}")
        self.img_size = ckpt["img_size"]
        self.model = build_model(ckpt["model"], pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = eval_transform(self.img_size)
        # Grad-CAM için hedef katman: sınıflandırıcıdan önceki son özellik haritası
        # (timm EfficientNet'te conv_head -> bn2 (+aktivasyon)); çıktısı 7x7x1280
        self.target_layer = self.model.get_submodule(cam_layer)

    def preprocess_view(self, image: Image.Image) -> Image.Image:
        """Modelin gördüğü kırpılmış görüntü (ısı haritası bununla hizalı)."""
        view = transforms.Compose([
            transforms.Resize(int(self.img_size * 1.14)),
            transforms.CenterCrop(self.img_size),
        ])
        return view(image.convert("RGB"))

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, float]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return {c: float(p) for c, p in zip(CLASSES, probs)}

    def gradcam(self, image: Image.Image, class_name: str | None = None) -> np.ndarray:
        """class_name verilmezse en olası sınıf için haritayı üretir. Çıktı: (H, W), 0-1."""
        store = {}

        def fwd_hook(_m, _i, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = self.target_layer.register_forward_hook(fwd_hook)
        try:
            x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_name is None else CLASSES.index(class_name)
            logits[0, idx].backward()
        finally:
            handle.remove()

        act, grad = store["act"][0], store["grad"][0]          # (C, h, w)
        weights = grad.mean(dim=(1, 2))                         # her kanalın önemi
        cam = F.relu((weights[:, None, None] * act).sum(0))     # (h, w), sadece pozitif katkı
        cam = F.interpolate(cam[None, None], size=(self.img_size, self.img_size),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def overlay(image: Image.Image, heat: np.ndarray, alpha: float = 0.6, threshold: float = 0.3) -> Image.Image:
    """Isı haritasını görüntünün üstüne bindirir.

    Saydamlık önemle orantılıdır: önemi `threshold` altındaki bölgeler hiç boyanmaz,
    böylece görüntünün tamamı maviye bürünmez, sadece kararı etkileyen bölge renklenir.
    """
    base = np.asarray(image.convert("RGB").resize(heat.shape[::-1]), dtype=np.float32) / 255
    color = colormaps["turbo"](heat)[..., :3]
    a = alpha * np.clip((heat - threshold) / (1 - threshold), 0, 1)[..., None]
    mixed = (1 - a) * base + a * color
    return Image.fromarray((mixed * 255).clip(0, 255).astype(np.uint8))
