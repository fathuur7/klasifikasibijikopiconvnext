from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
from src.config import resolve_weight

# ==============================================================================
# 1. KONSTRUKSI ARSITEKTUR CONVNEXT DENGAN CUSTOM HEAD
# ==============================================================================
def build_convnext_model(num_classes: int, backend: str = "timm", custom_head: bool = True) -> nn.Module:
    """Membangun arsitektur ConvNeXt-Tiny dengan Sequential classification head."""
    if backend == "torchvision":
        from torchvision.models import convnext_tiny
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    import timm
    m = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes, drop_rate=0.3)
    if custom_head:
        in_features = m.head.fc.in_features
        m.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    return m

def extract_state_dict(ckpt):
    """Mengekstrak state_dict murni dari berbagai format checkpoint PyTorch."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return {k.replace("module.", "", 1): v for k, v in ckpt.items()}

# ==============================================================================
# 2. PEMUATAN MODEL TEROPTIMASI DENGAN STREAMLIT CACHING
# ==============================================================================
@st.cache_resource(show_spinner=False, max_entries=4)
def load_model(
    file_name: str,
    num_classes: int,
    device_str: str,
    repo_id: str = "",
    token: str | None = None
) -> nn.Module:
    """Memuat model PyTorch secara efisien dari Hugging Face atau disk lokal."""
    path = resolve_weight(file_name, repo_id=repo_id, token=token)
    try:
        ckpt = torch.load(path, map_location=device_str, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device_str, weights_only=False)

    if isinstance(ckpt, nn.Module):
        return ckpt.to(device_str).eval()

    state = extract_state_dict(ckpt)
    has_custom_head = any("head.fc.0" in k for k in state.keys())
    backends_to_try = [("timm", True)] if has_custom_head else [("timm", False), ("torchvision", False), ("timm", True)]

    last_err = None
    for backend, is_custom in backends_to_try:
        try:
            model = build_convnext_model(num_classes, backend=backend, custom_head=is_custom)
            model.load_state_dict(state, strict=True)
            return model.to(device_str).eval()
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Gagal memuat {file_name}. Pastikan arsitektur sesuai (torchvision/timm convnext_tiny). "
        f"Error terakhir: {last_err}"
    )

@torch.no_grad()
def calc_probabilities(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Menghitung probabilitas softmax dari tensor citra."""
    return torch.softmax(model(x), dim=1)[0].cpu()
