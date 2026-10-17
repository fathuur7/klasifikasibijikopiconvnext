from PIL import Image
import torch
from torchvision import transforms
from src.config import VARIETAS, SANGRAI, FLAT_CLASSES, IMG_SIZE, MEAN, STD
from src.model import load_model, calc_probabilities

# Transformasi pra-pemrosesan citra standar ConvNeXt
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

def predict_cascade(
    img: Image.Image,
    fold: int,
    device: str,
    repo_id: str = "",
    token: str | None = None
) -> dict:
    """Inferensi Hierarchical CASCADE 2-Tahap: Tahap 1 Varietas -> Tahap 2 Spesialis Sangrai."""
    x = TRANSFORM(img).unsqueeze(0).to(device)

    # Tahap 1: Model Gerbang Varietas
    m1 = load_model(f"ConvNeXt_Jenis_Fold{fold}.pth", len(VARIETAS), device, repo_id=repo_id, token=token)
    p1 = calc_probabilities(m1, x)
    i1 = int(p1.argmax())
    varietas = VARIETAS[i1]

    # Tahap 2: Model Spesialis Sangrai sesuai Varietas Terpilih
    m2 = load_model(f"ConvNeXt_{varietas}_Fold{fold}.pth", len(SANGRAI), device, repo_id=repo_id, token=token)
    p2 = calc_probabilities(m2, x)
    i2 = int(p2.argmax())
    sangrai = SANGRAI[i2]

    return {
        "varietas": varietas,
        "conf_varietas": float(p1[i1]),
        "probs_varietas": p1.numpy(),
        "sangrai": sangrai,
        "conf_sangrai": float(p2[i2]),
        "probs_sangrai": p2.numpy(),
        "kelas": f"{varietas}_{sangrai}",
        "joint": float(p1[i1] * p2[i2]),
    }

def predict_flat(
    img: Image.Image,
    fold: int,
    device: str,
    repo_id: str = "",
    token: str | None = None
) -> dict:
    """Inferensi Single-Stage FLAT: 16 Kelas sekaligus."""
    x = TRANSFORM(img).unsqueeze(0).to(device)
    m = load_model(f"ConvNeXt_FLAT_Fold{fold}.pth", len(FLAT_CLASSES), device, repo_id=repo_id, token=token)
    p = calc_probabilities(m, x)
    i = int(p.argmax())
    return {
        "kelas": FLAT_CLASSES[i],
        "conf": float(p[i]),
        "probs": p.numpy()
    }
