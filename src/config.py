import os
import json
from pathlib import Path
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. KELAS DAN PEMETAAN LABEL RESMI (SESUAI URUTAN TRAINING)
# ==============================================================================
VARIETAS = ["Arabika", "Ekselsa", "Liberika", "Robusta"]
SANGRAI = ["Dark", "Greenbean", "Light", "Medium"]
FLAT_CLASSES = [f"{v}_{s}" for v in VARIETAS for s in SANGRAI]

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==============================================================================
# 2. PENGATURAN AUTENTIKASI HUGGING FACE SECARA DINAMIS
# ==============================================================================
def load_hf_token() -> str | None:
    """Mengambil token autentikasi Hugging Face secara dinamis dari file token, env, secrets, atau cookies."""
    # 1. Cek file hf_write_token.txt
    for tp in [
        Path(r"d:\Riset_Mbkm_2026\hf_write_token.txt"),
        Path(__file__).resolve().parent.parent / "hf_write_token.txt",
    ]:
        if tp.exists():
            val = tp.read_text(encoding="utf-8").strip()
            if val:
                return val

    # 2. Cek Environment Variable
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    # 3. Cek Streamlit Secrets
    try:
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass

    # 4. Cek cookies_huggingface.txt
    for cp in [
        Path(__file__).resolve().parent.parent / "cookies_huggingface.txt",
        Path(r"d:\Riset_Mbkm_2026\cookies_huggingface.txt"),
    ]:
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    cookies = json.load(f)
                    for c in cookies:
                        if c.get("name") == "token" and c.get("value"):
                            return c["value"]
            except Exception:
                pass
    return None

HF_DEFAULT_REPO = os.getenv("HF_REPO_ID", "turasa22/klasifikasibijikopiconvnext")
HF_DEFAULT_TOKEN = load_hf_token()

# ==============================================================================
# 3. DETEKSI PATH LOKAL & FALLBACK DIREKTORI
# ==============================================================================
LOCAL_MODEL_DIRS = [
    Path(r"d:\Riset_Mbkm_2026\MASTER_FINAL_CONVNEXT_CLASSIFIER\01_Model_Weights_Final"),
    Path(r"D:\MASTER_FINAL_CONVNEXT_CLASSIFIER\MASTER_FINAL_CONVNEXT_CLASSIFIER\01_Model_Weights_Final"),
    Path(__file__).resolve().parent.parent / "01_Model_Weights_Final",
]
MODEL_DIR = next((p for p in LOCAL_MODEL_DIRS if p.exists()), LOCAL_MODEL_DIRS[0])

LOCAL_ASSET_DIRS = [
    Path(r"d:\Riset_Mbkm_2026\MASTER_FINAL_CONVNEXT_CLASSIFIER\02_Reports_and_Metrics"),
    Path(r"D:\MASTER_FINAL_CONVNEXT_CLASSIFIER\MASTER_FINAL_CONVNEXT_CLASSIFIER\02_Reports_and_Metrics"),
    Path(__file__).resolve().parent.parent / "assets",
]
ASSETS_DIR = next((p for p in LOCAL_ASSET_DIRS if p.exists()), Path("assets"))

LOCAL_DATASET_DIRS = [
    Path(r"d:\Riset_Mbkm_2026\PROJECT_1V\Dataset_Split\test"),
    Path(r"D:\MASTER_FINAL_CONVNEXT_CLASSIFIER\MASTER_FINAL_CONVNEXT_CLASSIFIER\Dataset\Dataset_Original"),
    Path(__file__).resolve().parent.parent / "sampel",
]
DATASET_DIR = next((p for p in LOCAL_DATASET_DIRS if p.exists()), Path("sampel"))

# ==============================================================================
# 4. RESOLVER BOBOT MODEL (HUGGING FACE FIRST -> FALLBACK LOKAL)
# ==============================================================================
def resolve_weight(file_name: str, repo_id: str = "", token: str | None = None) -> Path:
    """Prioritaskan unduh dinamis dari Hugging Face Hub, lalu fallback ke lokal jika offline."""
    target_repo = repo_id or HF_DEFAULT_REPO
    target_token = token if token is not None else HF_DEFAULT_TOKEN

    if target_repo:
        try:
            from huggingface_hub import hf_hub_download
            hf_filename = (
                f"01_Model_Weights_Final/FLAT/{file_name}"
                if "FLAT" in file_name
                else f"01_Model_Weights_Final/CASCADE/{file_name}"
            )
            return Path(hf_hub_download(repo_id=target_repo, filename=hf_filename, token=target_token))
        except Exception:
            pass

    path = next(MODEL_DIR.rglob(file_name), None) if MODEL_DIR.exists() else None
    if path is not None:
        return path

    raise FileNotFoundError(
        f"File bobot {file_name} gagal diunduh dari Hugging Face ({target_repo}) dan tidak ditemukan di lokal ({MODEL_DIR})."
    )

# ==============================================================================
# 5. DATA EVALUASI STRATIFIED 5-FOLD CROSS VALIDATION
# ==============================================================================
EVAL_FOLD = pd.DataFrame(
    {
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "CASCADE Accuracy (%)": [95.31, 93.75, 94.06, 94.06, 94.38],
        "FLAT Accuracy (%)": [91.88, 94.06, 93.12, 93.12, 92.81],
        "CASCADE Macro F1 (%)": [95.32, 93.60, 93.86, 93.91, 94.34],
        "FLAT Macro F1 (%)": [91.87, 94.06, 93.02, 93.14, 92.76],
    }
)
EVAL_FOLD["Selisih Accuracy (%)"] = (EVAL_FOLD["CASCADE Accuracy (%)"] - EVAL_FOLD["FLAT Accuracy (%)"]).round(2)
EVAL_FOLD["Selisih Macro F1 (%)"] = (EVAL_FOLD["CASCADE Macro F1 (%)"] - EVAL_FOLD["FLAT Macro F1 (%)"]).round(2)
