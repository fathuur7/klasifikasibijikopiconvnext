"""
Sistem Klasifikasi Biji Kopi dan Identifikasi Tingkat Sangrai
Berbasis Arsitektur ConvNeXt-Tiny (Hierarchical CASCADE)

Jalankan dengan:  streamlit run app.py
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# ============================================================
# KONFIGURASI  (sesuaikan dengan folder dan urutan kelas Anda)
# ============================================================
# Gunakan list path lokal agar otomatis mendeteksi folder riset lokal jika ada, atau fallback ke Hugging Face saat cloud deploy.
LOCAL_MODEL_DIRS = [
    Path(r"d:\Riset_Mbkm_2026\MASTER_FINAL_CONVNEXT_CLASSIFIER\01_Model_Weights_Final"),
    Path(r"D:\MASTER_FINAL_CONVNEXT_CLASSIFIER\MASTER_FINAL_CONVNEXT_CLASSIFIER\01_Model_Weights_Final"),
    Path(__file__).resolve().parent / "01_Model_Weights_Final",
]
MODEL_DIR = next((p for p in LOCAL_MODEL_DIRS if p.exists()), LOCAL_MODEL_DIRS[0])

LOCAL_ASSET_DIRS = [
    Path(r"d:\Riset_Mbkm_2026\MASTER_FINAL_CONVNEXT_CLASSIFIER\02_Reports_and_Metrics"),
    Path(r"D:\MASTER_FINAL_CONVNEXT_CLASSIFIER\MASTER_FINAL_CONVNEXT_CLASSIFIER\02_Reports_and_Metrics"),
    Path(__file__).resolve().parent / "assets",
]
ASSETS_DIR = next((p for p in LOCAL_ASSET_DIRS if p.exists()), Path("assets"))

LOCAL_DATASET_DIRS = [
    Path(r"d:\Riset_Mbkm_2026\PROJECT_1V\Dataset_Split\test"),
    Path(r"D:\MASTER_FINAL_CONVNEXT_CLASSIFIER\MASTER_FINAL_CONVNEXT_CLASSIFIER\Dataset\Dataset_Original"),
    Path(__file__).resolve().parent / "sampel",
]
DATASET_DIR = next((p for p in LOCAL_DATASET_DIRS if p.exists()), Path("sampel"))

HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "" if MODEL_DIR.exists() else "Naufalthrq10/klasifikasibijikopiconvnext",
)
HF_TOKEN = os.getenv("HF_TOKEN") or None

# PENTING: urutan harus SAMA dengan urutan indeks kelas saat training
# (ImageFolder mengurutkan secara alfabetis).
VARIETAS = ["Arabika", "Ekselsa", "Liberika", "Robusta"]
SANGRAI = ["Dark", "Greenbean", "Light", "Medium"]
FLAT_CLASSES = [f"{v}_{s}" for v in VARIETAS for s in SANGRAI]

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INFO_SANGRAI = {
    "Greenbean": {
        "nama": "Greenbean (biji mentah)",
        "ciri": "Biji belum disangrai, berwarna hijau keabuan hingga hijau kekuningan, permukaan matte dan keras.",
        "karakter": "Belum memiliki aroma dan rasa khas kopi sangrai; masih bersifat herbal/rumput.",
    },
    "Light": {
        "nama": "Light Roast (sangrai muda)",
        "ciri": "Warna cokelat muda, permukaan kering tanpa minyak, belahan tengah masih terlihat jelas.",
        "karakter": "Umumnya berasa lebih asam (acidity tinggi), body ringan, aroma floral/buah.",
    },
    "Medium": {
        "nama": "Medium Roast (sangrai sedang)",
        "ciri": "Warna cokelat sedang, permukaan umumnya masih kering atau sedikit berminyak.",
        "karakter": "Keseimbangan antara asam, manis, dan pahit; body sedang.",
    },
    "Dark": {
        "nama": "Dark Roast (sangrai gelap)",
        "ciri": "Warna cokelat tua hingga hampir hitam, permukaan mengilap berminyak.",
        "karakter": "Dominan pahit dan aroma smoky/karamel gosong; acidity rendah, body tebal.",
    },
}

# Hasil evaluasi 5-Fold Cross Validation (1.600 citra uji OOF)
EVAL_FOLD = pd.DataFrame(
    {
        "Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "CASCADE Accuracy (%)": [95.31, 93.75, 94.06, 94.06, 94.38],
        "FLAT Accuracy (%)": [91.88, 94.06, 93.12, 93.12, 92.81],
        "CASCADE Macro F1 (%)": [95.32, 93.60, 93.86, 93.91, 94.34],
        "FLAT Macro F1 (%)": [91.87, 94.06, 93.02, 93.14, 92.76],
    }
)
EVAL_FOLD["Selisih Accuracy (%)"] = (
    EVAL_FOLD["CASCADE Accuracy (%)"] - EVAL_FOLD["FLAT Accuracy (%)"]
).round(2)
EVAL_FOLD["Selisih Macro F1 (%)"] = (
    EVAL_FOLD["CASCADE Macro F1 (%)"] - EVAL_FOLD["FLAT Macro F1 (%)"]
).round(2)

# ============================================================
# MODEL
# ============================================================
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def _build_model(num_classes: int, backend: str, custom_head: bool = True) -> nn.Module:
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


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return {k.replace("module.", "", 1): v for k, v in ckpt.items()}


def resolve_weight(file_name: str) -> Path:
    """Kembalikan lokasi file bobot: unduh dari Hugging Face (mode online) atau cari di folder lokal."""
    if HF_REPO_ID:
        from huggingface_hub import hf_hub_download
        
        if "FLAT" in file_name:
            hf_filename = f"01_Model_Weights_Final/FLAT/{file_name}"
        else:
            hf_filename = f"01_Model_Weights_Final/CASCADE/{file_name}"
            
        return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename, token=HF_TOKEN))
        
    path = next(MODEL_DIR.rglob(file_name), None)  # cari di semua subfolder
    if path is None:
        raise FileNotFoundError(f"File bobot {file_name} tidak ditemukan di {MODEL_DIR}")
    return path


# max_entries=2: hanya model Jenis + satu model varietas yang disimpan di RAM
@st.cache_resource(show_spinner=False, max_entries=2)
def load_model(file_name: str, num_classes: int, device_str: str) -> nn.Module:
    path = resolve_weight(file_name)
    try:
        ckpt = torch.load(path, map_location=device_str, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device_str, weights_only=False)

    if isinstance(ckpt, nn.Module):  # jika yang disimpan adalah model utuh
        return ckpt.to(device_str).eval()

    state = _extract_state_dict(ckpt)
    has_custom_head = any("head.fc.0" in k for k in state.keys())
    backends_to_try = [("timm", True)] if has_custom_head else [("timm", False), ("torchvision", False), ("timm", True)]
    
    last_err = None
    for backend, is_custom in backends_to_try:
        try:
            model = _build_model(num_classes, backend, custom_head=is_custom)
            model.load_state_dict(state, strict=True)
            return model.to(device_str).eval()
        except Exception as e:  # coba backend berikutnya
            last_err = e
    raise RuntimeError(
        f"Gagal memuat {file_name}. Pastikan arsitektur sesuai (torchvision/timm convnext_tiny). "
        f"Error terakhir: {last_err}"
    )


@torch.no_grad()
def _probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(model(x), dim=1)[0].cpu()


def predict_cascade(img: Image.Image, fold: int, device: str) -> dict:
    x = TRANSFORM(img).unsqueeze(0).to(device)

    m1 = load_model(f"ConvNeXt_Jenis_Fold{fold}.pth", len(VARIETAS), device)
    p1 = _probs(m1, x)
    i1 = int(p1.argmax())
    varietas = VARIETAS[i1]

    m2 = load_model(f"ConvNeXt_{varietas}_Fold{fold}.pth", len(SANGRAI), device)
    p2 = _probs(m2, x)
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


def predict_flat(img: Image.Image, fold: int, device: str) -> dict:
    x = TRANSFORM(img).unsqueeze(0).to(device)
    m = load_model(f"ConvNeXt_FLAT_Fold{fold}.pth", len(FLAT_CLASSES), device)
    p = _probs(m, x)
    i = int(p.argmax())
    return {"kelas": FLAT_CLASSES[i], "conf": float(p[i]), "probs": p.numpy()}


# ============================================================
# UI HELPER
# ============================================================
def bar_chart(labels, values, title, color, horizontal=True):
    df = pd.DataFrame({"Kelas": labels, "Probabilitas (%)": [v * 100 for v in values]})
    df = df.sort_values("Probabilitas (%)", ascending=True)
    fig = px.bar(
        df,
        x="Probabilitas (%)",
        y="Kelas",
        orientation="h",
        text=df["Probabilitas (%)"].map(lambda v: f"{v:.1f}%"),
        title=title,
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        height=max(220, 40 * len(df) + 90),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(range=[0, 105]),
    )
    fig.update_traces(textposition="outside")
    return fig


def list_dataset_images():
    if not DATASET_DIR.exists():
        return []
    return sorted(
        p.relative_to(DATASET_DIR).as_posix()
        for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in IMG_EXT
    )


def ground_truth(rel_path: str):
    parts = Path(rel_path).parts
    if len(parts) >= 3 and parts[0] in VARIETAS:
        return parts[0], parts[1]
    return None


# ============================================================
# HALAMAN
# ============================================================
st.set_page_config(page_title="Sistem Klasifikasi Biji Kopi", page_icon="☕", layout="wide")

# ---------- Sidebar ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
with st.sidebar:
    st.header("Pengaturan Model")
    fold = st.selectbox("Pilih Bobot Model Fold (1 - 5):", [1, 2, 3, 4, 5], index=0)
    st.markdown(f"**Device Komputasi:** `{device}`")
    show_flat = st.checkbox("Tampilkan pembanding model FLAT", value=False)
    st.divider()
    st.subheader("Ringkasan Kinerja Riset")
    st.markdown(
        "- **CASCADE (2-Stage)**\n"
        "  - Akurasi OOF: **94.31%**\n"
        "  - Macro F1: **94.21%**\n"
        "  - Standar Deviasi: ±0.60%\n"
    )
    st.divider()
    st.caption("Riset MBKM & Pengajuan HKI 2026")

st.title("Sistem Klasifikasi Biji Kopi")
st.subheader("Identifikasi Varietas dan Tingkat Sangrai Berbasis ConvNeXt-Tiny (Hierarchical CASCADE)")

tab_klas, tab_profil, tab_eval, tab_algo = st.tabs(
    ["Klasifikasi", "Profil Sangrai", "Evaluasi Model", "Arsitektur & Algoritma"]
)

# ---------- Tab 1: Klasifikasi ----------
with tab_klas:
    st.write(
        "Unggah citra biji kopi atau pilih sampel dari dataset uji. Sistem akan menentukan "
        "**varietas** (Tahap 1) lalu **tingkat sangrai** (Tahap 2)."
    )
    metode = st.radio(
        "Metode Input Citra:",
        ["Pilih dari Dataset Uji (Sampel)", "Unggah Citra Sendiri"],
        horizontal=True,
    )

    img, gt = None, None
    if metode.startswith("Pilih"):
        files = list_dataset_images()
        if not files:
            st.warning(
                f"Folder dataset `{DATASET_DIR}` tidak ditemukan atau kosong. "
                "Ubah `DATASET_DIR` di app.py atau gunakan opsi unggah citra."
            )
        else:
            pilihan = st.selectbox("Pilih sampel biji kopi dari dataset uji:", files)
            img = Image.open(DATASET_DIR / pilihan).convert("RGB")
            gt = ground_truth(pilihan)
    else:
        up = st.file_uploader("Unggah citra biji kopi", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up is not None:
            img = Image.open(up).convert("RGB")

    if img is not None:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.image(img, caption="Citra Masukan", use_container_width=True)
        with c2:
            st.markdown("#### Informasi Citra")
            st.markdown(f"- **Ukuran Citra:** {img.size[0]} x {img.size[1]} piksel")
            st.markdown(f"- **Format Warna:** {img.mode}")
            if gt:
                st.markdown(f"- **Ground Truth (Dataset):** Varietas `{gt[0]}` | Sangrai `{gt[1]}`")
            run = st.button("Klasifikasikan", type="primary")

        if run:
            try:
                with st.spinner("Memproses citra..."):
                    res = predict_cascade(img, fold, device)
                    flat = predict_flat(img, fold, device) if show_flat else None
                st.session_state["hasil"] = res
                st.session_state["hasil_flat"] = flat
            except Exception as e:
                st.error(str(e))

    res = st.session_state.get("hasil")
    if res:
        st.divider()
        st.markdown(f"### Hasil Klasifikasi (Bobot Fold {fold})")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Varietas", res["varietas"], f"{res['conf_varietas'] * 100:.2f}%", delta_color="off")
        k2.metric("Tingkat Sangrai", res["sangrai"], f"{res['conf_sangrai'] * 100:.2f}%", delta_color="off")
        k3.metric("Kelas Akhir", res["kelas"])
        k4.metric("Joint Confidence", f"{res['joint'] * 100:.2f}%")

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(
                bar_chart(VARIETAS, res["probs_varietas"], "Tahap 1: Probabilitas Varietas", "#2e7d32"),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(
                bar_chart(
                    SANGRAI,
                    res["probs_sangrai"],
                    f"Tahap 2: Probabilitas Sangrai (Model {res['varietas']})",
                    "#8d6e63",
                ),
                use_container_width=True,
            )

        flat = st.session_state.get("hasil_flat")
        if flat:
            with st.expander("Pembanding: Model FLAT (16 kelas)", expanded=True):
                st.markdown(
                    f"Prediksi FLAT: `{flat['kelas']}` (confidence {flat['conf'] * 100:.2f}%) — "
                    + ("**sepakat** dengan CASCADE." if flat["kelas"] == res["kelas"] else "**berbeda** dari CASCADE.")
                )
                top = sorted(zip(FLAT_CLASSES, flat["probs"]), key=lambda t: -t[1])[:5]
                st.plotly_chart(
                    bar_chart([t[0] for t in top], [t[1] for t in top], "Top-5 Probabilitas FLAT", "#7b1fa2"),
                    use_container_width=True,
                )

# ---------- Tab 2: Profil Sangrai ----------
with tab_profil:
    res = st.session_state.get("hasil")
    if not res:
        st.info("Jalankan klasifikasi pada tab **Klasifikasi** terlebih dahulu.")
    else:
        info = INFO_SANGRAI[res["sangrai"]]
        st.markdown(f"### {res['varietas']} — {info['nama']}")
        st.markdown(f"**Ciri visual:** {info['ciri']}")
        st.markdown(f"**Karakter umum:** {info['karakter']}")
        st.caption(
            "Keterangan karakter bersifat umum sebagai gambaran edukatif, bukan hasil uji cita rasa."
        )
        st.plotly_chart(
            bar_chart(SANGRAI, res["probs_sangrai"], "Distribusi Probabilitas Tingkat Sangrai", "#8d6e63"),
            use_container_width=True,
        )
    with st.expander("Referensi seluruh tingkat sangrai"):
        for k in ["Greenbean", "Light", "Medium", "Dark"]:
            st.markdown(f"**{INFO_SANGRAI[k]['nama']}**  \n{INFO_SANGRAI[k]['ciri']}  \n{INFO_SANGRAI[k]['karakter']}")

# ---------- Tab 3: Evaluasi ----------
with tab_eval:
    st.markdown("### Ringkasan Evaluasi 5-Fold Cross Validation (1.600 Citra Uji OOF)")
    st.caption("Model FLAT digunakan sebagai pembanding untuk membuktikan alasan pemilihan arsitektur CASCADE.")
    st.dataframe(EVAL_FOLD, use_container_width=True, hide_index=True)

    ringkas = pd.DataFrame(
        {
            "Metrik": ["Accuracy", "Macro F1"],
            "CASCADE (%)": [94.31, 94.21],
            "FLAT (%)": [93.00, 92.97],
        }
    ).melt(id_vars="Metrik", var_name="Model", value_name="Skor (%)")
    fig = px.bar(
        ringkas,
        x="Metrik",
        y="Skor (%)",
        color="Model",
        barmode="group",
        text="Skor (%)",
        color_discrete_map={"CASCADE (%)": "#2e7d32", "FLAT (%)": "#7b1fa2"},
        title="Perbandingan Kinerja Rata-rata 5-Fold",
    )
    fig.update_layout(yaxis=dict(range=[88, 96]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Matriks Konfusi Out-of-Fold (OOF)")
    def cari_cm(sub: str):
        folder = ASSETS_DIR / sub
        if folder.exists():
            for p in sorted(folder.rglob("*.png")):
                if "confusion" in p.name.lower():
                    return p
        return None

    cols = st.columns(2)
    for col, nama in zip(cols, ["CASCADE", "FLAT"]):
        with col:
            path = cari_cm(nama)
            if path:
                st.image(str(path), caption=f"Confusion Matrix {nama}", use_container_width=True)
            else:
                st.info(f"Gambar confusion matrix {nama} tidak ditemukan di {ASSETS_DIR / nama}.")

# ---------- Tab 4: Arsitektur ----------
with tab_algo:
    st.markdown("### Algoritma Klasifikasi Hierarchical CASCADE")
    st.code(
        """ALGORITMA KLASIFIKASI CASCADE:
1. Input citra biji kopi.
2. Praproses citra (resize ke 224x224, normalisasi).
3. Tahap 1: Model Jenis memprediksi varietas (Arabika, Ekselsa, Liberika, Robusta).
4. Kunci varietas terpilih (misal: "Arabika").
5. Tahap 2: Aktifkan model spesifik varietas tersebut untuk memprediksi
   tingkat sangrai (Greenbean, Light, Medium, Dark).
6. Gabungkan hasil: "Arabika" + "_" + "Dark" = "Arabika_Dark".
7. Hitung joint confidence = keyakinan_varietas x keyakinan_sangrai.
8. Selesai.""",
        language="text",
    )
    st.markdown("### Pembanding: Algoritma FLAT (End-to-End)")
    st.code(
        """ALGORITMA INFERENSI FLAT:
1. Input citra biji kopi.
2. Praproses citra (resize ke 224x224, normalisasi).
3. Masukkan ke Model FLAT (16 output neuron).
4. Ambil kelas dengan probabilitas tertinggi dari 16 kombinasi.
5. Selesai.""",
        language="text",
    )
    st.markdown(
        "| Aspek | FLAT | CASCADE |\n|---|---|---|\n"
        "| Strategi | 1 model, 16 kelas | 2 tahap: varietas lalu sangrai |\n"
        "| Forward pass | 1 kali | 2 kali |\n"
        "| Akurasi OOF | 93.00% (±0.78%) | 94.31% (±0.60%) |\n"
        "| Macro F1 | 92.97% | 94.21% |"
    )