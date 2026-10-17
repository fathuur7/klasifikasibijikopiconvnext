"""
Sistem Klasifikasi Biji Kopi dan Analisis Profil Sangrai Menggunakan Arsitektur ConvNeXt
Riset MBKM & Pengajuan HKI 2026
"""
from pathlib import Path
from PIL import Image
import streamlit as st
import torch

from src.config import (
    VARIETAS,
    SANGRAI,
    FLAT_CLASSES,
    HF_DEFAULT_REPO,
    HF_DEFAULT_TOKEN,
    MODEL_DIR,
    DATASET_DIR,
    EVAL_FOLD,
)
from src.inference import predict_cascade, predict_flat
from src.profiling import ROAST_PROFILES, VARIETY_INFO, build_radar_chart
from src.ui import bar_chart, list_dataset_images, ground_truth

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Biji Kopi & Profil Sangrai (ConvNeXt)",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# BILAH SISI (SIDEBAR): PENGATURAN MODEL & BOBOT SECARA DINAMIS
# ==============================================================================
with st.sidebar:
    st.header("Pengaturan Model & Bobot")
    fold = st.selectbox("Pilih Bobot Model Fold (1 - 5):", [1, 2, 3, 4, 5], index=0)
    st.markdown(f"**Akselerator Device:** `{DEVICE.upper()}`")
    show_flat = st.checkbox("Tampilkan Pembanding Model FLAT", value=True)

    # Pengunduhan model dinamis di latar belakang (Hugging Face Hub + Fallback Lokal)
    active_repo = HF_DEFAULT_REPO
    active_token = HF_DEFAULT_TOKEN

    st.markdown("---")
    st.subheader("Ringkasan Metrik Riset (OOF)")
    st.markdown(
        "- **CASCADE (2-Stage)**\n"
        "  - Akurasi OOF: **94.31%**\n"
        "  - Macro F1: **94.21%**\n"
        "  - Stabilitas: $\pm$0.60%\n"
        "- **FLAT (Single-Stage)**\n"
        "  - Akurasi OOF: **93.00%**\n"
        "  - Macro F1: **92.97%**\n"
    )
    st.caption("Riset MBKM & Pengajuan HKI 2026")

# ==============================================================================
# HEADER UTAMA
# ==============================================================================
st.title("Sistem Klasifikasi Biji Kopi dan Analisis Profil Sangrai")
st.caption("Implementasi Deep Learning Berbasis Arsitektur ConvNeXt-Tiny untuk Klasifikasi Varietas dan Karakterisasi Mutu Pemanggangan")

tab_klas, tab_profil, tab_eval, tab_algo = st.tabs(
    [
        "Klasifikasi & Inferensi",
        "Analisis Profil Sangrai",
        "Laporan Evaluasi & Benchmark",
        "Arsitektur & Algoritma"
    ]
)

# ==============================================================================
# TAB 1: KLASIFIKASI & INFERENSI
# ==============================================================================
with tab_klas:
    st.markdown("Unggah citra biji kopi atau pilih sampel dari dataset uji untuk melakukan klasifikasi komparatif.")

    metode = st.radio(
        "Metode Input Citra:",
        ["Pilih dari Dataset Uji (Sampel)", "Unggah Citra Mandiri"],
        horizontal=True,
    )

    img, gt = None, None
    if metode.startswith("Pilih"):
        files = list_dataset_images(DATASET_DIR)
        if not files:
            st.warning(f"Folder dataset `{DATASET_DIR}` tidak ditemukan atau kosong. Gunakan opsi unggah citra.")
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
            st.markdown(f"- **Dimensi:** {img.size[0]} x {img.size[1]} piksel")
            st.markdown(f"- **Format Warna:** {img.mode}")
            if gt:
                st.markdown(f"- **Ground Truth (Dataset):** Varietas `{gt[0]}` | Sangrai `{gt[1]}`")
            run = st.button("Jalankan Klasifikasi", type="primary")

        if run:
            try:
                with st.spinner("Mengunduh bobot & memproses inferensi..."):
                    res = predict_cascade(img, fold, DEVICE, repo_id=active_repo, token=active_token)
                    flat = predict_flat(img, fold, DEVICE, repo_id=active_repo, token=active_token) if show_flat else None
                st.session_state["hasil"] = res
                st.session_state["hasil_flat"] = flat
            except Exception as e:
                st.error(f"Terjadi kesalahan saat inferensi: {e}")

    res = st.session_state.get("hasil")
    if res:
        st.markdown("---")
        st.subheader(f"Hasil Klasifikasi (Model Fold {fold})")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Varietas Terdeteksi", res["varietas"], f"{res['conf_varietas'] * 100:.2f}%", delta_color="off")
        k2.metric("Derajat Sangrai", res["sangrai"], f"{res['conf_sangrai'] * 100:.2f}%", delta_color="off")
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
                    f"Tahap 2: Probabilitas Sangrai ({res['varietas']})",
                    "#8d6e63",
                ),
                use_container_width=True,
            )

        flat = st.session_state.get("hasil_flat")
        if flat:
            with st.expander("Pembanding: Model FLAT (16 Kelas Langsung)", expanded=True):
                is_agree = flat["kelas"] == res["kelas"]
                status_txt = "**Sepakat** dengan hasil CASCADE." if is_agree else "**Berbeda keputusan** dari CASCADE."
                st.markdown(f"Prediksi Model FLAT: `{flat['kelas']}` (Confidence: {flat['conf'] * 100:.2f}%) — {status_txt}")
                top5 = sorted(zip(FLAT_CLASSES, flat["probs"]), key=lambda t: -t[1])[:5]
                st.plotly_chart(
                    bar_chart([t[0] for t in top5], [t[1] for t in top5], "Top-5 Probabilitas Model FLAT", "#7b1fa2"),
                    use_container_width=True,
                )

# ==============================================================================
# TAB 2: PROFIL SANGRAI & ANALISIS MUTU SENSORIS (DINAMIS)
# ==============================================================================
with tab_profil:
    res = st.session_state.get("hasil")
    if not res:
        st.info("Silakan jalankan klasifikasi pada tab **Klasifikasi & Inferensi** terlebih dahulu untuk melihat analisis profil.")
    else:
        active_varietas = res["varietas"]
        active_sangrai = res["sangrai"]
        prof = ROAST_PROFILES.get(active_sangrai, ROAST_PROFILES["Medium"])
        var_data = VARIETY_INFO.get(active_varietas, VARIETY_INFO["Arabika"])

        st.subheader(f"Karakterisasi Mutu Fisikokimiawi: {active_varietas} {active_sangrai}")
        
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown("#### 1. Parameter Fisikokimiawi Pemanggangan")
            st.markdown(f"""
            - **Indeks Warna (Skala Agtron)**: `{prof['agtron']}`
            - **Fase Pemanggangan (Thermal Stage)**: {prof['fase']}
            - **Suhu Biji / Drum Perkiraan**: `{prof['suhu']}`
            - **Kadar Air Perkiraan**: `{prof['kadar_air']}`
            - **Deskripsi Fisik**: {prof['deskripsi']}
            """)

            st.markdown("---")
            st.markdown("#### 2. Karakteristik Varietas Biji Kopi")
            st.markdown(f"""
            - **Nama Ilmiah**: *{var_data['nama_ilmiah']}*
            - **Elevasi Habitat Optimal**: {var_data['ketinggian']}
            - **Kadar Kafein Rata-rata**: {var_data['kadar_kafein']}
            - **Ciri Morfologi Biji**: {var_data['morfologi']}
            - **Karakter Asal (Terroir)**: {var_data['karakter_rasa']}
            """)

        with col_p2:
            st.markdown("#### 3. Diagram Radar Profil Sensoris (Rasa & Aroma)")
            fig_radar = build_radar_chart(active_varietas, active_sangrai)
            st.plotly_chart(fig_radar, use_container_width=True)

    with st.expander("Referensi Seluruh Tingkat Sangrai (Agtron & Karakteristik Fisik)"):
        for k in ["Greenbean", "Light", "Medium", "Dark"]:
            p = ROAST_PROFILES[k]
            st.markdown(f"**{p['nama']}** (`{p['agtron']}`) — {p['fase']}  \n*{p['deskripsi']}*")

# ==============================================================================
# TAB 3: LAPORAN EVALUASI & BENCHMARK
# ==============================================================================
with tab_eval:
    st.subheader("Rekapitulasi Kinerja Model: Stratified 5-Fold Cross Validation")
    st.caption("Hasil pengujian komparatif pada 1.600 citra uji Out-of-Fold (OOF).")
    st.dataframe(EVAL_FOLD, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Analisis Keunggulan Arsitektur Hierarchical CASCADE")
    st.markdown("""
    - **Isolasi Fitur Morfologi**: Model CASCADE mengunci bentuk varietas di tahap pertama (Akurasi Jenis 99.31%), sehingga kebal terhadap kesalahan silang varietas saat biji berwarna gelap (*Dark Roast*).
    - **Akurasi Lebih Tinggi**: CASCADE memperoleh akurasi **94.31%** (unggul +1.31% dibanding FLAT 93.00%) dan Macro F1 **94.21%** (unggul +1.24%).
    - **Stabilitas Konsisten**: Standar deviasi CASCADE ($\pm 0.60\%$) lebih stabil dibanding FLAT ($\pm 0.78\%$).
    """)

# ==============================================================================
# TAB 4: ARSITEKTUR & PANDUAN ALGORITMA
# ==============================================================================
with tab_algo:
    st.subheader("Panduan Alur Algoritma dalam Bahasa Manusia (Non-Code)")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        **1. Alur Model ConvNeXt Hierarchical CASCADE (2-Stage)**
        1. **Standardisasi Citra**: Citra diubah ke format 224 x 224 piksel dan dinormalisasi.
        2. **Tahap 1 - Model Jenis (Gatekeeper)**: Memindai morfologi fisik biji untuk mengunci varietas (*Arabika, Ekselsa, Liberika, atau Robusta*).
        3. **Tahap 2 - Model Spesialis Sangrai**: Mengaktifkan model spesialis yang bersesuaian untuk menentukan derajat sangrai (*Dark, Greenbean, Light, atau Medium*).
        4. **Agregasi Keputusan**: Menggabungkan hasil varietas dan sangrai serta menghitung nilai *Joint Confidence*.
        """)

    with col_a2:
        st.markdown("""
        **2. Alur Model ConvNeXt FLAT (Single-Stage 16 Kelas)**
        1. **Standardisasi Citra**: Citra disiapkan dalam ukuran standar 224 x 224 piksel.
        2. **Ekstraksi Fitur Tunggal**: Jaringan memindai bentuk dan warna secara simultan melalui konvolusi ConvNeXt.
        3. **Penetapan Keputusan**: Jaringan langsung membagi probabilitas ke dalam 16 kelas kombinasi sekaligus dan memilih probabilitas tertinggi.
        """)