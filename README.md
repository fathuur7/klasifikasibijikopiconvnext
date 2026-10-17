# Sistem Klasifikasi Biji Kopi dan Analisis Profil Sangrai Menggunakan Arsitektur ConvNeXt

Repositori ini berisi sistem klasifikasi citra biji kopi dan analisis mutu sangrai berbasis deep learning arsitektur ConvNeXt-Tiny. Sistem membandingkan dua pendekatan klasifikasi: arsitektur Hierarchical Cascade 2-Tahap dan Single-Stage Flat 16 Kelas, serta dilengkapi evaluasi fisikokimia profil pemanggangan.

## Ringkasan Proyek

Penelitian ini mengklasifikasikan 4 varietas biji kopi dan 4 level sangrai:
- Varietas: Arabika, Ekselsa, Liberika, Robusta
- Level Sangrai: Dark, Greenbean, Light, Medium
- Total Kelas Kombinasi: 16 Kelas

### Arsitektur Klasifikasi

1. Hierarchical CASCADE (2-Tahap):
   - Tahap 1: Model gerbang mengklasifikasikan citra ke dalam 4 varietas.
   - Tahap 2: Model spesialis varietas mengklasifikasikan level sangrai (4 kelas per varietas).
   - Menghasilkan probabilitas gabungan (joint probability) $P(\text{Varietas}, \text{Sangrai}) = P(\text{Varietas}) \times P(\text{Sangrai} \mid \text{Varietas})$.

2. Single-Stage FLAT:
   - Satu model ConvNeXt memprediksi langsung ke 16 kombinasi kelas sekaligus sebagai baseline pembanding.

### Hasil Evaluasi (Stratified 5-Fold Cross Validation)

| Pendekatan | Akurasi OOF (%) | Macro F1 (%) | Keunggulan Utama |
|---|---|---|---|
| CASCADE (2-Stage) | 94.31 | 94.21 | Memisahkan ruang fitur varietas dan derajat pemanggangan |
| FLAT (Single-Stage) | 93.00 | 92.97 | Baseline pembanding langsung |

## Struktur Repositori

Proyek disusun dengan struktur kode modular:

```text
klasifikasibijikopiconvnext/
├── app.py                      # Entrypoint antarmuka web Streamlit
├── requirements.txt            # Dependensi pustaka Python
├── .streamlit/
│   ├── secrets.toml.example    # Template konfigurasi rahasia Streamlit
│   └── secrets.toml            # Konfigurasi lokal (diabaikan oleh git)
├── src/
│   ├── __init__.py             # Inisialisasi paket modul
│   ├── config.py               # Pemetaan label, resolusi bobot dinamis, dan metrik
│   ├── model.py                # Konstruksi arsitektur ConvNeXt dan pemuatan bobot
│   ├── inference.py            # Pipeline inferensi CASCADE dan FLAT
│   ├── profiling.py            # Parameter fisikokimia profil sangrai & radar chart
│   └── ui.py                   # Fungsi visualisasi probabilitas dan pemilih sampel
└── README.md
```

## Integrasi Bobot Model

Seluruh bobot model hasil training (Fold 1 sampai Fold 5) disimpan di repositori Hugging Face:
- Repository Hub: `turasa22/klasifikasibijikopiconvnext`
- Mekanisme Unduh: Bobot model diunduh secara dinamis saat aplikasi dijalankan dan disimpan pada cache lokal. Jika koneksi internet tidak tersedia, aplikasi secara otomatis beralih menggunakan bobot di penyimpanan lokal komputer.

## Panduan Menjalankan Aplikasi

### 1. Kloning Repositori
```bash
git clone https://github.com/fathuur7/klasifikasibijikopiconvnext.git
cd klasifikasibijikopiconvnext
```

### 2. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Secrets (Opsional)
Jika menggunakan repositori privat di Hugging Face, buat file `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_your_access_token"
HF_REPO_ID = "turasa22/klasifikasibijikopiconvnext"
```

### 4. Jalankan Aplikasi Web
```bash
streamlit run app.py
```
Aplikasi dapat diakses melalui browser pada alamat `http://localhost:8501`.