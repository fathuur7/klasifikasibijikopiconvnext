import plotly.graph_objects as go

# ==============================================================================
# BASIS PENGETAHUAN PROFIL SANGRAI (PARAMETER FISIKOKIMIAWI & SENSORIS)
# ==============================================================================
ROAST_PROFILES = {
    "Greenbean": {
        "nama": "Greenbean (Biji Kopi Mentah)",
        "agtron": "> 95 (Biji Mentah / Raw Green Coffee)",
        "fase": "Pra-Pemanggangan (Fase Pengeringan / Drying Phase)",
        "suhu": "Suhu Lingkungan (20°C - 30°C)",
        "kadar_air": "10.5% - 12.0%",
        "deskripsi": "Biji kopi mentah belum mengalami reaksi pirolisis atau pemanasan termal. Tekstur sangat keras, padat, dengan aroma dedaunan segar (grassy).",
        "sensoris": {"Acidity": 10, "Body": 10, "Sweetness": 5, "Bitterness": 5, "Aroma": 15}
    },
    "Light": {
        "nama": "Light Roast (Sangrai Muda)",
        "agtron": "75 - 90 (Cinnamon / New England Roast)",
        "fase": "Selesai First Crack (Fase Pengembangan Awal)",
        "suhu": "195°C - 205°C",
        "kadar_air": "< 3.5%",
        "deskripsi": "Warna cokelat muda keemasan, permukaan kering tanpa minyak. Karakter rasa asli daerah asal (terroir) sangat menonjol dengan tingkat keasaman buah (acidity) cerah dan aroma floral.",
        "sensoris": {"Acidity": 92, "Body": 45, "Sweetness": 68, "Bitterness": 20, "Aroma": 90}
    },
    "Medium": {
        "nama": "Medium Roast (Sangrai Sedang)",
        "agtron": "55 - 70 (City / City+ / Full City Roast)",
        "fase": "Karamelisasi Gula Puncak (Menjelang Second Crack)",
        "suhu": "210°C - 219°C",
        "kadar_air": "< 2.5%",
        "deskripsi": "Warna cokelat tua merata dengan aroma karamel dan cokelat yang matang. Merupakan profil sangrai paling seimbang antara keasaman (acidity), rasa manis (sweetness), dan ketebalan rasa (body).",
        "sensoris": {"Acidity": 55, "Body": 72, "Sweetness": 90, "Bitterness": 38, "Aroma": 94}
    },
    "Dark": {
        "nama": "Dark Roast (Sangrai Gelap)",
        "agtron": "< 45 (Vienna / French / Italian Roast)",
        "fase": "Second Crack Selesai (Fase Pirolisis Lanjut)",
        "suhu": "225°C - 235°C",
        "kadar_air": "< 1.5%",
        "deskripsi": "Warna cokelat sangat gelap mendekati hitam dengan lapisan minyak alami mengilap di permukaan. Keasaman terdegradasi total, rasa pahit kuat dengan aroma asap sangrai (roasty/smoky) dan cokelat hitam.",
        "sensoris": {"Acidity": 15, "Body": 92, "Sweetness": 35, "Bitterness": 88, "Aroma": 76}
    }
}

# ==============================================================================
# BASIS PENGETAHUAN VARIETAS BIJI KOPI
# ==============================================================================
VARIETY_INFO = {
    "Arabika": {
        "nama_ilmiah": "Coffea arabica",
        "ketinggian": "1.000 - 2.100 mdpl",
        "kadar_kafein": "1.1% - 1.5% (Rendah)",
        "morfologi": "Biji lonjong/oval memanjang dengan garis belahan tengah berlekuk huruf S yang jelas.",
        "karakter_rasa": "Asam buah segar (citrus, berries), manis alami tinggi, aroma bunga lembut."
    },
    "Robusta": {
        "nama_ilmiah": "Coffea canephora",
        "ketinggian": "400 - 900 mdpl",
        "kadar_kafein": "2.2% - 2.7% (Tinggi, hampir 2x Arabika)",
        "morfologi": "Biji berbentuk bulat membal dengan garis belahan tengah relatif lurus.",
        "karakter_rasa": "Kekentalan rasa (body) tebal, rasa pahit tegas, nuansa cokelat hitam, kacang, dan rempah."
    },
    "Liberika": {
        "nama_ilmiah": "Coffea liberica",
        "ketinggian": "0 - 600 mdpl (Lahan Gambut / Pasang Surut)",
        "kadar_kafein": "1.2% - 1.5%",
        "morfologi": "Biji berukuran sangat besar (hampir 2 kali ukuran Arabika), asimetris dengan ujung runcing.",
        "karakter_rasa": "Aroma sangat harum khas buah nangka matang (jackfruit), rasa smoky-woody, dan body pekat."
    },
    "Ekselsa": {
        "nama_ilmiah": "Coffea liberica var. dewevrei",
        "ketinggian": "400 - 1.000 mdpl",
        "kadar_kafein": "1.0% - 1.3%",
        "morfologi": "Biji berukuran sedang dengan siluet tetesan air (teardrop) dan sisi ujung sedikit melengkung.",
        "karakter_rasa": "Perpaduan rasa unik antara keasaman buah tropis (tart/fruity) dan rasa rempah gelap gurih."
    }
}

# ==============================================================================
# PEMBUAT DIAGRAM RADAR PROFIL SENSORIS 5-DIMENSI
# ==============================================================================
def build_radar_chart(varietas: str, sangrai: str) -> go.Figure:
    """Membangun diagram radar sensoris interaktif berdasarkan interaksi Varietas x Sangrai."""
    prof = ROAST_PROFILES.get(sangrai, ROAST_PROFILES["Medium"])
    sensoris = dict(prof["sensoris"])

    if varietas == "Robusta":
        sensoris["Body"] = min(100, sensoris["Body"] + 15)
        sensoris["Bitterness"] = min(100, sensoris["Bitterness"] + 12)
        sensoris["Acidity"] = max(5, sensoris["Acidity"] - 20)
    elif varietas == "Liberika":
        sensoris["Body"] = min(100, sensoris["Body"] + 10)
        sensoris["Aroma"] = min(100, sensoris["Aroma"] + 10)
    elif varietas == "Ekselsa":
        sensoris["Acidity"] = min(100, sensoris["Acidity"] + 8)

    categories = list(sensoris.keys())
    values = list(sensoris.values())
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name=f"{varietas} {sangrai}",
        line=dict(color='#8c510a', width=2.5),
        fillcolor='rgba(140, 81, 10, 0.35)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=320,
        margin=dict(l=30, r=30, t=20, b=20)
    )
    return fig
