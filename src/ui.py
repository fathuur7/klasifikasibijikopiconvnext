from pathlib import Path
import pandas as pd
import plotly.express as px
from src.config import VARIETAS, IMG_EXT

def bar_chart(labels, values, title: str, color: str) -> px.bar:
    """Membangun horizontal bar chart interaktif untuk distribusi probabilitas."""
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

def list_dataset_images(dataset_dir: Path) -> list[str]:
    """Mengumpulkan seluruh file citra sampel dari folder dataset secara aman."""
    if not dataset_dir.exists():
        return []
    return sorted(
        p.relative_to(dataset_dir).as_posix()
        for p in dataset_dir.rglob("*")
        if p.suffix.lower() in IMG_EXT
    )

def ground_truth(rel_path: str):
    """Mengekstrak label ground truth dari struktur direktori Varietas/Sangrai/nama.jpg."""
    parts = Path(rel_path).parts
    if len(parts) >= 3 and parts[0] in VARIETAS:
        return parts[0], parts[1]
    return None
