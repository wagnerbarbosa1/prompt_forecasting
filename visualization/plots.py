import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Optional
from siaqua.visualization.colors import COLORS, NEUTRALS, PALETTES

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EDA_GRAPH_FOLDER = PROJECT_ROOT / "eda_graphs"
OUTPUT_MODELS_FOLDER = PROJECT_ROOT / "output_models"


def _resolve_save_folder(output_type: Optional[str], model_name: Optional[str]) -> Optional[Path]:
    if output_type == "eda":
        return EDA_GRAPH_FOLDER

    if output_type == "output_models":
        if not model_name:
            raise ValueError("model_name é obrigatório quando output_type='output_models'")
        return OUTPUT_MODELS_FOLDER / f"output_{model_name}" / f"{model_name}_graph"

    if output_type is None:
        return None

    raise ValueError("wrong output_type, set 'eda' or 'output_models' or None")


def linear_graph(
    df: pd.DataFrame,
    timestamp_col: str,
    highlight_col: str,
    output_type: Optional[str] = None,
    compare_cols: Sequence[str] = (),
    model_name: Optional[str] = None,
    title: Optional[str] = None,
    filename: str = "line_graph.png",
):
    save_folder = _resolve_save_folder(output_type, model_name)

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use("default")

    gray_tones = [NEUTRALS["medium"], NEUTRALS["light"], NEUTRALS["lighter"]]
    for i, col in enumerate(compare_cols):
        ax.plot(
            df[timestamp_col], df[col],
            color=gray_tones[i % len(gray_tones)],
            alpha=0.8, linewidth=1.0, label=col, zorder=1,
        )

    highlight_color = PALETTES["highlight"]["highlight"]
    ax.plot(
        df[timestamp_col], df[highlight_col],
        color=highlight_color, linewidth=1.8, alpha=0.95,
        label=highlight_col, zorder=3,
    )

    ax.set_title(
        title or f"{highlight_col} x {', '.join(compare_cols)}",
        fontsize=16, loc="left", pad=20, color=NEUTRALS["dark"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(NEUTRALS["lighter"])
    ax.spines["bottom"].set_color(NEUTRALS["lighter"])
    ax.tick_params(axis="both", colors=NEUTRALS["medium"])
    ax.grid(axis="y", linestyle="--", alpha=0.3, color=NEUTRALS["lighter"])

    ax.legend(frameon=False, loc="upper left", fontsize=9)

    plt.tight_layout()
    if output_type is None:
        plt.show()
        return None
    
    save_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_folder / filename, dpi=150)
    plt.close(fig)