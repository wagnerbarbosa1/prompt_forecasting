"""One data preparation path shared by prompts, models and evaluation."""
from src.data.ingestion.files import read_series
from src.data.preprocessing.resampling import daily_grid
from src.data.preprocessing.missing_values import fill_past_only
from src.evaluation.cross_validation import build_windows


def prepare_dataset(path, context, *, horizon=7, step=7):
    df = fill_past_only(daily_grid(read_series(path)))
    windows = build_windows(df, context, horizon, step)
    return df, windows
