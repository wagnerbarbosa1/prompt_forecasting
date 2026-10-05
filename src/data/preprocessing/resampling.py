"""Expose missing calendar days explicitly; do not interpolate future values."""
import pandas as pd


def daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    result = df.set_index('timestamp').asfreq('D')
    result['observed'] = result.water_produced.notna()
    return result.rename_axis('timestamp').reset_index()
