"""Causal missing-day treatment; imputed targets never enter evaluation."""
import pandas as pd


def fill_past_only(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    columns = [c for c in result.columns if c not in {'timestamp', 'observed'}]
    result[columns] = result[columns].ffill()
    if result.water_produced.isna().any():
        raise ValueError('Cannot fill initial missing demand without past data')
    return result
