"""Validate observed data without silently dropping or repairing observations."""
import numpy as np
import pandas as pd


def validate_series(df: pd.DataFrame) -> pd.DataFrame:
    """Require unique daily dates and finite, nonnegative water demand."""
    required = {'timestamp', 'water_produced'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    result = df.copy()
    result['timestamp'] = pd.to_datetime(result['timestamp'], errors='raise')
    if result.empty or result.timestamp.isna().any() or result.timestamp.duplicated().any():
        raise ValueError('Dates must be nonempty, nonmissing and unique')
    if not result.timestamp.eq(result.timestamp.dt.normalize()).all():
        raise ValueError('Expected daily timestamps at midnight')
    result['water_produced'] = pd.to_numeric(result.water_produced, errors='raise')
    values = result.water_produced.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Demand must be finite and nonnegative')
    return result.sort_values('timestamp').reset_index(drop=True)
