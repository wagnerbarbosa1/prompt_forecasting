"""Only deterministic calendar data may be computed for future dates."""
import pandas as pd


def future_dates(last_observed: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_observed + pd.Timedelta(days=1), periods=horizon, freq='D')
