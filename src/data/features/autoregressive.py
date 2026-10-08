"""Lag features derived exclusively from observations preceding each target."""
import numpy as np


def lagged_samples(values, lags: int):
    values = np.asarray(values, dtype=float)
    if lags < 1 or len(values) <= lags or not np.isfinite(values).all():
        raise ValueError('Finite history longer than a positive lag count is required')
    windows = np.lib.stride_tricks.sliding_window_view(values, lags + 1)
    return windows[:, :-1].copy(), windows[:, -1].copy()


def recursive_forecast(estimator, history, horizon: int, lags: int):
    """Append predictions, never real values from inside the forecast horizon."""
    history = list(np.asarray(history, dtype=float))
    if horizon < 1 or len(history) < lags:
        raise ValueError('Insufficient history or invalid horizon')
    predictions = []
    for _ in range(horizon):
        value = float(estimator.predict(np.asarray(history[-lags:]).reshape(1, -1))[0])
        if not np.isfinite(value):
            raise ValueError('Nonfinite recursive forecast')
        predictions.append(value)
        history.append(value)
    return np.asarray(predictions)
