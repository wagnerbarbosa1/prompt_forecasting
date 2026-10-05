"""Explicit metric definitions, with no independent dropping of true/predicted rows."""
import numpy as np


def deterministic_metrics(ytrue, yhat) -> dict:
    actual, predicted = np.asarray(ytrue, dtype=float), np.asarray(yhat, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError('Aligned nonempty targets and predictions are required')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Nonfinite values are not silently removed')
    error = predicted - actual
    nonzero = actual != 0
    denom = np.abs(actual) + np.abs(predicted)
    smape = np.divide(2 * np.abs(error), denom, out=np.zeros_like(error), where=denom != 0)
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'mape_percent': float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            'smape_percent': float(np.mean(smape) * 100),
            'r2': 1 - float(np.sum(error ** 2)) / total if total else None,
            'mape_zero_targets_excluded': int((~nonzero).sum())}
