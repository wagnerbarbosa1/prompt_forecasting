"""PromptCast prompts with demand and the same covariates used by the baselines; future targets are absent by construction."""
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.data.features.temporal import future_dates


@dataclass(frozen=True)
class Demonstration:
    history: tuple[float, ...]
    target: tuple[float, ...]
    target_end: int


def select_demonstrations(training_values, history, horizon: int, *,
                          strategy: str = 'zero', count: int = 3,
                          seed: int = 42) -> list[Demonstration]:
    """Select only from the frozen prefix preceding the FIRST evaluated target.

    Similarity uses historical inputs only. The caller passes an already
    restricted training prefix; test observations can never become examples.
    """
    if strategy not in {'zero', 'few_random', 'few_similar'}:
        raise ValueError(f'Unknown strategy: {strategy}')
    if strategy == 'zero':
        return []
    if count < 1 or horizon < 1:
        raise ValueError('count and horizon must be positive')
    values = np.asarray(training_values, dtype=float)
    width = len(history)
    origins = np.arange(width, len(values) - horizon + 1)
    if len(origins) < count:
        raise ValueError('Insufficient pre-test data for requested demonstrations')
    if strategy == 'few_random':
        selected = np.random.default_rng(seed).choice(origins, size=count, replace=False)
    else:
        def standardized(x):
            return (x - np.mean(x)) / max(float(np.std(x)), 1e-8)
        query = standardized(np.asarray(history))
        distances = [np.mean((standardized(values[o-width:o]) - query) ** 2) for o in origins]
        selected = origins[np.argsort(distances, kind='stable')[:count]]
    return [Demonstration(tuple(values[o-width:o]), tuple(values[o:o+horizon]), int(o+horizon))
            for o in sorted(selected)]


def format_values(values) -> str:
    """Round only the text representation, retaining raw values in model baselines."""
    return ','.join(f'{float(value):.2f}' for value in values)


def build_prompt(history: pd.DataFrame, horizon: int, *,
                 demonstrations: list[Demonstration] | None = None,
                 future_covariates: pd.DataFrame | None = None) -> str:
    """Accept history only, not the complete dataframe or a target dataframe."""
    if horizon < 1 or history.empty:
        raise ValueError('History and positive horizon are required')
    values = history.water_produced.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('History contains invalid values')
    dates = future_dates(history.timestamp.iloc[-1], horizon)
    parts = [
        'Forecast daily water production in cubic metres.',
        'Use the chronological daily history below to infer level, trend and seasonality.',
        f'Return ONLY a JSON object with key "predictions" and exactly {horizon} nonnegative numbers.',
        'Do not explain. Do not repeat the historical values as an answer.'
    ]
    for example in demonstrations or []:
        parts.append('Historical example input: ' + format_values(example.history))
        parts.append('Historical example output: ' + json.dumps({'predictions': list(example.target)}))
    parts.extend([
        f'History: {len(history)} consecutive daily observations, '
        f'from {history.timestamp.iloc[0].date()} to {history.timestamp.iloc[-1].date()}.',
        'Values (oldest to newest): ' + format_values(values),
        'Forecast dates in output order: ' + ','.join(dates.strftime('%Y-%m-%d')),
        f'Output schema: {{"predictions": [number, ...]}}; array length must be {horizon}.'
    ])
    if future_covariates is not None:
        if 'water_produced' in future_covariates.columns:
            raise ValueError('Future target must never appear in a prompt')
        if len(future_covariates) != horizon or not dates.equals(pd.DatetimeIndex(future_covariates.timestamp)):
            raise ValueError('Future covariate dates must match the requested horizon')
        columns = [c for c in future_covariates.columns if c != 'timestamp']
        if not set(columns).issubset(history.columns):
            raise ValueError('Historical and future covariates must match')
        parts.extend([
            'Use the historical and supplied future covariates below in this forecast.',
            'Covariate values for forecast dates are supplied by the dataset.',
            'Historical covariates (CSV):',
            history[['timestamp'] + columns].to_csv(index=False, float_format='%.6g').strip(),
            'Future covariates (CSV, no future water production):',
            future_covariates.to_csv(index=False, float_format='%.6g').strip()
        ])
    return '\n'.join(parts)


def build_tabular_prompt(history: pd.DataFrame, horizon: int, *, future_covariates: pd.DataFrame) -> str:
    """Full daily rows with a stable historical prefix for Ollama cache reuse.

    Both arms of the context experiment use this exact layout and precision.
    Metadata that changes with the origin follows the history, never preceding it.
    """
    if horizon < 1 or history.empty or 'water_produced' in future_covariates:
        raise ValueError('Require historical demand and future covariates without targets')
    dates = future_dates(history.timestamp.iloc[-1], horizon)
    if len(future_covariates) != horizon or not dates.equals(pd.DatetimeIndex(future_covariates.timestamp)):
        raise ValueError('Future covariate dates do not match the horizon')
    columns = [c for c in future_covariates if c != 'timestamp']
    if not set(columns).issubset(history):
        raise ValueError('Historical covariates are missing')
    frame = history[['timestamp','water_produced'] + columns].copy()
    if not np.isfinite(frame.water_produced.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite demand in history')
    frame['water_produced'] = frame.water_produced.map(lambda value: f'{float(value):.2f}')
    return '\n'.join([
        'Forecast daily water production in cubic metres from the complete historical table below.',
        'Use demand, weather and calendar covariates to infer level, trend and seasonality.',
        f'Return ONLY a JSON object with key "predictions" and exactly {horizon} nonnegative numbers.',
        'Do not explain. Do not repeat the historical values as an answer.',
        'HISTORICAL DAILY OBSERVATIONS (CSV, oldest to newest):',
        frame.to_csv(index=False, float_format='%.6g').strip(),
        f'END HISTORY: {len(frame)} observations from {frame.timestamp.iloc[0].date()} to {frame.timestamp.iloc[-1].date()}.',
        'SUPPLIED FUTURE COVARIATES (CSV, no future water production):',
        future_covariates.to_csv(index=False, float_format='%.6g').strip(),
        'Forecast dates in output order: ' + ','.join(dates.strftime('%Y-%m-%d')),
        f'Output schema: {{"predictions": [number, ...]}}; array length must be {horizon}.'
    ])
