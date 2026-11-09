"""PromptCast prompts with demand and the same covariates used by the baselines; future targets are absent by construction."""
import json
import random
import numpy as np
import pandas as pd
from src.data.features.temporal import future_dates
from src.data.preprocessing.calendar_text import calendar_to_text, date_to_text, FLAGS


def z_normalize_vector(values):
    """Original normalization: constant histories become zeros."""
    values = np.asarray(values, dtype=float)
    return np.zeros_like(values) if values.std() < 1e-12 else (values - values.mean()) / values.std()


def build_similarity_vector(frame):
    """Original S3 selection: demand, cyclic calendar, flags and weather history."""
    parts = [z_normalize_vector(frame.water_produced)]
    for values, period in [(frame.timestamp.dt.dayofweek, 7), (frame.timestamp.dt.month, 12)]:
        parts.extend([np.sin(2 * np.pi * values / period), np.cos(2 * np.pi * values / period)])
    for column, labels in FLAGS.items():
        if column in frame:
            values = frame[column].map(lambda value: 0 if value == labels[0] else 1 if value == labels[1] else value)
            parts.append(pd.to_numeric(values, errors='raise').to_numpy(dtype=float))
    for column in frame:
        if column.startswith(('temperature_', 'radiation_', 'shortwave_', 'precipitation_', 'relative_humidity_')):
            parts.append(z_normalize_vector(frame[column]))
    return np.concatenate(parts)


def select_demonstrations(training, history, horizon, *, strategy='zero', count=3, rng=None):
    """Select complete examples from a pre-test DataFrame, using historical inputs only."""
    if strategy not in {'zero', 'few_random', 'few_similar'}:
        raise ValueError(f'Unknown strategy: {strategy}')
    if strategy == 'zero':
        return []
    if count < 1 or horizon < 1:
        raise ValueError('count and horizon must be positive')
    width = len(history)
    candidates = [origin for origin in range(width, len(training) - horizon + 1)
                  if 'observed' not in training or training.iloc[origin-width:origin+horizon].observed.all()]
    if len(candidates) < count:
        raise ValueError('Insufficient pre-test examples')
    if strategy == 'few_random':
        selected = (rng if rng is not None else random.Random(42)).sample(candidates, count)
    else:
        query = build_similarity_vector(history)
        selected = sorted(candidates, key=lambda origin: np.linalg.norm(
            query - build_similarity_vector(training.iloc[origin-width:origin])))[:count]
    return [(training.iloc[origin-width:origin].copy(), training.iloc[origin:origin+horizon].copy())
            for origin in selected]


LABELS = {
    'water_produced': 'water production (water_produced)',
    'temperature_mean': 'average temperature',
    'temperature_2m_previsao_mean': 'forecast average temperature at two metres',
    'relative_humidity_mean': 'average relative humidity',
    'radiation_mean': 'average radiation',
    'precipitation_mean': 'average precipitation',
    'precipitation_median': 'median precipitation',
    'shortwave_radiation_previsao_std': 'forecast standard deviation of shortwave radiation',
    'day': 'day of the month',
    'dayofweek': 'day of the week',
    'season': 'season of the year',
    'is_holiday': 'public holiday status',
    'is_holiday_ctba_gtba_jve': 'regional holiday status',
    'is_weekend': 'weekend status',
    'is_carnival': 'Carnival status',
    'is_school_recess_pr': 'Paraná school recess status',
}


def _narrate(frame: pd.DataFrame, region: str, *, future: bool = False) -> list[str]:
    text = calendar_to_text(frame)
    start, end = text.timestamp.iloc[0], text.timestamp.iloc[-1]
    sentences = []
    for column in text.columns:
        if column in {'timestamp', 'unique_id', 'observed'}:
            continue
        label = LABELS.get(column, column.replace('_', ' '))
        if column in FLAGS:
            active_dates = text.loc[text[column].eq(FLAGS[column][1]), 'timestamp']
            if not active_dates.empty:
                sentences.append(f'In region {region}, {label} applies on: ' + '; '.join(active_dates) + '.')
            continue
        values = '; '.join(f'{value:.2f}' if column == 'water_produced' else
                           f'{value:.6g}' if isinstance(value, (float, int)) else str(value)
                           for value in text[column])
        unit = {'water_produced': ' cubic metres', 'temperature_mean': ' degrees Celsius',
                'temperature_2m_previsao_mean': ' degrees Celsius'}.get(column, '')
        subject = f'the supplied {label}' if future else f'the {label}'
        verb = 'is' if future else 'was'
        sentences.append(f'From {start} to {end}, {subject} of region {region} '
                         f'{verb} {values}{unit} on each day, respectively.')
    return sentences


def build_prompt(history: pd.DataFrame, horizon: int, *,
                 demonstrations: list | None = None,
                 future_covariates: pd.DataFrame | None = None,
                 region: str = 'the study area') -> str:
    """Narrate historical inputs and target-free supplied covariates in English.

    Region is passed explicitly by the pipeline. Calendar conversion is purely
    representational: no fitting, aggregation, or access to future targets.
    """
    if horizon < 1 or history.empty:
        raise ValueError('History and positive horizon are required')
    if not np.isfinite(history.water_produced.to_numpy(dtype=float)).all():
        raise ValueError('History contains invalid values')
    timestamps = pd.DatetimeIndex(history.timestamp)
    if timestamps.hasnans or not timestamps.equals(pd.date_range(timestamps[0], periods=len(history), freq='D')):
        raise ValueError('History must contain ordered consecutive daily observations')
    dates = future_dates(timestamps[-1], horizon)
    if future_covariates is not None:
        if 'water_produced' in future_covariates.columns:
            raise ValueError('Future target must never appear in a prompt')
        if len(future_covariates) != horizon or not dates.equals(pd.DatetimeIndex(future_covariates.timestamp)):
            raise ValueError('Future covariate dates must match the requested horizon')
        columns = [c for c in future_covariates.columns if c != 'timestamp']
        if not set(columns).issubset(history.columns):
            raise ValueError('Historical and future covariates must match')
    parts = [
        'Forecast daily water production in cubic metres using demand, weather and calendar context.',
        'Each sequence below contains one value per day, aligned with the listed dates in chronological order.',
        f'Return ONLY a JSON object with key "predictions" and exactly {horizon} nonnegative numbers.',
        'Do not explain. Do not repeat the historical values as an answer.',
    ]
    if region.casefold().startswith('guaratuba'):
        parts.append('Guaratuba is a tourist city with strong seasonality during Brazilian holidays '
                     'and school recesses, especially from December through March. '
                     'Consider this context when interpreting changes in water demand.')
    for example_history, example_target in demonstrations or []:
        if example_target.timestamp.max() >= history.timestamp.iloc[-1] + pd.Timedelta(days=1):
            raise ValueError('Demonstration outcome must be known before the forecast origin')
        parts.append('Historical training example:')
        parts.extend(_narrate(example_history, region))
        parts.extend(_narrate(example_target.drop(columns='water_produced'), region, future=True))
        parts.append('Observed production on those example forecast dates: ' +
                     json.dumps({'predictions': example_target.water_produced.tolist()}))
    parts.extend(_narrate(history, region))
    if future_covariates is not None:
        parts.append('The following covariates are supplied inputs for the forecast dates; '
                     'they do not include future water production.')
        parts.extend(_narrate(future_covariates, region, future=True))
    for date in dates:
        parts.append(f'What is the water_produced going to be on {date_to_text(date)} in region {region}?')
    parts.append(f'Output schema: {{"predictions": [number, ...]}}; array length must be {horizon}, '
                 'in the same order as the questions.')
    return '\n'.join(parts)
