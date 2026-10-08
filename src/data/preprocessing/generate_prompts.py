"""PromptCast prompts with demand and the same covariates used by the baselines; future targets are absent by construction."""
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.data.features.temporal import future_dates
from src.data.preprocessing.calendar_text import calendar_to_text, date_to_text


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
    sentences = [
        ('The forecast dates' if future else 'The observation dates') +
        ' in chronological order are: ' + '; '.join(text.timestamp) + '.',
    ]
    for column in text.columns:
        if column in {'timestamp', 'unique_id', 'observed'}:
            continue
        label = LABELS.get(column, column.replace('_', ' '))
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
                 demonstrations: list[Demonstration] | None = None,
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
    for example in demonstrations or []:
        parts.append('In a historical training example, daily water production was ' +
                     format_values(example.history) + ' cubic metres, respectively.')
        parts.append('The following daily production values were: ' +
                     json.dumps({'predictions': list(example.target)}))
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
