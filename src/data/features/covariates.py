"""One explicit covariate contract, shared by Chronos, TimesFM and Qwen."""
import numpy as np
import pandas as pd


def covariate_columns(dataset: dict) -> list[str]:
    columns = dataset['numerical_covariates'] + dataset['categorical_covariates']
    if len(columns) != len(set(columns)) or {'timestamp', 'water_produced', 'observed', 'unique_id'} & set(columns):
        raise ValueError('Covariates must be unique and cannot contain the target or metadata')
    return columns


def validate_covariates(df: pd.DataFrame, dataset: dict) -> None:
    columns = covariate_columns(dataset)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'Missing configured covariates: {missing}')
    if df[columns].isna().any().any():
        raise ValueError('Covariates contain missing values')
    for column in dataset['numerical_covariates']:
        if not np.isfinite(pd.to_numeric(df[column], errors='raise').to_numpy()).all():
            raise ValueError(f'Nonfinite covariate: {column}')


def model_inputs(df, window, dataset, *, history_days=None):
    """Exclude target structurally from the future frame, including for Qwen."""
    columns = covariate_columns(dataset)
    start = window.context_start if history_days is None else max(window.context_start, window.origin-history_days)
    history = df.iloc[start:window.origin][['timestamp', 'water_produced'] + columns].copy()
    future = df.iloc[window.target][['timestamp'] + columns].copy()
    if len(future) != window.horizon or history.timestamp.max() >= future.timestamp.min():
        raise ValueError('History and future covariates must be temporally disjoint')
    return history, future


def timesfm_covariates(history, future, dataset):
    if 'water_produced' in future:
        raise ValueError('Future targets cannot enter model inputs')
    columns = covariate_columns(dataset)
    full = pd.concat([history[columns], future[columns]], ignore_index=True)
    numerical = {c: [full[c].to_numpy(dtype=float)] for c in dataset['numerical_covariates']}
    categorical = {c: [full[c].astype(str).tolist()] for c in dataset['categorical_covariates']}
    return numerical, categorical
