"""Univariate classical baselines with one explicit fit/predict contract."""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.data.features.autoregressive import lagged_samples, recursive_forecast


class LagRegressor:
    def __init__(self, name: str, *, lags: int, neighbors=5, weights='uniform'):
        self.lags = lags
        if name == 'linear_regression':
            regressor = LinearRegression()
        elif name == 'knn':
            regressor = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, n_jobs=1)
        else:
            raise ValueError(f'Unknown lag regressor: {name}')
        self.estimator = make_pipeline(StandardScaler(), regressor)

    def fit(self, history):
        self.history = np.asarray(history, dtype=float).copy()
        features, targets = lagged_samples(self.history, self.lags)
        self.estimator.fit(features, targets)
        return self

    def predict(self, horizon: int):
        return recursive_forecast(self.estimator, self.history, horizon, self.lags)


class ARIMAForecaster:
    def __init__(self, order):
        self.order = tuple(order)

    def fit(self, history):
        values = np.asarray(history, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite ARIMA history')
        self.mean = float(values.mean())
        self.scale = max(float(values.std()), 1e-8)
        scaled = (values - self.mean) / self.scale
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            self.result = ARIMA(scaled, order=self.order).fit(method_kwargs={'maxiter': 200})
        return self

    def predict(self, horizon: int):
        values = np.asarray(self.result.forecast(steps=horizon))*self.scale + self.mean
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite ARIMA predictions')
        return values


def create_classical_model(name, parameters):
    if name == 'arima':
        return ARIMAForecaster(**parameters)
    return LagRegressor(name, **parameters)
