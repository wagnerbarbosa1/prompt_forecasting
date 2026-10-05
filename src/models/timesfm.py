"""Original TimesFM XReg + TimesFM path, with context growing at each origin."""
import importlib.metadata
import numpy as np
from src.data.features.covariates import timesfm_covariates, covariate_columns


class TimesFMForecaster:
    def __init__(self, context: int, dataset: dict):
        import torch
        import timesfm
        self.dataset = dataset
        torch.manual_seed(42)
        torch.set_float32_matmul_precision('high')
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            'google/timesfm-2.5-200m-pytorch', local_files_only=True)
        self.identity = {'model': 'google/timesfm-2.5-200m-pytorch', 'initial_context': context,
                         'version': importlib.metadata.version('timesfm'), 'xreg_mode': 'xreg + timesfm'}
        self.last_audit = {}

    def predict(self, history, horizon: int, *, future_covariates, prompt=None) -> list[float]:
        import timesfm
        context = len(history)
        self.model.compile(timesfm.ForecastConfig(
            max_context=((context + 31) // 32) * 32,
            max_horizon=((horizon + 127) // 128) * 128,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True,
            fix_quantile_crossing=True, return_backcast=True))
        numerical, categorical = timesfm_covariates(history, future_covariates, self.dataset)
        self.last_audit = {'history_rows': context,
                           'history_start': str(history.timestamp.iloc[0]),
                           'history_end': str(history.timestamp.iloc[-1]),
                           'max_context': ((context+31)//32)*32,
                           'covariates': covariate_columns(self.dataset),
                           'future_rows': len(future_covariates), 'xreg_mode': 'xreg + timesfm'}
        point, _ = self.model.forecast_with_covariates(
            inputs=[history.water_produced.to_numpy(dtype=float)],
            dynamic_numerical_covariates=numerical,
            dynamic_categorical_covariates=categorical,
            xreg_mode='xreg + timesfm')
        return np.asarray(point[0], dtype=float)[:horizon].tolist()
