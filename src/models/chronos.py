"""Chronos-2 with original historical/future covariates and growing context."""
import importlib.metadata


class ChronosForecaster:
    def __init__(self, context: int, dataset: dict):
        import torch
        from chronos import Chronos2Pipeline
        self.dataset = dataset
        torch.manual_seed(42)
        self.pipeline = Chronos2Pipeline.from_pretrained(
            'amazon/chronos-2', device_map='cuda' if torch.cuda.is_available() else 'cpu',
            torch_dtype=torch.float32, local_files_only=True)
        self.identity = {'model': 'amazon/chronos-2', 'initial_context': context,
                         'version': importlib.metadata.version('chronos-forecasting')}
        self.last_audit = {}

    def predict(self, history, horizon: int, *, future_covariates, prompt=None) -> list[float]:
        from src.data.features.covariates import covariate_columns
        if 'water_produced' in future_covariates:
            raise ValueError('Future targets cannot enter Chronos')
        columns = covariate_columns(self.dataset)
        frame = history[['timestamp', 'water_produced'] + columns].copy()
        future = future_covariates[['timestamp'] + columns].copy()
        frame['unique_id'] = future['unique_id'] = 'water'
        self.last_audit = {'history_rows': len(frame), 'context_length': len(frame),
                           'history_start': str(frame.timestamp.iloc[0]),
                           'history_end': str(frame.timestamp.iloc[-1]),
                           'future_columns': future.columns.tolist(), 'covariates': columns}
        result = self.pipeline.predict_df(
            frame, future_df=future, prediction_length=horizon, freq='D', batch_size=1,
            id_column='unique_id', timestamp_column='timestamp', target='water_produced',
            quantile_levels=[0.1, 0.5, 0.9], context_length=len(frame))
        return result['0.5'].to_numpy(dtype=float).tolist()
