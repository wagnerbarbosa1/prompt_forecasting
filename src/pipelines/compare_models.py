"""Refuse incomplete or mismatched comparisons; report aggregate and per-lead metrics."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.forecasting_pipeline import ROOT


def compare(output: Path):
    summaries, per_lead = [], []
    for dataset in ('joinville', 'guaratuba'):
        reference = None
        reference_run = None
        for model in ('qwen', 'chronos', 'timesfm'):
            folder = output / dataset / model
            status = json.loads((folder / 'status.json').read_text())
            run = json.loads((folder / 'run.json').read_text())
            if status['failures'] or status['successful_windows'] != status['expected_windows'] or run['limit']:
                raise ValueError(f'Incomplete benchmark: {dataset}/{model}')
            frame = pd.read_csv(folder / 'predictions.csv').sort_values(['origin', 'lead']).reset_index(drop=True)
            keys = frame[['origin', 'timestamp', 'lead', 'y_true', 'initial_context', 'horizon']]
            origins = frame.drop_duplicates('origin')
            initial = int(frame.initial_context.iloc[0])
            expected_contexts = ([run['qwen_history_days']] * len(origins) if model == 'qwen'
                                 else [initial + 7*i for i in range(len(origins))])
            if origins.context.tolist() != expected_contexts:
                raise ValueError(f'Incorrect history lengths: {dataset}/{model}')
            if model != 'qwen' and origins.history_start.nunique() != 1:
                raise ValueError('Expanding history must always start on the first date')
            if frame.duplicated(['origin', 'lead']).any() or len(frame) != status['expected_windows'] * 7:
                raise ValueError('Duplicate or missing forecasts')
            comparable = {key: run[key] for key in ('config', 'dataset_sha256', 'source_sha256', 'n_windows')}
            if reference is not None and (not keys.equals(reference) or comparable != reference_run):
                raise ValueError(f'Incomparable evaluation for {dataset}/{model}')
            reference, reference_run = keys, comparable
            common = {'dataset': dataset, 'model': model, 'model_id': run['model']['model'],
                      'initial_context_days': int(frame.initial_context.iloc[0]),
                      'context_days_first': int(frame.context.iloc[0]),
                      'context_days_last': int(frame.context.iloc[-1]),
                      'history_policy': 'last 15 days' if model == 'qwen' else 'expanding',
                      'covariates': '|'.join(run['covariates']), 'horizon_days': 7, 'step_days': 7,
                      'strategy': run['config']['strategy'] if model == 'qwen' else 'pretrained',
                      'n_windows': status['expected_windows'], 'n_predictions': len(frame),
                      'test_start': frame.timestamp.min(), 'test_end': frame.timestamp.max()}
            timing = frame.drop_duplicates('origin').elapsed_seconds_window
            summaries.append(dict(common, **deterministic_metrics(frame.y_true, frame.y_pred),
                                  inference_seconds=float(timing.sum()), mean_window_seconds=float(timing.mean())))
            for lead, subset in frame.groupby('lead'):
                per_lead.append(dict(dataset=dataset, model=model, lead=int(lead),
                                     **deterministic_metrics(subset.y_true, subset.y_pred)))
    pd.DataFrame(summaries).to_csv(output / 'metrics_comparison.csv', index=False)
    pd.DataFrame(per_lead).to_csv(output / 'metrics_by_horizon.csv', index=False)
    print(pd.DataFrame(summaries)[['dataset', 'model', 'mae', 'rmse', 'mape_percent', 'r2']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/benchmark')
    compare(parser.parse_args().output)


if __name__ == '__main__':
    main()
