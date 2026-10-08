"""Execute expanding 355/30/30 baseline folds with auditable pre-test fitting."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
from src.data.ingestion.files import read_series, write_json
from src.data.preprocessing.resampling import daily_grid
from src.data.preprocessing.missing_values import fill_past_only
from src.evaluation.cross_validation import expanding_validation_folds
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.forecasting_pipeline import ROOT, fingerprint, source_fingerprint
from src.training.classical import select_and_forecast

OUTPUT = ROOT / 'experiments/classical_baselines'


def run(config, output=OUTPUT):
    summaries = []
    source_hash = source_fingerprint()
    for city, spec in config['datasets'].items():
        path = ROOT / spec['path']
        frame = fill_past_only(daily_grid(read_series(path)))
        folds = expanding_validation_folds(frame, config['initial_training'],
                                           config['validation_days'], config['test_days'])
        for name, candidates in config['candidates'].items():
            folder = output / city / name
            provenance = {'dataset': city, 'model': name, 'config': config,
                          'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'source_sha256': source_hash, 'folds': [asdict(f) for f in folds],
                          'unused_tail_days': len(frame)-folds[-1].test_end,
                          'protocol': 'expanding selection training; trailing validation; refit pre-test prefix',
                          'no_future_covariates': True}
            run_id = fingerprint(provenance)
            write_json(folder/'run.json', dict(provenance, run_id=run_id))
            rows, failures = [], []
            for fold in folds:
                origin = frame.timestamp.iloc[fold.validation_end].strftime('%Y-%m-%d')
                record_path = folder/'folds'/f'{origin}.json'
                request_id = fingerprint({'run_id': run_id, 'fold': asdict(fold)})
                record = json.loads(record_path.read_text()) if record_path.exists() else {}
                if record.get('request_id') != request_id or record.get('status') != 'ok':
                    start = time.monotonic()
                    record = {'request_id': request_id, 'fold': asdict(fold), 'origin': origin,
                              'training_end': str(frame.timestamp.iloc[fold.train_end-1].date()),
                              'validation_end': str(frame.timestamp.iloc[fold.validation_end-1].date()),
                              'test_end': str(frame.timestamp.iloc[fold.test_end-1].date())}
                    try:
                        # Only these two pre-test arrays cross the fitting boundary.
                        predictions, audit = select_and_forecast(name,
                            frame.iloc[fold.training].water_produced.to_numpy(),
                            frame.iloc[fold.validation].water_produced.to_numpy(),
                            config['test_days'], candidates)
                        record.update(status='ok', predictions=predictions.tolist(), audit=audit)
                    except Exception as exc:
                        record.update(status='error', error=f'{type(exc).__name__}: {exc}')
                    record['elapsed_seconds'] = time.monotonic()-start
                    write_json(record_path, record)
                print(f'{city}/{name} fold {fold.number}/{len(folds)} {origin}: {record["status"]}', flush=True)
                if record['status'] != 'ok':
                    failures.append({'origin': origin, 'error': record['error']})
                    continue
                # Ground truth is read only after predictions have been produced.
                truth = frame.iloc[fold.test]
                for lead, (observation, predicted) in enumerate(zip(truth.itertuples(), record['predictions']), 1):
                    rows.append({'dataset': city, 'model': name, 'fold': fold.number, 'origin': origin,
                                 'timestamp': observation.timestamp.strftime('%Y-%m-%d'), 'lead': lead,
                                 'y_true': observation.water_produced, 'y_pred': predicted,
                                 'initial_context': config['initial_training'], 'training_days': fold.train_end,
                                 'validation_days': config['validation_days'], 'fit_days': fold.validation_end,
                                 'horizon': config['test_days'], 'run_id': run_id})
            pd.DataFrame(rows).to_csv(folder/'predictions.csv', index=False)
            write_json(folder/'status.json', {'expected_folds': len(folds),
                       'successful_folds': len(rows)//config['test_days'], 'failures': failures})
            if failures:
                raise RuntimeError(f'Incomplete {city}/{name}: {failures}')
            predictions = pd.DataFrame(rows)
            scores = deterministic_metrics(predictions.y_true, predictions.y_pred)
            simple = {'modelo': name, 'contexto_inicial': config['initial_training'],
                      'RMSE': scores['rmse'], 'R2': scores['r2'], 'MAE': scores['mae'], 'MAPE': scores['mape_percent']}
            pd.DataFrame([simple]).to_csv(folder/'comparison.csv', index=False)
            summaries.append(dict(dataset=city, model=name, folds=len(folds), predictions=len(rows), **scores))
        city_rows = [row for row in summaries if row['dataset']==city]
        pd.DataFrame([{'modelo': row['model'], 'contexto_inicial': config['initial_training'],
                      'RMSE': row['rmse'], 'R2': row['r2'], 'MAE': row['mae'], 'MAPE': row['mape_percent']}
                      for row in city_rows]).to_csv(output/f'comparison_{city}.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/'metrics.csv', index=False)
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/classical_baselines.json')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['step_days'] != config['test_days'] or config['refit_train_validation'] is not True:
        raise ValueError('This protocol requires nonoverlapping tests and pre-test refitting')
    run(config, args.output)


if __name__ == '__main__':
    main()
