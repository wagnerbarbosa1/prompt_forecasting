"""Run one model/environment with shared windows and resumable, fingerprinted records."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.data.ingestion.files import write_json
from src.data.preprocessing.generate_prompts import build_prompt, build_tabular_prompt, select_demonstrations
from src.pipelines.data_pipeline import prepare_dataset
from src.data.features.covariates import model_inputs, validate_covariates, covariate_columns

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_fingerprint() -> str:
    """Invalidate cached inference when any model/data/pipeline source changes."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'src').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def create_model(name, context, config, dataset):
    if name == 'qwen':
        from src.models.promptcast import OllamaConfig, QwenForecaster
        return QwenForecaster(OllamaConfig(**config['qwen'], seed=config['seed']))
    if name == 'chronos':
        from src.models.chronos import ChronosForecaster
        return ChronosForecaster(context, dataset)
    if name == 'timesfm':
        from src.models.timesfm import TimesFMForecaster
        return TimesFMForecaster(context, dataset)
    raise ValueError(f'Unknown model {name}')


def run_dataset(name, dataset, config, model_name, output, limit=0):
    path = ROOT / dataset['path']
    df, windows = prepare_dataset(path, dataset['context'], horizon=config['horizon'],
                                  step=config['step'])
    validate_covariates(df, dataset)
    if limit:
        windows = windows[:limit]
    model = create_model(model_name, dataset['context'], config, dataset)
    provenance = {'config': config, 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'source_sha256': source_fingerprint(), 'model': model.identity,
                  'dataset': name, 'limit': limit, 'n_windows': len(windows),
                  'first_test_origin': windows[0].origin,
                  'missing_days_forward_filled': df.loc[~df.observed, 'timestamp'].dt.strftime('%Y-%m-%d').tolist(),
                  'protocol': 'expanding history with historical and supplied future covariates',
                  'covariates': covariate_columns(dataset),
                  'future_covariates_policy': config['future_covariates_policy'],
                  'qwen_history_days': config['qwen_history_days']}
    run_id = fingerprint(provenance)
    destination = output / name / model_name
    write_json(destination / 'run.json', dict(provenance, run_id=run_id))
    # This prefix is frozen for every window, including all few-shot strategies.
    training_values = df.iloc[:windows[0].origin].water_produced.to_numpy()
    rows, failures = [], []
    for index, window in enumerate(windows, 1):
        history, future = model_inputs(df, window, dataset,
            history_days=config['qwen_history_days'] if model_name == 'qwen' else None)
        origin_date = df.timestamp.iloc[window.origin].strftime('%Y-%m-%d')
        target = destination / 'responses' / f'{origin_date}.json'
        prompt = None
        if model_name == 'qwen':
            examples = select_demonstrations(training_values, history.water_produced.to_numpy(),
                window.horizon, strategy=config['strategy'], count=config['few_shot_examples'], seed=config['seed'])
            if config.get('prompt_layout', 'legacy') == 'tabular':
                if examples:
                    raise ValueError('The controlled tabular experiment uses zero-shot prompts')
                prompt = build_tabular_prompt(history, window.horizon, future_covariates=future)
            else:
                prompt = build_prompt(history, window.horizon, demonstrations=examples, future_covariates=future)
            prompt_path = output / name / 'prompts' / f'{origin_date}.txt'
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt)
        request_id = fingerprint({'run_id': run_id, 'window': asdict(window), 'prompt': prompt})
        saved = json.loads(target.read_text()) if target.exists() else {}
        cached = saved.get('request_id') == request_id and saved.get('status') == 'ok'
        if not cached:
            started = time.monotonic()
            saved = {'request_id': request_id, 'window': asdict(window), 'origin': origin_date,
                     'history_rows': len(history), 'history_start': str(history.timestamp.iloc[0]),
                     'history_end': str(history.timestamp.iloc[-1]), 'covariates': covariate_columns(dataset)}
            try:
                predictions = model.predict(history, window.horizon, prompt=prompt, future_covariates=future)
                if len(predictions) != window.horizon or not np.isfinite(predictions).all():
                    raise ValueError('Invalid model output shape or finite values')
                saved.update(status='ok', predictions=predictions)
            except Exception as exc:
                saved.update(status='error', error=f'{type(exc).__name__}: {exc}')
            saved.update(elapsed_seconds=time.monotonic()-started, audit=model.last_audit)
            write_json(target, saved)
        print(f'{name}/{model_name} {index}/{len(windows)} {origin_date}: {saved["status"]}'
              + (' cached' if cached else ''), flush=True)
        if saved['status'] != 'ok':
            failures.append({'origin': origin_date, 'error': saved['error']})
            continue
        # Ground truth is accessed only AFTER model inference finishes.
        truth = df.iloc[window.target]
        for lead, ((_, observation), prediction) in enumerate(zip(truth.iterrows(), saved['predictions']), 1):
            rows.append({'dataset': name, 'model': model_name, 'origin': origin_date,
                         'timestamp': observation.timestamp.strftime('%Y-%m-%d'), 'lead': lead,
                         'y_true': float(observation.water_produced), 'y_pred': prediction,
                         'initial_context': dataset['context'], 'context': len(history),
                         'history_start': history.timestamp.iloc[0].strftime('%Y-%m-%d'),
                         'horizon': window.horizon,
                         'elapsed_seconds_window': saved['elapsed_seconds'], 'run_id': run_id})
    pd.DataFrame(rows).to_csv(destination / 'predictions.csv', index=False)
    write_json(destination / 'status.json', {'expected_windows': len(windows),
               'successful_windows': len(rows)//config['horizon'], 'failures': failures})
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen', 'chronos', 'timesfm'], default='qwen')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/benchmark.json')
    parser.add_argument('--dataset', choices=['all', 'joinville', 'guaratuba'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/benchmark')
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 evaluates every window')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    config = json.loads(args.config.read_text())
    failures = 0
    for name, dataset in config['datasets'].items():
        if args.dataset in ('all', name):
            failures += run_dataset(name, dataset, config, args.model, args.output, args.limit)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
