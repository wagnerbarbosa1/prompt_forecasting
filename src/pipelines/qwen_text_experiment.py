"""Prepare, execute and summarize Qwen runs sourced from the textual CSVs."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.features.covariates import model_inputs, validate_covariates
from src.data.ingestion.files import write_json
from src.data.preprocessing.generate_prompts import build_prompt
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.data_pipeline import prepare_dataset
from src.pipelines.forecasting_pipeline import ROOT, run_dataset

OUTPUT = ROOT / 'experiments/qwen_text'
RUNS = [('1_7b', 7), ('1_7b', 15), ('4b', 7), ('4b', 15)]


def configuration(slug, days):
    return json.loads((ROOT / f'configs/qwen_text_{slug}_{days}.json').read_text())


def prepare():
    records = []
    for slug, days in RUNS:
        config = configuration(slug, days)
        for name, dataset in config['datasets'].items():
            frame, windows = prepare_dataset(ROOT / dataset['path'], dataset['context'],
                                            textual_calendar=True)
            validate_covariates(frame, dataset)
            destination = OUTPUT / f'{slug}_{days}' / name / 'prompts'
            destination.mkdir(parents=True, exist_ok=True)
            maximum = 0
            for window in windows:
                history, future = model_inputs(frame, window, dataset, history_days=days)
                prompt = build_prompt(history, 7, future_covariates=future, region=name.capitalize())
                required = len(prompt.encode()) + config['qwen']['num_predict'] + 512
                if required > config['qwen']['num_ctx']:
                    raise ValueError(f'Prompt does not fit: {slug}/{days}/{name}')
                maximum = max(maximum, required)
                origin = future.timestamp.iloc[0].strftime('%Y-%m-%d')
                (destination / f'{origin}.txt').write_text(prompt)
            records.append(dict(model=slug, context=days, dataset=name, windows=len(windows),
                                source=dataset['path'], maximum_byte_budget=maximum))
    write_json(OUTPUT / 'prompt_manifest.json', records)
    print(f'Prepared {sum(r["windows"] for r in records)} prompts from textual CSVs.', flush=True)


def summarize():
    rows, per_lead = [], []
    for name in ('joinville', 'guaratuba'):
        reference = None
        simple = []
        for slug, days in RUNS:
            folder = OUTPUT / f'{slug}_{days}' / name / 'qwen'
            status = json.loads((folder / 'status.json').read_text())
            run = json.loads((folder / 'run.json').read_text())
            if status['failures'] or status['successful_windows'] != status['expected_windows']:
                raise ValueError(f'Incomplete run: {folder}')
            frame = pd.read_csv(folder / 'predictions.csv').sort_values(['origin', 'lead']).reset_index(drop=True)
            if not frame.context.eq(days).all() or frame.duplicated(['origin', 'lead']).any():
                raise ValueError(f'Invalid contexts/duplicate forecasts: {folder}')
            if len(frame) != status['expected_windows'] * 7 or not frame.groupby('origin').lead.apply(
                    lambda values: sorted(values) == list(range(1, 8))).all():
                raise ValueError('Missing forecast leads')
            keys = frame[['origin', 'timestamp', 'lead', 'y_true']]
            if reference is not None:
                pd.testing.assert_frame_equal(reference, keys)
            reference = keys
            metrics = deterministic_metrics(frame.y_true, frame.y_pred)
            label = 'Qwen3 1.7B' if slug == '1_7b' else 'Qwen3 4B'
            rows.append(dict(dataset=name, model=label, context=days, slug=slug,
                             windows=status['successful_windows'], predictions=len(frame), **metrics))
            simple.append({'modelo': label, 'contexto_inicial': days, 'RMSE': metrics['rmse'],
                           'R2': metrics['r2'], 'MAE': metrics['mae'], 'MAPE': metrics['mape_percent']})
            pd.DataFrame([simple[-1]]).to_csv(folder / 'comparison.csv', index=False)
            for lead, subset in frame.groupby('lead'):
                per_lead.append(dict(dataset=name, model=label, context=days, lead=int(lead),
                                     **deterministic_metrics(subset.y_true, subset.y_pred)))
        pd.DataFrame(simple).to_csv(OUTPUT / f'comparison_{name}.csv', index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / 'metrics.csv', index=False)
    pd.DataFrame(per_lead).to_csv(OUTPUT / 'metrics_by_horizon.csv', index=False)
    print(pd.DataFrame(rows)[['dataset', 'model', 'context', 'rmse', 'r2']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if not args.summarize_only:
        prepare()
        if args.prepare_only:
            return
        for slug, days in RUNS:
            config = configuration(slug, days)
            for name, dataset in config['datasets'].items():
                failures = run_dataset(name, dataset, config, 'qwen', OUTPUT / f'{slug}_{days}')
                if failures:
                    raise RuntimeError(f'{failures} failed windows in {slug}/{days}/{name}')
    summarize()


if __name__ == '__main__':
    main()
