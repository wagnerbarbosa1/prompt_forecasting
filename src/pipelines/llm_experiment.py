"""Run local LLMs with the original zero/random/similar prompt strategies."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import deterministic_metrics
from src.pipelines.forecasting_pipeline import ROOT, run_dataset


def configurations(spec):
    base = json.loads((ROOT / spec['base_config']).read_text())
    for slug, model in spec['models'].items():
        for days in spec['contexts']:
            for strategy in spec['strategies']:
                config = deepcopy(base)
                config['ollama'] = config.pop('qwen')
                config['ollama'].update(model=model, base_url=spec['base_url'],
                                        num_ctx=spec['num_ctx'], num_predict=spec['num_predict'])
                config.update(qwen_history_days=days, strategy=strategy,
                              few_shot_examples=spec['examples'])
                yield slug, days, strategy, config


def summarize(output, runs):
    """Compare complete runs on identical forecast dates, including all seven leads."""
    for city in ('joinville', 'guaratuba'):
        rows, reference = [], None
        for slug, days, strategy, config in runs:
            folder = output / slug / str(days) / strategy / city / 'llm'
            status = json.loads((folder / 'status.json').read_text())
            if status['failures'] or status['successful_windows'] != status['expected_windows']:
                raise ValueError(f'Incomplete run: {folder}')
            frame = pd.read_csv(folder / 'predictions.csv').sort_values(['origin', 'lead']).reset_index(drop=True)
            if not frame.context.eq(days).all() or not frame.groupby('origin').lead.apply(
                    lambda leads: sorted(leads) == list(range(1, 8))).all():
                raise ValueError(f'Invalid forecast windows: {folder}')
            keys = frame[['origin', 'timestamp', 'lead', 'y_true']]
            if reference is not None:
                pd.testing.assert_frame_equal(reference, keys)
            reference = keys
            metrics = deterministic_metrics(frame.y_true, frame.y_pred)
            row = {'modelo': config['ollama']['model'] + ' / ' + strategy,
                   'contexto_inicial': days, 'RMSE': metrics['rmse'], 'R2': metrics['r2'],
                   'MAE': metrics['mae'], 'MAPE': metrics['mape_percent']}
            pd.DataFrame([row]).to_csv(folder / 'comparison.csv', index=False)
            rows.append(row)
        pd.DataFrame(rows).to_csv(output / f'comparison_{city}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/llm_experiment.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/llm')
    parser.add_argument('--models', nargs='+', help='Model keys from the configuration')
    parser.add_argument('--strategies', nargs='+', choices=['zero', 'few_random', 'few_similar'])
    parser.add_argument('--limit', type=int, default=0, help='Windows per dataset; zero runs all')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 0 or (args.prepare_only and args.summarize_only):
        parser.error('Use a nonnegative limit and one execution mode')
    spec = json.loads(args.config.read_text())
    if args.models:
        if set(args.models) - set(spec['models']):
            parser.error('Unknown model key')
        spec['models'] = {key: spec['models'][key] for key in args.models}
    if args.strategies:
        spec['strategies'] = args.strategies
    runs = list(configurations(spec))
    if not args.summarize_only:
        for slug, days, strategy, config in runs:
            for city, dataset in config['datasets'].items():
                destination = args.output / slug / str(days) / strategy
                failures = run_dataset(city, dataset, config, 'llm', destination,
                                       limit=args.limit, prepare_only=args.prepare_only)
                if failures:
                    raise RuntimeError(f'{failures} failed windows: {slug}/{days}/{strategy}/{city}')
    if not args.prepare_only:
        summarize(args.output, runs)


if __name__ == '__main__':
    main()
