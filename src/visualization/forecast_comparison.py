"""Render completed forecasts using the visualization style copied from Siaqua."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from . import plots

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'experiments/forecast_plots'
MODELS = [
    ('chronos2', 'Chronos2', 'experiments/benchmark', 'chronos', 'expansivo'),
    ('timesfm25', 'TimesFM2.5', 'experiments/benchmark', 'timesfm', 'expansivo'),
    ('qwen17b_15', 'Qwen3 1.7B', 'experiments/benchmark', 'qwen', '15 dias'),
    ('qwen4b_15', 'Qwen3 4B', 'experiments/qwen_15_days', 'qwen', '15 dias'),
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Reuse the original saver with a destination inside this experiment.
    plots.OUTPUT_MODELS_FOLDER = OUTPUT
    records = []
    for dataset in ('joinville', 'guaratuba'):
        reference = None
        for slug, label, root, model, policy in MODELS:
            source = ROOT / root / dataset / model
            status = json.loads((source / 'status.json').read_text())
            if status['failures'] or status['successful_windows'] != status['expected_windows']:
                raise ValueError(f'Incomplete predictions: {source}')
            csv = source / 'predictions.csv'
            frame = pd.read_csv(csv).sort_values(['timestamp', 'lead']).reset_index(drop=True)
            if frame.timestamp.duplicated().any() or len(frame) != status['expected_windows'] * 7:
                raise ValueError('Duplicate dates or incomplete forecast horizon')
            if not frame.groupby('origin').lead.apply(lambda x: sorted(x) == list(range(1,8))).all():
                raise ValueError('Missing forecast leads')
            keys = frame[['timestamp', 'y_true']]
            if reference is not None:
                pd.testing.assert_frame_equal(reference, keys)
            reference = keys
            plot_frame = frame[['timestamp', 'y_true', 'y_pred']].rename(
                columns={'y_true': 'Real', 'y_pred': 'Predito'})
            name = f'{dataset}_{slug}'
            title = f'{dataset.title()} — {label} | contexto {policy}'
            plots.linear_graph(plot_frame, timestamp_col='timestamp', highlight_col='Predito',
                compare_cols=['Real'], output_type='output_models', model_name=name,
                title=title, filename='real_vs_predito.png')
            destination = OUTPUT / f'output_{name}' / f'{name}_graph' / 'real_vs_predito.png'
            records.append(dict(dataset=dataset,model=label,context=policy,title=title,
                n_predictions=len(frame),first_date=frame.timestamp.iloc[0],last_date=frame.timestamp.iloc[-1],
                context_first=int(frame.context.iloc[0]),context_last=int(frame.context.iloc[-1]),
                predictions_csv=str(csv.relative_to(ROOT)),source_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                image=str(destination.relative_to(ROOT))))
    pd.DataFrame(records).to_csv(OUTPUT / 'manifest.csv', index=False)
    print(f'{len(records)} plots generated in {OUTPUT}')


if __name__ == '__main__':
    main()
