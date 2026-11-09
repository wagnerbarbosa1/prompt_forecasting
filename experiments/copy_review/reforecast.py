"""Retry copied sequences without consulting future targets; preserve original runs."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion.files import write_json
from src.evaluation.metrics import deterministic_metrics
from src.models.promptcast import OllamaConfig, OllamaForecaster
from src.pipelines.forecasting_pipeline import ROOT, fingerprint

OUT = ROOT / 'experiments/copy_review'
NUMBER = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'


def copied_sequences(predictions, prompt, minimum=3):
    """Report contiguous matches of >=3 values, including full-horizon matches."""
    sources = []
    for line in prompt.splitlines():
        if line.startswith('From ') and ' on each day, respectively.' in line:
            match = re.search(r' (?:was|is) (.+?) on each day, respectively\.', line)
            if match:
                values = [float(v) for v in re.findall(NUMBER, match[1])]
                if values:
                    sources.append(('water_history' if 'water_produced' in line else 'covariate', values))
        if line.startswith('Observed production on those example forecast dates: '):
            sources.append(('example_answer', json.loads(line.split(': ', 1)[1])['predictions']))
    matches = []
    for kind, values in sources:
        longest = 0
        position = None
        for i in range(len(predictions)):
            for j in range(len(values)):
                length = 0
                while i+length < len(predictions) and j+length < len(values):
                    if not np.isclose(predictions[i+length], values[j+length], rtol=0, atol=1e-6):
                        break
                    length += 1
                if length > longest:
                    longest, position = length, (i, j)
        if longest >= minimum:
            matches.append({'source': kind, 'length': longest,
                            'prediction_start': position[0], 'source_start': position[1]})
    return matches


def main():
    reports = []
    models = {}
    for run_path in sorted((ROOT / 'experiments/llm').rglob('run.json')):
        folder = run_path.parent
        status_path = folder / 'status.json'
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status['failures'] or status['successful_windows'] != status['expected_windows']:
            continue
        run = json.loads(run_path.read_text())
        relative = folder.relative_to(ROOT / 'experiments/llm')
        destination = OUT / relative
        predictions = pd.read_csv(folder / 'predictions.csv')
        window_records = []
        for response_path in sorted((folder / 'responses').glob('*.json')):
            saved = json.loads(response_path.read_text())
            if saved['status'] != 'ok':
                continue
            prompt = saved['audit']['request']['prompt']
            matches = copied_sequences(saved['predictions'], prompt)
            if not matches:
                continue
            key = fingerprint({'parent_request_id': saved['request_id'], 'policy': 'three_consecutive_v1'})
            target = destination / 'responses' / response_path.name
            result = json.loads(target.read_text()) if target.exists() else {}
            if result.get('review_id') != key or result.get('status') != 'ok':
                config = dict(run['model']['config'])
                config['seed'] += 1
                identity = fingerprint(config)
                if identity not in models:
                    models[identity] = OllamaForecaster(OllamaConfig(**config))
                model = models[identity]
                correction = ('\nReview instruction: the previous answer repeated at least three consecutive '
                              'values from the supplied input or a historical example. Reassess the forecast '
                              'for the requested future dates using all supplied covariates. Do not return '
                              'an input sequence or an example answer as the forecast. Return only the required JSON.')
                result = {'review_id': key, 'parent_request_id': saved['request_id'],
                          'original_predictions': saved['predictions'], 'original_matches': matches,
                          'origin': saved['origin']}
                try:
                    values = model.predict(None, len(saved['predictions']), prompt=prompt + correction)
                    result.update(status='ok', predictions=values,
                                  remaining_matches=copied_sequences(values, prompt))
                except Exception as error:
                    result.update(status='error', error=str(error))
                result['audit'] = model.last_audit
                write_json(target, result)
            row = {'run': str(relative), 'origin': saved['origin'],
                   'longest_original_match': max(m['length'] for m in matches),
                   'status': result['status'],
                   'still_matches': bool(result.get('remaining_matches', [])) if result['status']=='ok' else None}
            window_records.append(row)
            reports.append(row)
            if result['status'] == 'ok':
                mask = predictions.origin.eq(saved['origin'])
                assert mask.sum() == len(result['predictions'])
                indices = predictions.loc[mask].sort_values('lead').index
                predictions.loc[indices, 'y_pred'] = result['predictions']
            print(str(relative), saved['origin'], result['status'], 'still_matches', row['still_matches'], flush=True)
        destination.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(destination / 'predictions.csv', index=False)
        write_json(destination / 'review.json', {'source': str(folder), 'policy': 'three_consecutive_v1',
                   'changed_windows': window_records, 'failed_retries_keep_original': True})
        # Ground truth is used only here, after the retry decisions and model calls.
        metrics = deterministic_metrics(predictions.y_true, predictions.y_pred)
        config = run['config']
        pd.DataFrame([{'modelo': run['model']['model']+' / '+config['strategy']+' / revisado',
                      'contexto_inicial': config['qwen_history_days'], 'RMSE': metrics['rmse'],
                      'R2': metrics['r2'], 'MAE': metrics['mae'], 'MAPE': metrics['mape_percent']}]).to_csv(
                          destination / 'comparison.csv', index=False)
    pd.DataFrame(reports).to_csv(OUT / 'copy_review.csv', index=False)
    print('Reviewed copied windows:', len(reports), flush=True)


if __name__ == '__main__':
    main()
