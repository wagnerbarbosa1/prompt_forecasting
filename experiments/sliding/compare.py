"""Compare preserved expanding baselines, sliding baselines and reviewed LLMs."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.forecasting_pipeline import ROOT

hashes=json.loads((ROOT/'experiments/sliding/original_baselines_sha256.json').read_text())
for path, expected in hashes.items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected, 'Original baseline changed: '+path

for city in ['joinville','guaratuba']:
    rows=[]
    reference=None
    for name in ['chronos','timesfm']:
        original=ROOT/'experiments/benchmark'/city/name
        frame=pd.read_csv(original/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
        keys=frame[['origin','timestamp','lead','y_true']]
        if reference is not None:
            pd.testing.assert_frame_equal(reference,keys)
        reference=keys
        metrics=deterministic_metrics(frame.y_true,frame.y_pred)
        rows.append({'modelo':name+' / expanding','contexto_inicial':365 if city=='joinville' else 730,
                     'RMSE':metrics['rmse'],'R2':metrics['r2'],'MAE':metrics['mae'],'MAPE':metrics['mape_percent']})
        for days in [7,15]:
            folder=ROOT/'experiments/sliding'/str(days)/city/name
            if not (folder/'status.json').exists(): continue
            status=json.loads((folder/'status.json').read_text())
            if status['failures'] or status['successful_windows']!=status['expected_windows']: continue
            predictions=pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
            pd.testing.assert_frame_equal(reference,predictions[['origin','timestamp','lead','y_true']])
            rows.extend(pd.read_csv(folder/'comparison.csv').to_dict('records'))
    for comparison in (ROOT/'experiments/copy_review').glob(f'*/*/*/{city}/llm/comparison.csv'):
        predictions=pd.read_csv(comparison.with_name('predictions.csv')).sort_values(['origin','lead']).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference,predictions[['origin','timestamp','lead','y_true']])
        rows.extend(pd.read_csv(comparison).to_dict('records'))
    pd.DataFrame(rows).to_csv(ROOT/'experiments/sliding'/f'comparison_{city}.csv',index=False)
    print(city,len(rows),'comparison rows',flush=True)
