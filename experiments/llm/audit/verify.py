"""Audit complete Qwen runs without modifying predictions or the running experiment."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from src.pipelines.data_pipeline import prepare_dataset
from src.pipelines.forecasting_pipeline import ROOT, fingerprint
from src.evaluation.metrics import deterministic_metrics
from src.models.promptcast import parse_prediction

OUTPUT=ROOT/'experiments/llm/audit'
records=[]
checks=[]
references={}
for slug in ['qwen_1_7b','qwen_4b']:
 for days in [7,15]:
  for strategy in ['zero','few_random','few_similar']:
   for city in ['joinville','guaratuba']:
    folder=ROOT/'experiments/llm'/slug/str(days)/strategy/city/'llm'
    run=json.loads((folder/'run.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    assert not status['failures'] and status['successful_windows']==status['expected_windows']
    config=run['config']; spec=config['datasets'][city]
    data,windows=prepare_dataset(ROOT/spec['path'],spec['context'],textual_calendar=spec.get('textual_calendar',False))
    pred=pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
    assert len(windows)==status['expected_windows'] and len(pred)==len(windows)*7
    assert not pred.duplicated(['origin','lead']).any()
    assert np.isfinite(pred[['y_true','y_pred']]).all().all() and pred.y_pred.ge(0).all()
    assert pred.context.eq(days).all()
    for w in windows:
     origin=data.timestamp.iloc[w.origin].strftime('%Y-%m-%d')
     rows=pred[pred.origin==origin]
     saved=json.loads((folder/'responses'/f'{origin}.json').read_text())
     prompt=(folder.parent/'prompts'/f'{origin}.txt').read_text()
     assert saved['status']=='ok' and saved['audit']['request']['prompt']==prompt
     raw=saved['audit']['attempts'][-1]['response']['response']
     np.testing.assert_array_equal(parse_prediction(raw,7),saved['predictions'])
     assert saved['request_id']==fingerprint({'run_id':run['run_id'],'window':saved['window'],'prompt':prompt})
     assert rows.lead.tolist()==list(range(1,8))
     assert rows.timestamp.tolist()==data.iloc[w.target].timestamp.dt.strftime('%Y-%m-%d').tolist()
     np.testing.assert_array_equal(rows.y_true,data.iloc[w.target].water_produced)
     np.testing.assert_allclose(rows.y_pred,saved['predictions'],rtol=1e-14,atol=1e-10)
     assert pd.Timestamp(saved['history_end']) < pd.Timestamp(origin)
     history=data.iloc[w.origin-days:w.origin].water_produced.to_numpy()
     values=np.asarray(saved['predictions'])
     history_copy=any(np.allclose(values,np.round(history[i:i+7],2),atol=0.000001,rtol=0)
                      for i in range(len(history)-6))
     examples=re.findall(r'Observed production on those example forecast dates: (\{[^\n]+\})',prompt)
     example_copy=any(np.allclose(values,json.loads(e)['predictions'],atol=0.000001,rtol=0) for e in examples)
     checks.append(dict(dataset=city,model=slug,context=days,strategy=strategy,origin=origin,
                        history_sequence_match=history_copy,example_sequence_match=example_copy))
    keys=pred[['origin','timestamp','lead','y_true']]
    if city in references: pd.testing.assert_frame_equal(references[city],keys)
    references[city]=keys
    metrics=deterministic_metrics(pred.y_true,pred.y_pred)
    independent_rmse=float(np.sqrt(((pred.y_pred-pred.y_true)**2).mean()))
    assert np.isclose(metrics['rmse'],independent_rmse)
    records.append(dict(dataset=city,model=slug,context=days,strategy=strategy,windows=len(windows),**metrics))
for city,reference in references.items():
 for name in ['chronos','timesfm']:
  pred=pd.read_csv(ROOT/'experiments/benchmark'/city/name/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
  pd.testing.assert_frame_equal(reference,pred[['origin','timestamp','lead','y_true']])
  records.append(dict(dataset=city,model=name,context=365 if city=='joinville' else 730,
                      strategy='expansivo',windows=len(pred)//7,**deterministic_metrics(pred.y_true,pred.y_pred)))
metrics=pd.DataFrame(records)
metrics.to_csv(OUTPUT/'metrics_verified.csv',index=False)
pd.DataFrame(checks).to_csv(OUTPUT/'sequence_checks.csv',index=False)
for city,group in metrics.groupby('dataset'):
 simple=pd.DataFrame({'modelo':group.model+' / '+group.strategy,'contexto_inicial':group.context,
                      'RMSE':group.rmse,'R2':group.r2,'MAE':group.mae,'MAPE':group.mape_percent})
 simple.to_csv(OUTPUT/f'comparison_{city}.csv',index=False)
print('Validated Qwen windows:',len(checks),'predictions:',len(checks)*7)
print('History sequence matches:',sum(c['history_sequence_match'] for c in checks))
print('Few-shot answer matches:',sum(c['example_sequence_match'] for c in checks))
print(metrics.groupby(['dataset','model']).rmse.min().round(2).to_string())
