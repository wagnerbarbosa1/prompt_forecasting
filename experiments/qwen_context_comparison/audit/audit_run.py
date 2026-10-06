"""Audit saved runs without invoking models or modifying inference sources."""
from pathlib import Path
import hashlib
import json
from dataclasses import asdict
import numpy as np
import pandas as pd
from src.pipelines.data_pipeline import prepare_dataset
from src.pipelines.forecasting_pipeline import fingerprint, source_fingerprint
from src.data.features.covariates import model_inputs
from src.data.preprocessing.generate_prompts import build_tabular_prompt
from src.models.promptcast import parse_prediction
from src.data.ingestion.files import read_series
from src.data.preprocessing.resampling import daily_grid
from src.data.preprocessing.missing_values import fill_past_only

ROOT=Path.cwd()
BASE=ROOT/'experiments/qwen_context_comparison'
OUT=BASE/'audit'
checks=[]; window_rows=[]; summary=[]; examples=[]
def check(condition, name, **where):
    checks.append(dict(check=name, passed=bool(condition), **where))

for dataset in ('joinville','guaratuba'):
    paired={}
    for policy in ('15_days','expanding'):
        folder=BASE/policy/dataset/'qwen'
        run=json.loads((folder/'run.json').read_text()); config=run['config']; spec=config['datasets'][dataset]
        frame=pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
        paired[policy]=frame
        df,windows=prepare_dataset(ROOT/spec['path'],spec['context'],horizon=7,step=7)
        raw=read_series(ROOT/spec['path'])
        where=dict(dataset=dataset,policy=policy)
        check(run['source_sha256']==source_fingerprint(),'source_hash',**where)
        check(run['dataset_sha256']==hashlib.sha256((ROOT/spec['path']).read_bytes()).hexdigest(),'dataset_hash',**where)
        check(run['run_id']==fingerprint({k:v for k,v in run.items() if k!='run_id'}),'run_hash',**where)
        check(len(frame)==7*len(windows) and not frame.duplicated(['origin','lead']).any(),'complete_unique_predictions',**where)
        for w in windows:
            origin=str(df.timestamp.iloc[w.origin].date()); loc=dict(**where,origin=origin)
            saved=json.loads((folder/'responses'/f'{origin}.json').read_text())
            h,f=model_inputs(df,w,spec,history_days=config['qwen_history_days'])
            p=build_tabular_prompt(h,7,future_covariates=f)
            request=saved['audit']['request']; answer=saved['audit']['attempts'][-1]['response']
            observed=frame[frame.origin==origin].sort_values('lead')
            check(p==(BASE/policy/dataset/'prompts'/f'{origin}.txt').read_text()==request['prompt'],'exact_prompt_reconstruction',**loc)
            check(saved['request_id']==fingerprint(dict(run_id=run['run_id'],window=asdict(w),prompt=p)),'request_cache_identity',**loc)
            check(saved['window']==asdict(w) and saved['history_rows']==len(h),'window_record',**loc)
            check(h.timestamp.max()<f.timestamp.min() and 'water_produced' not in f,'history_disjoint_future_target_absent',**loc)
            check(observed.timestamp.tolist()==df.iloc[w.target].timestamp.dt.strftime('%Y-%m-%d').tolist(),'forecast_dates_order',**loc)
            check(np.array_equal(observed.y_true,df.iloc[w.target].water_produced),'ground_truth_exact',**loc)
            check(np.array_equal(observed.y_pred,saved['predictions']) and saved['predictions']==parse_prediction(answer['response'],7),'raw_response_equals_csv',**loc)
            check(answer['done'] and answer.get('done_reason')!='length' and answer['prompt_eval_count']+answer['eval_count']<request['options']['num_ctx'],'generation_complete_below_limit',**loc)
            check(len(p.encode())+768<=request['options']['num_ctx'],'full_prompt_budget',**loc)
            mutated=df.copy(); mutated.loc[mutated.index>=w.origin,'water_produced']=987654321.123
            hm,fm=model_inputs(mutated,w,spec,history_days=config['qwen_history_days'])
            check(build_tabular_prompt(hm,7,future_covariates=fm)==p,'future_target_mutation_invariant',**loc)
            # Independent preprocessing on the prefix detects backfill/interpolation leakage.
            past_only=fill_past_only(daily_grid(raw[raw.timestamp < df.timestamp.iloc[w.origin]]))
            expected_history=past_only.tail(len(h))[h.columns].reset_index(drop=True)
            check(h.reset_index(drop=True).equals(expected_history),'independent_past_only_preprocessing',**loc)
            pred=observed.y_pred.to_numpy(); actual=observed.y_true.to_numpy(); err=pred-actual
            diffs=np.diff(pred)
            row=dict(**loc,context=len(h),history_start=str(h.timestamp.iloc[0].date()),history_end=str(h.timestamp.iloc[-1].date()),
                rmse=float(np.sqrt(np.dot(err,err)/len(err))),mae=float(np.abs(err).mean()),bias=float(err.mean()),
                sse=float(np.dot(err,err)),actual_mean=float(actual.mean()),predicted_mean=float(pred.mean()),
                last7_mean=float(h.water_produced.tail(7).mean()),history_mean=float(h.water_produced.mean()),
                nondecreasing=bool(np.all(diffs>=0)),strictly_increasing=bool(np.all(diffs>0)),
                arithmetic_sequence=bool(np.allclose(diffs,diffs[0],atol=1e-6,rtol=0)),
                prompt_tokens=answer['prompt_eval_count'],output_tokens=answer['eval_count'],
                response_created_at=answer.get('created_at'),attempts=len(saved['audit']['attempts']))
            window_rows.append(row)
            examples.append(dict(**loc,actual=actual.tolist(),predicted=pred.tolist()))
        err=frame.y_pred.to_numpy()-frame.y_true.to_numpy(); y=frame.y_true.to_numpy()
        ws=[r for r in window_rows if r['dataset']==dataset and r['policy']==policy]
        metric=dict(**where,n=len(err),rmse=float(np.sqrt(np.dot(err,err)/len(err))),bias=float(err.mean()),
            mae=float(np.abs(err).mean()),r2=float(1-np.dot(err,err)/np.dot(y-y.mean(),y-y.mean())),
            mape=float(np.mean(np.abs(err/y))*100),overprediction_fraction=float(np.mean(err>0)),
            bias_fraction_mse=float(err.mean()**2/np.mean(err**2)),
            nondecreasing_windows=sum(r['nondecreasing'] for r in ws),arithmetic_windows=sum(r['arithmetic_sequence'] for r in ws),
            windows=len(ws),tokens_min=min(r['prompt_tokens'] for r in ws),tokens_max=max(r['prompt_tokens'] for r in ws),
            top5_sse_fraction=float(sum(sorted([r['sse'] for r in ws],reverse=True)[:5])/np.dot(err,err)))
        summary.append(metric)
        stored=pd.read_csv(BASE/'metrics.csv').query('dataset == @dataset and context_policy == @policy').iloc[0]
        check(np.isclose(stored.rmse,metric['rmse'],rtol=1e-12),'independent_rmse',**where)
    a,b=paired['15_days'],paired['expanding']
    check(a[['origin','timestamp','lead','y_true']].equals(b[['origin','timestamp','lead','y_true']]),'paired_same_targets',dataset=dataset)

pd.DataFrame(checks).to_csv(OUT/'checks.csv',index=False)
wdf=pd.DataFrame(window_rows)
wdf.to_csv(OUT/'windows.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
worst=wdf.query("policy == 'expanding'").sort_values('rmse',ascending=False).groupby('dataset',sort=False).head(5)
worst.to_csv(OUT/'worst_windows.csv',index=False)
keys={(r.dataset,r.policy,r.origin) for r in worst.itertuples()}
(OUT/'worst_examples.json').write_text(json.dumps([r for r in examples if (r['dataset'],r['policy'],r['origin']) in keys],indent=2))
print(pd.DataFrame(summary).to_string(index=False))
print('CHECKS',len(checks),'FAILED',sum(not c['passed'] for c in checks))
print(pd.DataFrame([c for c in checks if not c['passed']]).to_string(index=False))
print('WORST WINDOWS\n',worst[['dataset','origin','rmse','bias','actual_mean','predicted_mean','last7_mean']].to_string(index=False))
