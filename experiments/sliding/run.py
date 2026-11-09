"""Original SlidingWindow with the same evaluation origins as expanding baselines."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion.files import write_json
from src.evaluation.cross_validation import SlidingWindow
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.data_pipeline import prepare_dataset
from src.pipelines.forecasting_pipeline import ROOT, create_model, fingerprint, source_fingerprint
from src.data.features.covariates import covariate_columns, validate_covariates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['chronos', 'timesfm'], required=True)
    parser.add_argument('--days', type=int, choices=[7,15], required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/benchmark.json').read_text())
    for city, dataset in config['datasets'].items():
        data, windows = prepare_dataset(ROOT/dataset['path'], dataset['context'], horizon=7, step=7)
        validate_covariates(data, dataset)
        # Offset the original splitter so the first target remains at 365/730.
        offset = dataset['context'] - args.days
        split = SlidingWindow(len(data)-offset, args.days, 7, 7)
        expected = [w.origin for w in windows]
        pairs = [(slice(a.start+offset,a.stop+offset),slice(b.start+offset,b.stop+offset))
                 for a,b in split.split() if data.iloc[b.start+offset:b.stop+offset].observed.all()]
        assert [b.start for _,b in pairs] == expected
        if args.limit:
            pairs = pairs[:args.limit]
        model = create_model(args.model, args.days, config, dataset)
        folder = ROOT/'experiments/sliding'/str(args.days)/city/args.model
        provenance = {'model':model.identity,'dataset':city,'history_days':args.days,
                      'evaluation_initial_context':dataset['context'],'horizon':7,'step':7,
                      'covariates':covariate_columns(dataset),'source_sha256':source_fingerprint(),
                      'dataset_sha256':hashlib.sha256((ROOT/dataset['path']).read_bytes()).hexdigest(),
                      'limit':args.limit,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        run_id = fingerprint(provenance)
        write_json(folder/'run.json',dict(provenance,run_id=run_id))
        rows, errors = [], []
        columns = covariate_columns(dataset)
        for i,(trainidxs,testidxs) in enumerate(pairs,1):
            history = data.iloc[trainidxs][['timestamp','water_produced']+columns].copy()
            future = data.iloc[testidxs][['timestamp']+columns].copy()
            assert len(history)==args.days and history.timestamp.max()<future.timestamp.min()
            origin = future.timestamp.iloc[0].strftime('%Y-%m-%d')
            path = folder/'responses'/f'{origin}.json'
            key = fingerprint({'run_id':run_id,'origin':origin})
            saved = json.loads(path.read_text()) if path.exists() else {}
            if saved.get('request_id') != key or saved.get('status') != 'ok':
                saved = {'request_id':key,'origin':origin}
                try:
                    values = model.predict(history,7,future_covariates=future)
                    if len(values)!=7 or not np.isfinite(values).all():
                        raise ValueError('Expected seven finite predictions')
                    saved.update(status='ok',predictions=values,audit=model.last_audit)
                except Exception as error:
                    saved.update(status='error',error=str(error))
                write_json(path,saved)
            print(city,args.model,args.days,i,len(pairs),saved['status'],flush=True)
            if saved['status']!='ok':
                errors.append({'origin':origin,'error':saved['error']})
                continue
            for lead, ((_, actual), value) in enumerate(zip(data.iloc[testidxs].iterrows(),saved['predictions']),1):
                rows.append({'dataset':city,'model':args.model,'origin':origin,'timestamp':actual.timestamp.strftime('%Y-%m-%d'),
                             'lead':lead,'y_true':actual.water_produced,'y_pred':value,'context':args.days,
                             'initial_context':dataset['context'],'history_start':str(history.timestamp.iloc[0].date())})
        frame = pd.DataFrame(rows)
        frame.to_csv(folder/'predictions.csv',index=False)
        write_json(folder/'status.json',{'expected_windows':len(pairs),'successful_windows':len(rows)//7,'failures':errors})
        if not errors:
            reference = pd.read_csv(ROOT/'experiments/benchmark'/city/args.model/'predictions.csv').iloc[:len(frame)]
            pd.testing.assert_frame_equal(frame[['origin','timestamp','lead','y_true']].reset_index(drop=True),
                                          reference[['origin','timestamp','lead','y_true']].reset_index(drop=True))
            metrics = deterministic_metrics(frame.y_true,frame.y_pred)
            pd.DataFrame([{'modelo':args.model+' / sliding','contexto_inicial':args.days,
                           'RMSE':metrics['rmse'],'R2':metrics['r2'],'MAE':metrics['mae'],
                           'MAPE':metrics['mape_percent']}]).to_csv(folder/'comparison.csv',index=False)


if __name__=='__main__':
    main()
