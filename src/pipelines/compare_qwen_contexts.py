"""Compare 15 days versus expanding history for the SAME Qwen checkpoint/runtime."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.forecasting_pipeline import ROOT


def relative_reduction(short, long):
    """Positive is an improvement; do not invent a percentage for zero baseline error."""
    return 100 * (short-long) / short if short else None


def load_completed_run(root, dataset, label):
    folder = root / label / dataset / 'qwen'
    status = json.loads((folder/'status.json').read_text())
    run = json.loads((folder/'run.json').read_text())
    if run['limit'] or status['failures'] or status['successful_windows'] != status['expected_windows']:
        raise ValueError(f'Incomplete run: {dataset}/{label}')
    frame = pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
    if len(frame) != status['expected_windows']*7 or frame.duplicated(['origin','lead']).any():
        raise ValueError('Missing or duplicated forecasts')
    if not frame.groupby('origin').lead.apply(lambda x: sorted(x)==list(range(1,8))).all():
        raise ValueError('Every forecast must contain days 1 through 7')
    origins = frame.drop_duplicates('origin')
    initial = int(frame.initial_context.iloc[0])
    expected = [15]*len(origins) if label=='15_days' else list(range(initial,initial+7*len(origins),7))
    if origins.context.tolist() != expected:
        raise ValueError('Unexpected historical context lengths')
    if label=='expanding' and origins.history_start.nunique()!=1:
        raise ValueError('Expanding history must start on the first date')
    return frame, run


def compare_contexts(output):
    metrics, changes, lead_metrics, lead_changes = [], [], [], []
    for dataset in ('joinville','guaratuba'):
        frames, runs, scores = {}, {}, {}
        for label in ('15_days','expanding'):
            frame, run = load_completed_run(output,dataset,label)
            frames[label], runs[label] = frame, run
            score = deterministic_metrics(frame.y_true,frame.y_pred)
            scores[label] = score
            metrics.append(dict(dataset=dataset,context_policy=label,model=run['model']['model'],
                context_first=int(frame.context.iloc[0]),context_last=int(frame.context.iloc[-1]),
                num_ctx=run['config']['qwen']['num_ctx'],n_windows=frame.origin.nunique(),n_predictions=len(frame),
                **score,inference_seconds=float(frame.drop_duplicates('origin').elapsed_seconds_window.sum())))
            for lead,subset in frame.groupby('lead'):
                lead_metrics.append(dict(dataset=dataset,context_policy=label,lead=int(lead),
                                         **deterministic_metrics(subset.y_true,subset.y_pred)))
        columns=['origin','timestamp','lead','y_true','initial_context','horizon']
        pd.testing.assert_frame_equal(frames['15_days'][columns],frames['expanding'][columns])
        for field in ['dataset_sha256','source_sha256','model','covariates','first_test_origin']:
            if runs['15_days'][field] != runs['expanding'][field]:
                raise ValueError(f'Comparison changes {field} in addition to history')
        configs=[]
        for run in runs.values():
            c=dict(run['config']);c.pop('qwen_history_days');configs.append(c)
        if configs[0]!=configs[1]:
            raise ValueError('Only qwen_history_days may differ between configurations')
        short,long=scores['15_days'],scores['expanding']
        change={'dataset':dataset,'n_predictions':len(frames['15_days']),
                'model':runs['15_days']['model']['model']}
        for metric in ('rmse','mae','mape_percent','smape_percent'):
            change[f'{metric}_15_days']=short[metric]
            change[f'{metric}_expanding']=long[metric]
            change[f'{metric}_reduction_percent']=relative_reduction(short[metric],long[metric])
        change['r2_15_days']=short['r2'];change['r2_expanding']=long['r2']
        change['r2_difference']=long['r2']-short['r2']
        changes.append(change)
        for lead in range(1,8):
            scores_by_label={r['context_policy']:r for r in lead_metrics if r['dataset']==dataset and r['lead']==lead}
            short,long=scores_by_label['15_days'],scores_by_label['expanding']
            lead_changes.append({'dataset':dataset,'lead':lead,'rmse_15_days':short['rmse'],
                'rmse_expanding':long['rmse'],'rmse_reduction_percent':relative_reduction(short['rmse'],long['rmse'])})
    for filename,rows in [('metrics.csv',metrics),('improvement.csv',changes),
                          ('metrics_by_horizon.csv',lead_metrics),('improvement_by_horizon.csv',lead_changes)]:
        pd.DataFrame(rows).to_csv(output/filename,index=False)
    print(pd.DataFrame(changes)[['dataset','rmse_15_days','rmse_expanding','rmse_reduction_percent']].to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/qwen_context_comparison')
    compare_contexts(parser.parse_args().output)


if __name__=='__main__':
    main()
