"""Retain historical numeric Qwen outputs beside narrative prompts and new baselines."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from src.evaluation.metrics import deterministic_metrics
from src.pipelines.forecasting_pipeline import ROOT

OUTPUT = ROOT/'experiments/prompt_comparison'
LABELS = {'linear_regression':'Regressão linear', 'knn':'KNN', 'arima':'ARIMA'}


def prompt_comparison():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, changes = [], []
    for city in ('joinville','guaratuba'):
        for slug, label, old_root in [('1_7b','Qwen3 1.7B','experiments/benchmark'),
                                     ('4b','Qwen3 4B','experiments/qwen_15_days')]:
            old_folder = ROOT/old_root/city/'qwen'
            old_run = json.loads((old_folder/'run.json').read_text())
            old_status = json.loads((old_folder/'status.json').read_text())
            if old_status['failures'] or old_status['successful_windows']!=old_status['expected_windows']:
                raise ValueError('Incomplete historical Qwen evaluation')
            old = pd.read_csv(old_folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
            old_score = deterministic_metrics(old.y_true,old.y_pred)
            rows.append(dict(dataset=city,model=label,slug=slug,format='numérico/tabular anterior',context=15,
                             windows=old.origin.nunique(),predictions=len(old),**old_score))
            for days in (7,15):
                folder = ROOT/'experiments/qwen_text'/f'{slug}_{days}'/city/'qwen'
                new = pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
                new_run = json.loads((folder/'run.json').read_text())
                status = json.loads((folder/'status.json').read_text())
                if status['failures'] or status['successful_windows']!=status['expected_windows']:
                    raise ValueError('Incomplete narrative Qwen evaluation')
                keys = ['origin','timestamp','lead','y_true']
                pd.testing.assert_frame_equal(old[keys],new[keys])
                if old_run['model']['digest'] != new_run['model']['digest']:
                    raise ValueError('Historical and narrative checkpoints differ')
                scores = deterministic_metrics(new.y_true,new.y_pred)
                rows.append(dict(dataset=city,model=label,slug=slug,format='linguagem natural',context=days,
                                 windows=new.origin.nunique(),predictions=len(new),**scores))
                if days==15:
                    changes.append(dict(dataset=city,model=label,context=15,
                        old_num_ctx=old_run['config']['qwen']['num_ctx'],new_num_ctx=new_run['config']['qwen']['num_ctx'],
                        old_rmse=old_score['rmse'],new_rmse=scores['rmse'],
                        rmse_reduction_percent=100*(old_score['rmse']-scores['rmse'])/old_score['rmse'],
                        interpretation='Historical comparison; token budgets and prompt contents also differ'))
                    fig,axes = plt.subplots(2,1,figsize=(16,8),sharex=True,constrained_layout=True)
                    for ax,frame,title,score in [(axes[0],old,'Calendário numérico / prompt anterior',old_score),
                                                (axes[1],new,'Calendário textual / linguagem natural',scores)]:
                        dates=pd.to_datetime(frame.timestamp)
                        ax.plot(dates,frame.y_true,color='#777777',lw=1.25,label='Real')
                        ax.plot(dates,frame.y_pred,color='#2478b5',lw=1.1,label='Previsto')
                        ax.set_title(f'{title} | RMSE {score["rmse"]:,.2f} | R² {score["r2"]:.3f}',loc='left')
                        ax.set_ylabel('Produção (m³)');ax.grid(alpha=.2);ax.legend(ncol=2,loc='upper right')
                    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
                    fig.suptitle(f'{city.title()} — {label} | Comparação com contexto de 15 dias',fontsize=19)
                    fig.savefig(OUTPUT/f'{city}_{slug}_comparison.png',dpi=150)
                    plt.close(fig)
        selected=[row for row in rows if row['dataset']==city]
        pd.DataFrame([{'modelo':f'{r["model"]} — {r["format"]}','contexto_inicial':r['context'],
                       'RMSE':r['rmse'],'R2':r['r2'],'MAE':r['mae'],'MAPE':r['mape_percent']}
                      for r in selected]).to_csv(OUTPUT/f'comparison_{city}.csv',index=False)
    pd.DataFrame(rows).to_csv(OUTPUT/'metrics.csv',index=False)
    pd.DataFrame(changes).to_csv(OUTPUT/'prompt_change.csv',index=False)


def baseline_plots():
    root=ROOT/'experiments/classical_baselines'
    metrics=pd.read_csv(root/'metrics.csv')
    for city in ('joinville','guaratuba'):
        fig,axes=plt.subplots(3,1,figsize=(16,9),sharex=True,constrained_layout=True)
        reference=None
        for ax,name in zip(axes,LABELS):
            frame=pd.read_csv(root/city/name/'predictions.csv')
            keys=frame[['origin','timestamp','lead','y_true']]
            if reference is not None:pd.testing.assert_frame_equal(reference,keys)
            reference=keys
            m=metrics.query('dataset == @city and model == @name').iloc[0]
            dates=pd.to_datetime(frame.timestamp)
            ax.plot(dates,frame.y_true,color='#777777',lw=1.1,label='Real')
            ax.plot(dates,frame.y_pred,color='#2478b5',lw=1.1,label='Previsto')
            ax.set_title(f'{LABELS[name]} | RMSE {m.rmse:,.2f} | R² {m.r2:.3f}',loc='left')
            ax.set_ylabel('Produção (m³)');ax.grid(alpha=.2);ax.legend(ncol=2,loc='upper right')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        fig.suptitle(f'{city.title()} — Baselines | Treino inicial 355 + validação 30 + teste 30 dias',fontsize=18)
        fig.savefig(OUTPUT/f'{city}_baselines.png',dpi=150)
        plt.close(fig)


if __name__=='__main__':
    prompt_comparison()
    baseline_plots()
