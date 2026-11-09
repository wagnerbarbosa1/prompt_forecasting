"""Publish complete, aligned metrics and explicitly mark incomplete DeepSeek runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from src.evaluation.metrics import deterministic_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('DV','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DVB','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
review=pd.read_csv(ROOT/'experiments/copy_review/copy_review.csv')
records=[]
labels={'qwen_1_7b':'Qwen3 1.7B','qwen_4b':'Qwen3 4B','deepseek_1_5b':'DeepSeek-R1 1.5B','llama_3b':'Llama 3.2 3B',
        'chronos':'Chronos2','timesfm':'TimesFM2.5'}
strategies={'zero':'Zero-shot','few_random':'Aleatório','few_similar':'Similar'}
for city in ['joinville','guaratuba']:
    reference=None
    for name in ['chronos','timesfm']:
        for days in [None,7,15]:
            folder=ROOT/'experiments/benchmark'/city/name if days is None else ROOT/'experiments/sliding'/str(days)/city/name
            status=json.loads((folder/'status.json').read_text())
            assert not status['failures'] and status['successful_windows']==status['expected_windows']
            df=pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
            keys=df[['origin','timestamp','lead','y_true']]
            if reference is not None:pd.testing.assert_frame_equal(reference,keys)
            reference=keys
            records.append(dict(city=city,model=labels[name],days=days or (365 if city=='joinville' else 730),
                                strategy='Expansiva' if days is None else 'Deslizante',valid=status['successful_windows'],
                                expected=status['expected_windows'],state='Completo',**deterministic_metrics(df.y_true,df.y_pred)))
    for slug in ['qwen_1_7b','qwen_4b','llama_3b','deepseek_1_5b']:
        for days in [7,15]:
            for strategy in ['zero','few_random','few_similar']:
                relative=Path(slug)/str(days)/strategy/city/'llm'
                original=ROOT/'experiments/llm'/relative
                status=json.loads((original/'status.json').read_text())
                complete=not status['failures'] and status['successful_windows']==status['expected_windows']
                row=dict(city=city,model=labels[slug],days=days,strategy=strategies[strategy],
                         valid=status['successful_windows'],expected=status['expected_windows'],state='Revisado' if complete else 'Falhas')
                if complete:
                    folder=ROOT/'experiments/copy_review'/relative
                    df=pd.read_csv(folder/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
                    pd.testing.assert_frame_equal(reference,df[['origin','timestamp','lead','y_true']])
                    old=pd.read_csv(original/'predictions.csv').sort_values(['origin','lead']).reset_index(drop=True)
                    modified=json.loads((folder/'review.json').read_text())['changed_windows']
                    changed={r['origin'] for r in modified if r['status']=='ok'}
                    np.testing.assert_allclose(df.loc[~df.origin.isin(changed),'y_pred'],old.loc[~old.origin.isin(changed),'y_pred'],rtol=1e-14,atol=1e-10)
                    row.update(deterministic_metrics(df.y_true,df.y_pred))
                    row['original_rmse']=deterministic_metrics(old.y_true,old.y_pred)['rmse']
                records.append(row)
all_rows=pd.DataFrame(records)
all_rows.to_csv(OUT/'metrics_with_status.csv',index=False)

def fmt(value,decimals=2):
    if pd.isna(value):return '—'
    if abs(value)>=1e7:return f'{value:.2e}'
    return f'{value:,.{decimals}f}'.replace(',','@').replace('.',',').replace('@','.')

for city in ['joinville','guaratuba']:
    rows=all_rows[all_rows.city==city]
    c=canvas.Canvas(str(OUT/f'{city}.pdf'),pagesize=(1920,1080))
    c.setFillColor(HexColor('#dbd8cf'));c.rect(0,0,1920,1080,fill=1,stroke=0)
    def text(x,y,value,size=18,bold=False,color='#28352d'):
        c.setFillColor(HexColor(color));c.setFont('DVB' if bold else 'DV',size);c.drawString(x,y,str(value))
    n=54 if city=='joinville' else 104
    initial=365 if city=='joinville' else 730
    text(48,1028,city.capitalize()+' | Resultados após revisão',39,True)
    text(48,986,f'Horizonte: 7 dias  •  Passo: 7 dias  •  Origem inicial: {initial} dias  •  {n} janelas / {n*7} pontos por execução completa',22)
    text(48,951,'Covariáveis mantidas • LLMs e sliding: 7/15 dias • Expanding preservado • Métricas em todos os sete dias',21)
    xs=[48,268,355,532,720,960,1200,1430,1670]
    c.setFillColor(HexColor('#a3b28b'));c.rect(36,902,1848,35,fill=1,stroke=0)
    for x,h in zip(xs,['Modelo','Dias¹','Estratégia','Estado²','RMSE (m³)','MAE (m³)','MAPE (%)','R²','sMAPE (%)']):text(x,914,h,20,True)
    for i,(_,r) in enumerate(rows.iterrows()):
        y=880-i*24
        if i%2==0:
            c.setFillColor(HexColor('#e8e5df'));c.rect(36,y-6,1848,24,fill=1,stroke=0)
        status=r.state if r.state!='Falhas' else f'{r.valid}/{r.expected} válidas'
        vals=[r.model,r.days,r.strategy,status,fmt(r.get('rmse')),fmt(r.get('mae')),fmt(r.get('mape_percent')),fmt(r.get('r2'),3),fmt(r.get('smape_percent'))]
        for x,v in zip(xs,vals):text(x,y,v,18,False,'#8c372f' if r.state=='Falhas' else '#28352d')
    subset=review[review.run.str.contains('/'+city+'/')]
    repeated=int(subset.still_matches.sum())
    text(48,143,f'Revisão: {len(subset)} janelas refeitas; {repeated} ainda repetem ≥3 valores consecutivos do prompt. Coincidência não prova cópia.',19)
    text(48,111,'² Revisado: inclui novas respostas válidas, mesmo com repetição persistente. Traço: execução incompleta, sem métrica parcial.',18)
    text(48,79,'¹ LLMs/sliding: histórico fixo; expanding: histórico inicial crescente. Clima futuro disponível na origem, conforme informado.',18)
    text(48,47,'DeepSeek: 39 janelas sem resposta válida no lote original. Valores extremos foram mantidos; nenhuma previsão foi corrigida manualmente.',18)
    text(48,17,'Revisão concluída • 09/11/2026 • CSV completo e indicadores de falha em experiments/final_results/metrics_with_status.csv',15)
    c.save()
print('Rows:',len(all_rows),'complete:',int(all_rows.state.ne('Falhas').sum()),'incomplete:',int(all_rows.state.eq('Falhas').sum()))
