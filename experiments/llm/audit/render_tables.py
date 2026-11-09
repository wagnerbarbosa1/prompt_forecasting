from pathlib import Path
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
metrics=pd.read_csv(ROOT/'metrics_verified.csv')
checks=pd.read_csv(ROOT/'sequence_checks.csv')
labels={'qwen_1_7b':'Qwen3 1.7B','qwen_4b':'Qwen3 4B','chronos':'Chronos2','timesfm':'TimesFM2.5'}
strategies={'zero':'Zero-shot','few_random':'Few-shot aleatório','few_similar':'Few-shot similar','expansivo':'Baseline'}
def number(value, decimals=2):
 if abs(value)>=1e7: return f'{value:.3e}'
 return f'{value:,.{decimals}f}'.replace(',','@').replace('.',',').replace('@','.')
for city in ['joinville','guaratuba']:
 c=canvas.Canvas(str(ROOT/f'{city}.pdf'),pagesize=(1920,1080))
 c.setFillColor(HexColor('#dbd8cf'));c.rect(0,0,1920,1080,fill=1,stroke=0)
 def text(x,y,value,size=22,bold=False,color='#28352d'):
  c.setFillColor(HexColor(color));c.setFont('DejaVuBold' if bold else 'DejaVu',size);c.drawString(x,y,value)
 text(60,1000,city.capitalize()+' | Comparação auditada',46,True)
 initial=365 if city=='joinville' else 730
 n=54 if city=='joinville' else 104
 text(60,951,f'Horizonte: 7 dias  •  Passo: 7 dias  •  Histórico inicial dos baselines: {initial} dias',26)
 text(60,911,f'LLMs: contexto fixo 7/15 dias  •  {n} janelas / {n*7} previsões por configuração  •  Mesmos alvos',24)
 xs=[60,290,405,645,895,1145,1390,1650]
 heads=['Modelo','Dias¹','Estratégia','RMSE (m³)','MAE (m³)','MAPE (%)','R²','sMAPE (%)']
 c.setFillColor(HexColor('#a3b28b'));c.rect(45,836,1830,49,fill=1,stroke=0)
 for x,h in zip(xs,heads):text(x,851,h,23,True)
 group=metrics[metrics.dataset==city]
 for i,(_,r) in enumerate(group.iterrows()):
  y=803-i*43
  if i%2==0:
   c.setFillColor(HexColor('#e8e5df'));c.rect(45,y-12,1830,42,fill=1,stroke=0)
  vals=[labels[r.model],str(r.context),strategies[r.strategy],number(r.rmse),number(r.mae),number(r.mape_percent),number(r.r2,3),number(r.smape_percent)]
  for x,v in zip(xs,vals):text(x,y,v,22,r.model=='chronos')
 count=checks[checks.dataset==city]
 matched=int((count.history_sequence_match|count.example_sequence_match).sum())
 text(60,174,f'Auditoria: {matched}/{len(count)} janelas Qwen repetem uma sequência de entrada ou resposta de exemplo.',22)
 text(60,139,'Coincidência exata é diagnóstico; não prova cópia. Valores extremos do Qwen 1.7B foram mantidos.',22)
 text(60,104,'DeepSeek-R1 1.5B e Llama 3.2 3B: lote pendente, retomado após resposta inválida do DeepSeek.',22)
 text(60,69,'¹ LLMs: dias de contexto fixo; baselines: contexto inicial expansivo. Clima disponível na origem, conforme informado.',20)
 text(60,33,'Resultados completos dos Qwen • Auditoria de 09/11/2026 • Few-shot: três exemplos anteriores ao primeiro teste',19)
 c.save()
