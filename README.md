# PromptCast Forecast

Comparação de Qwen 3 1.7B via Ollama local, Chronos-2 e TimesFM 2.5, preservando as covariáveis e as datas de avaliação definidas no `AGENTS.md`. Qwen usa contexto fixo de 7 ou 15 dias; Chronos e TimesFM usam histórico expansivo.

## Protocolo atual

| Dataset | Histórico inicial de Chronos/TimesFM | Evolução do histórico | Histórico Qwen | Horizonte / passo |
|---|---:|---|---:|---|
| Joinville | 365 dias | 365, 372, 379… até 736 | 15 dias | 7 / 7 dias |
| Guaratuba | 730 dias | 730, 737, 744… até 1451 | 15 dias | 7 / 7 dias |

Chronos e TimesFM recebem todas as observações desde o início da série até a véspera da previsão. Chronos recebe `context_length=len(history)`; TimesFM recompila a capacidade de contexto em cada janela e usa `forecast_with_covariates`, com `xreg_mode="xreg + timesfm"` e `return_backcast=True`, como no código original. O Qwen usa a exceção de até 15 dias expressamente permitida no `AGENTS.md`; as datas dos sete alvos são idênticas às das baselines.

**As métricas principais usam TODOS os dias previstos, do 1º ao 7º.** Não se limitam aos dois primeiros dias dos scripts antigos. São 54 janelas/378 previsões de Joinville e 104 janelas/728 previsões de Guaratuba para cada modelo. O RMSE agregado é a raiz da média de todos os erros quadráticos, não a média dos RMSEs das janelas. O CSV adicional discrimina cada uma das sete antecedências.

## Covariáveis preservadas

A configuração explícita de `configs/benchmark.json` é compartilhada pelos três modelos:

- **Joinville:** `precipitation_median`, `temperature_2m_previsao_mean`, `shortwave_radiation_previsao_std`, `day`, `dayofweek_sin`, `dayofweek_cos`, `season_sin`, `season_cos` e `is_holiday`.
- **Guaratuba:** `relative_humidity_mean`, `temperature_mean`, `radiation_mean`, `precipitation_mean`, `day`, `dayofweek_sin`, `dayofweek_cos`, `season_sin`, `season_cos`, `is_holiday_ctba_gtba_jve`, `is_weekend`, `is_carnival` e `is_school_recess_pr`.

Chronos recebe essas colunas na história e em `future_df`. TimesFM recebe vetores com história + sete datas futuras, separados entre covariáveis numéricas e categóricas. Qwen recebe frases em inglês no formato “From … to …, the … of region … was … on each day, respectively”, seguidas de uma pergunta por data de previsão. Datas e calendário são escritos em inglês; pares seno/cosseno viram dias da semana e estações, e indicadores viram descrições. Guaratuba recebe o contexto de cidade turística com alta sazonalidade de dezembro a março. O texto usa seis algarismos significativos para covariáveis e duas casas decimais para demanda.

`water_produced` futuro nunca entra nos inputs dos modelos. As covariáveis futuras são as fornecidas pelos CSVs, preservando o comportamento solicitado. A disponibilidade antecipada das covariáveis meteorológicas é uma hipótese de entrada não comprovada pelos CSVs: se forem observações indisponíveis na origem, a avaliação é condicional a essas informações, não uma simulação operacional sem informação futura. Isso não é resolvido removendo colunas silenciosamente.

A grade diária já usada na reorganização foi mantida: o dia ausente 29/02/2016 de Guaratuba é preenchido com valores anteriores, sem alterar o CSV fonte. Por isso 730 dias de calendário terminam em 30/12/2017, e a primeira origem é 31/12/2017. Esse tratamento pode mudar datas em relação à contagem das linhas do script antigo. Alvos preenchidos são excluídos. Pesos dos modelos permanecem congelados; a regressão do TimesFM usa somente alvos anteriores à origem atual. Para um futuro ajuste/validação, a proporção é 20% do treino.

## Organização

```text
configs/benchmark.json              protocolo e lista explícita de covariáveis
src/data/ingestion/                 leitura e gravação atômica
src/data/validation/                validação que preserva as colunas
src/data/preprocessing/             grade diária, preenchimento e prompts
src/data/features/covariates.py     inputs comuns, separação história/futuro
src/models/                        Qwen, Chronos e TimesFM
src/evaluation/                    janela expansiva e métricas
src/training/                      partição de treino/validação
src/pipelines/                     dados, execução e comparação
experiments/benchmark/              resultados do protocolo atual
experiments/reports/                manual PDF/HTML e validação
experiments/previous_demand_only/   execução anterior substituída, não usar como principal
notebooks/                         análise dos dados e métricas
tests/                             testes de contexto, covariáveis, alvos e métricas
```

## Executar sem modificar os ambientes

```bash
# Se o servidor ainda não estiver ativo, em outro terminal:
ollama serve
# Na raiz do projeto:
bash experiments/run_benchmark.sh
```

O script usa os ambientes conda existentes; ajuste `CONDA_ENVS` se eles estiverem em outro caminho. Executa os modelos sequencialmente para liberar memória entre eles. `XLA_PYTHON_CLIENT_PREALLOCATE=false` afeta somente o processo TimesFM; não instala nem modifica pacotes.

Execução individual:

```bash
conda run -n promptcast python -m src.pipelines.forecasting_pipeline --model qwen
conda run -n chronos python -m src.pipelines.forecasting_pipeline --model chronos
conda run -n timesfm python -m src.pipelines.forecasting_pipeline --model timesfm
conda run -n promptcast python -m src.pipelines.compare_models
conda run -n promptcast python -m unittest discover -s tests -v
```

## Saídas

- `experiments/benchmark/metrics_comparison.csv`: métricas agregadas dos dias **1 a 7**, uma linha por cidade/modelo.
- `experiments/benchmark/metrics_by_horizon.csv`: métricas separadas para **cada dia, de 1 a 7**; 42 linhas.
- `<cidade>/<modelo>/predictions.csv`: uma linha por previsão, com `lead`, `y_true`, `y_pred` e tamanho real de `context`.
- `<cidade>/<modelo>/responses/*.json`: respostas e auditoria, incluindo número de observações fornecidas.
- `<cidade>/<modelo>/run.json`: configuração, covariáveis, hashes de dados/código e identificação do modelo.
- `<cidade>/prompts/*.txt`: texto enviado ao Ollama local.
- `experiments/reports/guia_qwen_forecast.pdf`: explicação didática de prompts, covariáveis, funções, argumentos e Ollama.

O comparador rejeita falhas, execução limitada, alvos/datas diferentes e histórico que não cresce corretamente. Resultados anteriores sem covariáveis foram separados em `experiments/previous_demand_only`; eles não são reutilizados no protocolo atual. Cache só é reaproveitado quando o hash da execução e da requisição coincide.

### Exportar calendário em inglês

```bash
conda run -n baseline python -m src.data.preprocessing.calendar_text
```

Gera cópias em `data/processed/calendar_text/*_calendar_text.csv`, preservando
linhas, ordem, demanda e meteorologia. Exemplo: `2016-01-01` vira
“two thousand and sixteen, day one of January”. Os pares `dayofweek_sin/cos`
e `season_sin/cos` são substituídos por `dayofweek` e `season` textuais.
Essas cópias são para leitura: o pipeline continua carregando os CSVs numéricos
originais e aplica a mesma conversão apenas aos inputs de cada prompt, depois
do recorte temporal. O exportador não treina modelos nem altera os splits.

O benchmark usa `num_ctx=16384` para acomodar o texto por extenso com 7/15 dias.
O LLM aceita somente `qwen_history_days=7` ou `15`; valores ausentes, nulos ou diferentes são rejeitados antes da inferência. Chronos e TimesFM mantêm o histórico expansivo.

### Qwen 1.7B e 4B com datasets textuais

As quatro configurações `configs/qwen_text_*.json` usam diretamente os CSVs de
`data/processed/calendar_text`, com contextos fixos de 7/15 dias e Ollama local
na porta 11435. Execute `python -m src.pipelines.qwen_text_experiment` no ambiente
promptcast; os resultados completos e comparações ficam em `experiments/qwen_text`.
Para os gráficos, execute `python -m src.visualization.qwen_text_results` no
ambiente baseline. Use `--prepare-only` no primeiro comando para gerar somente
os prompts.

O guia completo atualizado está em `guia_qwen_forecast.pdf` e
`experiments/reports/guia_qwen_forecast.pdf`; o HTML pesquisável fica ao lado da
segunda cópia. Reconstrua com
`python3 experiments/reports/guide_source/build_guide.py` (ReportLab já instalado).

### Baselines clássicos: 355/30/30

Regressão linear, ARIMA e KNN seguem `configs/classical_baselines.json`:
355 dias iniciais de treino, 30 de validação e 30 de teste, com avanço de 30 dias.
A seleção usa somente treino/validação e o reajuste final usa o prefixo anterior
ao teste. Previsões de cada bloco são recursivas, sem inserir os alvos reais
durante o horizonte. São modelos univariados, sem meteorologia futura.

Execute `python -m src.pipelines.classical_baselines` no ambiente `baseline`.
Os resultados e a auditoria estão em `experiments/classical_baselines`.
Esse protocolo é exclusivo desses três modelos; Qwen, Chronos e TimesFM mantêm
os anteriores. Veja os detalhes dos recortes e do reajuste no README da execução.

A comparação histórica entre Qwen numérico/tabular e narrativo está em
`experiments/prompt_comparison`. Os pares comparáveis têm contexto de 15 dias;
limites de tokens também mudaram, portanto a variação de erro não isola apenas
a redação do prompt. `python -m src.visualization.comparison_results` recria
os gráficos a partir dos CSVs verificados, sem executar novas inferências Qwen.
