# PromptCast Forecast

Comparação de Qwen 3 1.7B via Ollama local, Chronos-2 e TimesFM 2.5, preservando as covariáveis e o protocolo de janela expansiva definido no `AGENTS.md`.

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

Chronos recebe essas colunas na história e em `future_df`. TimesFM recebe vetores com história + sete datas futuras, separados entre covariáveis numéricas e categóricas. Qwen recebe os mesmos nomes e valores correspondentes em duas tabelas CSV dentro do prompt: histórico e futuro. O texto usa seis algarismos significativos para covariáveis e duas casas decimais para demanda.

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
