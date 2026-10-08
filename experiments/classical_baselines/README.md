# Baselines clássicos: 355 / 30 / 30

Regressão linear e KNN usam atrasos da demanda; ARIMA usa a série de demanda.
Os três são univariados: não recebem clima futuro ou covariáveis observadas no teste.
As bibliotecas são as já instaladas no ambiente `baseline`, sem instalações.

## Divisão temporal

| Fold | Treino para seleção | Validação | Teste |
|---|---|---|---|
| 1 | índices [0,355) | [355,385) | [385,415) |
| 2 | [0,385) | [385,415) | [415,445) |
| 3 | [0,415) | [415,445) | [445,475) |

A cada origem, candidatos são ajustados somente no treino. A validação é prevista
como um bloco recursivo de 30 dias, sem valores reais dentro desse horizonte.
O menor RMSE de validação seleciona os parâmetros. Depois, o modelo é reajustado
com treino + validação, ambos conhecidos antes do teste: 385 dias no primeiro
fold, 415 no segundo. StandardScaler e a normalização do ARIMA são ajustados
novamente somente nesse prefixo permitido. O teste é previsto em bloco, sem
atualizar o modelo ou os atrasos com seus valores reais durante os 30 dias.

Terminando o bloco, os 30 valores reais passam a integrar o passado da origem
seguinte. A faixa de validação se desloca e o prefixo de treino cresce 30 dias.
São usados somente folds completos: 12 em Joinville e 35 em Guaratuba. A cauda
incompleta de Guaratuba (26 dias) fica registrada em `run.json`, sem avaliação.
Dias ausentes no histórico recebem apenas preenchimento com passado; alvos
não observados não são usados para pontuar validação ou teste.

## Parâmetros e resultados

A grade explícita está em `configs/classical_baselines.json`: atrasos 7/15/30;
KNN com 3/5/10 vizinhos e pesos uniformes ou por distância; oito ordens ARIMA.
Falhas de candidatos, incluindo não convergência, ficam registradas por fold;
não são substituídas por previsões inventadas. Os parâmetros não são escolhidos
com métricas de teste. Nenhuma previsão negativa ou extrema é removida.

Cada modelo/cidade produz `predictions.csv`, `comparison.csv`, `status.json`,
`run.json` e `folds/*.json` com seleção, métricas de validação e limites temporais.
Os CSVs de comparação têm somente modelo, contexto inicial (355), RMSE, R2, MAE
e MAPE (%). `fit_days` nas previsões inclui a validação usada no reajuste final;
`training_days` identifica o treino usado na seleção. As métricas agregadas
usam todos os 30 leads e estão em `metrics.csv`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n baseline python -m src.pipelines.classical_baselines
```

O protocolo 355/30/30 se aplica somente a estes três baselines. Qwen, Chronos e
TimesFM mantêm seus protocolos anteriores; suas métricas de horizonte 7 não
são diretamente comparáveis às métricas de horizonte 30 deste experimento.
