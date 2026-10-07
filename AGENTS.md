#### Most importante things
De maneira nenhuma pode haver data leakage para esses dados, isto é, o modelo conseguir ver dados com indices de teste no seu treinamento.

#### Environment
todos os ambientes conda que você pode usar, mas para modificá-los você precisará me consultar antes:
chronos, baseline, moirai, promptcast, timesfm

#### Code style
Priorize clean code, lógica clara e deixe explícito suas alterações, motivos e lógicas.
Codigos que não serão utilizados, devido a outras alterações e não serão uteis para o pipeline devem ser excluidos.
Siga a estrutura de design pattern proposta:
project/
│
├── src/
│   │
│   ├── data/
│   │   ├── ingestion/
│   │   │   ├── api.py
│   │   │   ├── database.py
│   │   │   └── files.py
│   │   │
│   │   ├── validation/
│   │   │   └── schema.py
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── cleaning.py
│   │   │   ├── resampling.py
│   │   │   └── missing_values.py
|    |   |    |___ generate_prompts.py 
│   │   │
│   │   └── features/
│   │       ├── temporal.py
│   │       └── covariates.py
│   │
│   ├── models/
│   │   ├── chronos.py
│   │   ├── timesfm.py
│   │   └── moirai.py
|   |    |__ promptcast.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── cross_validation.py
│   │
│   └── pipelines/
│       ├── data_pipeline.py
│       └── forecasting_pipeline.py
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── experiments/
│
├── notebooks/
│
├── tests/
│
└── pyproject.toml

#### Train-Tests Models
O ollama precisará necessariamente rodar localmente.
os modelos precisam necessariamente estar nessa configuração dependendo de cada dataset e validação.
Janela expansiva:
Para "joinville_processed_1day" 7 dias de horizonte, 7 dias de passo, 365 dias de contexto, caso necessário validação será sempre na proporção de 20% da janela treinamento.
Para "guaratuba_processed_1daysin" 7 dias de horizonte, 7 dias de passo, com 730 dias de contexto, caso necessário validação será sempre na proporção de 20% da janela de treinamento treinamento.
Para o promptcast será necessário avalia-lo nos mesmos dias que os outros modelos, os modelos do ollama deverão ser executados com contexto de 7 dias e 15 dias.
Todo teste deverá retornar um csv de previsões de cada modelo e outro csv de comparações simples, com apenas o modelo, contexto inicial, e as métricas RMSE, R2, MAE, MAPE.
Toda e qualquer modificação, deve vir seguida de atualização no pdf guia_qwen_forecast, explicando linha por linha o código, suas relações com o projeto e por qual motivo houve a alteração.
