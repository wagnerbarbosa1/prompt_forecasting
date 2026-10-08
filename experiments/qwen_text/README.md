# Qwen com CSVs textuais e prompts em linguagem natural

Entradas: `data/processed/calendar_text/*_calendar_text.csv`. Datas são
interpretadas apenas para ordenar e recortar a série; os prompts usam datas,
dias da semana, estações e indicadores em inglês. Demanda e meteorologia
permanecem numéricas. Cada covariável é descrita em uma sequência diária no
formato “From ... to ..., the ... of region ... was ... on each day,
respectively”, seguida de uma pergunta por dia previsto. Guaratuba recebe
contexto de sazonalidade turística de dezembro a março.

Configurações: `configs/qwen_text_{1_7b,4b}_{7,15}.json`.
Ollama local: `http://127.0.0.1:11435`. Modelos: `qwen3:1.7b` e
`qwen3:4b-instruct-2507-q4_K_M`. Inferência zero-shot, sem treinamento.

- Horizonte e passo: sete dias.
- Contexto do LLM: sete ou quinze dias, sempre anteriores à origem.
- Joinville: 54 janelas, 378 previsões por configuração.
- Guaratuba: 104 janelas, 728 previsões por configuração.
- Total: 632 prompts e 4.424 previsões.
- Demanda futura excluída dos prompts. Todas as configurações usam os mesmos alvos.
- Covariáveis meteorológicas futuras seguem a entrada fornecida pelo dataset;
  sua disponibilidade na origem não foi comprovada. As métricas são condicionais
  a essas entradas e não demonstram, por si só, desempenho operacional.

```bash
conda run -n promptcast python -m src.pipelines.qwen_text_experiment
conda run -n baseline python -m src.visualization.qwen_text_results
```

O primeiro comando gera prompts, executa as oito avaliações e escreve métricas;
o segundo valida as avaliações e gera gráficos com todos os sete leads.
`--prepare-only` gera somente prompts; `--summarize-only` recalcula métricas
apenas quando todas as avaliações estiverem completas.

As pastas de cada configuração contêm `prompts/*.txt`, `qwen/predictions.csv`,
`qwen/comparison.csv`, `qwen/run.json`, `qwen/status.json` e as respostas auditáveis.
`comparison_<cidade>.csv` contém somente modelo, contexto inicial, RMSE, R2,
MAE e MAPE (%). `metrics.csv` inclui sMAPE, contagens e identificação da cidade;
`metrics_by_horizon.csv` separa os sete leads. Os gráficos mostram séries
observadas/previstas com contextos de sete e quinze dias e a comparação de RMSE.
