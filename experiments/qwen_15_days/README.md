# Qwen3 4B com contexto de 15 dias

Resultados históricos preservados com horizonte e passo de sete dias em
Joinville e Guaratuba. Cada pasta de cidade contém previsões, prompts,
respostas e procedência. `metrics.csv` contém métricas da execução de 15 dias.
Os prompts salvos usam o formato anterior; novas execuções usam linguagem
natural e devem ser gravadas em uma pasta diferente para preservar este registro.

Configuração: `configs/qwen_context_15_days.json`. Ollama local na porta 11435;
o servidor pode ser iniciado com `bash experiments/start_ollama_context_server.sh`.

```bash
conda run -n promptcast python -m src.pipelines.forecasting_pipeline \
  --model qwen --config configs/qwen_context_15_days.json \
  --output experiments/qwen_15_days_natural_language
```
