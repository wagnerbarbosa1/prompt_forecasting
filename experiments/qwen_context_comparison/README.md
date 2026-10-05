# Qwen: comparação controlada de contexto

O Qwen3 1.7B original possui limite local de 40.960 tokens e não comporta as janelas completas com covariáveis: até 52.891 tokens em Joinville e 116.207 em Guaratuba no prompt anterior. O usuário autorizou uma variante com contexto maior.

## Modelo e configurações

As duas configurações usam **qwen3:4b-instruct-2507-q4_K_M**, com os mesmos pesos, temperatura 0, seed 42, num_predict 256, num_ctx 131072, Flash Attention e cache KV q4_0. O modelo declara limite de 262.144 tokens. O servidor local separado usa a porta 11435; nenhum ambiente conda foi modificado.

- `15_days`: 15 dias históricos em cada origem.
- `expanding`: 365→736 dias em Joinville e 730→1451 em Guaratuba, desde o início da série, exatamente como Chronos e TimesFM.
- Mesmas covariáveis, sete dias de horizonte, passo de sete, datas e alvos nas duas configurações.
- Mesmo prompt tabular nos dois braços: uma linha por dia com demanda e todas as covariáveis. A história aparece antes dos metadados variáveis da origem para permitir reaproveitamento de prefixo no Ollama. Nenhum valor ou dia é removido. Precisão textual: duas casas decimais na demanda e seis algarismos significativos nas covariáveis, como anteriormente.
- A demanda futura não é incluída. Covariáveis futuras seguem a hipótese de disponibilidade já documentada no benchmark.

Os resultados anteriores de Qwen3 1.7B permanecem em `experiments/benchmark`. Compará-los diretamente com o 4B não isolaria o efeito do contexto; por isso executamos novamente os 15 dias no 4B, com o mesmo formato usado na execução expansiva.

## Saídas finais

- `metrics.csv`: métricas agregadas dos dias 1 a 7, por cidade e contexto.
- `improvement.csv`: erros das duas configurações e redução percentual. Valor positivo indica melhoria; negativo indica piora.
- `metrics_by_horizon.csv`: métricas separadas para cada dia previsto.
- `improvement_by_horizon.csv`: mudança de RMSE por antecedência.
- `15_days/` e `expanding/`: prompts, previsões por janela, respostas brutas e procedência.
- `context_preflight.csv`: diagnóstico do limite do modelo antigo, feito com seu prompt original.

Redução de RMSE (%) = 100 × (RMSE_15dias − RMSE_expansivo) / RMSE_15dias.

## Reproduzir

Servidor dedicado, sem mudar a configuração do servidor anterior:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=q4_0 OLLAMA_NUM_PARALLEL=1 LLAMA_ARG_CACHE_RAM=0 ollama serve
```

Em outro terminal, na raiz do projeto:

```bash
bash experiments/run_qwen_context_comparison.sh
```

O comparador exige execuções completas, sete leads por janela, os mesmos pesos/configuração/covariáveis e exatamente os mesmos alvos. Só `qwen_history_days` pode diferir. Não assume que contexto maior necessariamente melhora o resultado.

O cache adicional de snapshots de prompts está desabilitado (`LLAMA_ARG_CACHE_RAM=0`) para evitar swap. O cache KV da sequência ativa continua disponível para o reaproveitamento do prefixo histórico.
