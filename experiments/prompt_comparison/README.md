# Comparação histórica dos prompts Qwen

Preserva as previsões numéricas/tabulares anteriores e as previsões narrativas.
Os CSVs têm alvos e datas iguais, e o digest de cada checkpoint é conferido.
A comparação pareada usa contexto de 15 dias; os resultados narrativos de sete
dias permanecem visíveis, sem inventar uma execução numérica equivalente.

A diferença é descritiva: além da estrutura do prompt/calendário, o limite de
tokens mudou (1.7B: 8192 → 16384; 4B: 131072 → 16384). O contexto turístico de
Guaratuba também faz parte do novo prompt. Não atribua a variação exclusivamente
à redação em linguagem natural. Covariáveis futuras continuam condicionais à
entrada fornecida pelos datasets e à hipótese de disponibilidade já documentada.

`prompt_change.csv` usa redução de RMSE = 100 × (anterior − natural) / anterior.
Valor positivo indica erro menor; negativo, erro maior. Os gráficos têm dois
painéis, com escalas verticais automáticas, e preservam todos os pontos/extremos.
`comparison_<cidade>.csv` tem as seis colunas simples solicitadas; `metrics.csv`
inclui tipo de prompt e contexto para evitar misturar configurações.

```bash
conda run -n baseline python -m src.visualization.comparison_results
```

Esse comando usa resultados existentes, verifica sua correspondência e gera
os quatro gráficos comparativos de Qwen e os dois gráficos dos baselines
clássicos, que possuem protocolo próprio de 355/30/30.
