# Previsões com Qwen

O pipeline `src/pipelines/forecasting_pipeline.py` executa o Qwen pelo Ollama
local com `qwen_history_days` igual a 7 ou 15. A origem e os sete dias previstos
são os mesmos de Chronos e TimesFM. Apenas esses dois modelos recebem todo o
histórico anterior à origem.

`model_inputs` recorta o histórico e exclui `water_produced` das covariáveis
futuras. Exemplos few-shot vêm do prefixo congelado anterior ao primeiro teste.
A verdade observada é consultada depois da inferência para calcular métricas.
A disponibilidade antecipada das covariáveis meteorológicas fornecidas pelos
CSVs continua sendo uma hipótese não comprovada; a conversão do calendário não
altera essa condição.

`generate_prompts.py` descreve demanda e covariáveis em inglês, em sequências
diárias. `calendar_text.py` escreve datas e calendário por extenso. Guaratuba
recebe o contexto de sazonalidade turística de dezembro a março.

O modelo retorna JSON com sete números não negativos. O pipeline salva
prompts, respostas, procedência, `predictions.csv` e `status.json` por cidade.
`compare_models` compara execuções completas de Qwen, Chronos e TimesFM nos
mesmos alvos, usando todos os sete dias previstos.

Os resultados históricos do Qwen3 4B com 15 dias estão em
`experiments/qwen_15_days`. Seus prompts são anteriores à conversão para
linguagem natural; os arquivos de procedência preservam a execução original.


## Comparação histórica dos prompts e baselines clássicos

A comparação Qwen preserva as previsões numéricas anteriores e as narrativas.
Verifica mesmos alvos, origens, leads e checkpoint; a comparação pareada usa
contexto de 15 dias. O 4B reduziu RMSE em 5,16% em Joinville e 7,86% em Guaratuba;
o 1.7B teve aumento de 522,33% e 400,04%. Além da linguagem, mudaram conteúdo e
`num_ctx` (capacidade de contexto), portanto não se trata de ganho causal isolado.
Resultados: `experiments/prompt_comparison/`.

Somente regressão linear, ARIMA e KNN usam treino inicial 355 + validação 30 +
teste 30, com passo 30 e treino expansivo. No primeiro fold: treino [0,355),
validação [355,385), teste [385,415). Parâmetros são selecionados pelo RMSE de
validação; há reajuste com treino+validação (385 dias), sempre anterior ao teste.
No fold seguinte, os 30 valores reais já observados ocupam a nova validação;
o treino cresce 30 dias. Durante cada horizonte só previsões alimentam a recursão.

Os três modelos usam somente demanda passada. Regressão e KNN usam atrasos
7/15/30 com escala ajustada no histórico permitido; ARIMA testa oito ordens
não sazonais e normaliza com estatísticas desse mesmo prefixo. Foram concluídos
141 folds e 4230 previsões, em seis execuções; resultados e auditoria em
`experiments/classical_baselines/`. Os horizontes 30 e 7 não devem ser comparados
diretamente como se usassem o mesmo protocolo.

Execute `conda run -n baseline python -m src.pipelines.classical_baselines`
para os baselines e `conda run -n baseline python -m src.visualization.comparison_results`
para os gráficos a partir dos resultados salvos. O PDF completo inclui os novos
módulos comentados linha a linha e explica as grades, seleção, reajuste e auditoria.
