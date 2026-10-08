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
