# Revisão de sequências repetidas e comparação de contextos

`identified.json` lista as janelas encontradas na verificação inicial. O critério
é igualdade, com tolerância absoluta de 0,000001, de pelo menos três valores
consecutivos com alguma sequência do prompt. São examinados demanda histórica,
covariáveis numéricas e respostas dos exemplos few-shot. Coincidência é um
indicador de repetição, não prova da causa interna da resposta.

`reforecast.py` processa apenas execuções completas sem falhas. Acrescenta uma
instrução de revisão, usa seed original + 1 e solicita uma nova previsão. Os
valores reais futuros não entram na detecção, no pedido ou na escolha da resposta.
Uma nova resposta válida substitui a anterior na cópia dos resultados mesmo
quando seu erro aumenta ou a repetição persiste. Falhas conservam o resultado
original e ficam explícitas em `review.json`; as tentativas ficam em `responses`.
Os resultados de `experiments/llm` não são sobrescritos.

`run_review_and_sliding.sh` pausa temporariamente o lote pendente, faz a revisão,
libera a GPU do Ollama e executa os baselines nos respectivos ambientes conda.
Ao sair, retoma o lote anterior, inclusive se o controlador encerrar com erro.

`experiments/sliding/run.py` usa o SlidingWindow original com um deslocamento de
índice: a primeira previsão continua na posição 365 de Joinville e 730 de
Guaratuba; apenas o histórico é reduzido para 7/15 observações. Horizonte e passo
continuam em sete dias. Covariáveis históricas e futuras são preservadas.

Os CSVs individuais ficam em `experiments/sliding/<dias>/<cidade>/<modelo>`.
`experiments/sliding/compare.py` verifica que os alvos coincidem e gera uma tabela
por cidade, incluindo os resultados expansivos originais, os deslizantes completos
e as LLMs revisadas disponíveis. Hashes verificam que os CSVs expansivos originais
não foram modificados. Execuções incompletas não entram como métricas completas.
