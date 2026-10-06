# Auditoria do Qwen3 4B — 2026-10-06

## Conclusão

O RMSE publicado foi reproduzido. Não foi encontrada troca de datas, alvos ou respostas, nem demanda futura nos prompts. A piora aparece nas previsões brutas, com forte superestimação. Há um erro real nas covariáveis de calendário de um dia imputado de Guaratuba, procedência temporal das covariáveis meteorológicas confirmada pelo usuário e sensibilidade numérica/de execução em uma repetição isolada. Não se pode atribuir toda a piora apenas ao comprimento do contexto sem controles adicionais.

## Cobertura e verificações

- Quatro execuções completas, 316 janelas e 2.212 pares observado/previsão: 15 dias e expansivo, nos dois datasets.
- 3.498 checagens de integridade aprovadas (checks.csv). As checagens adicionais de calendário encontraram a inconsistência descrita abaixo.
- Cada prompt salvo foi reconstruído a partir dos dados e comparado byte a byte com o payload enviado ao Ollama.
- Hashes de código, dados, execução e request conferidos, incluindo caches reutilizados. O comparador valida mesmo checkpoint/digest/configuração nas duas condições, exceto o tamanho do histórico.
- As sete datas de saída, em ordem, correspondem ao CSV; os valores de previsão coincidem com o JSON bruto do Ollama e os alvos coincidem exatamente com o dataset.
- O RMSE foi recalculado por produto interno dos erros, sem usar a função de métricas do pipeline.
- História estritamente anterior à origem; sem water_produced na tabela futura. Substituir TODA a demanda a partir da origem por um sentinela não altera o prompt.
- Pré-processar somente o prefixo anterior à origem produz o mesmo histórico que o pipeline completo. Não há backfill/interpolação com demanda futura nesse caminho.
- Nenhum fine-tuning ou exemplo few-shot: inferência zero-shot. Observações de janelas já encerradas entram no histórico das seguintes, conforme a avaliação expansiva solicitada; os sete alvos da janela corrente não entram.
- Sem evidência de truncamento: 26.597–52.888 tokens nos prompts expansivos de Joinville e 58.798–116.204 em Guaratuba, abaixo de 131.072 incluindo a resposta. Todas as respostas estão completas. O log disponível cobre as últimas 124 janelas originais e a repetição; seus 125 registros dizem truncated=0. Não há log de servidor preservado para todas as janelas antigas, então a evidência completa dessas vem do payload, contadores e respostas auditadas.

## Padrão dos erros

| Dataset | Contexto | RMSE | Viés médio (previsto − real) | Pontos superestimados | Sequências não decrescentes |
|---|---|---:|---:|---:|---:|
| joinville | 15_days | 16797.92 | 10458.84 | 80.4% | 5/54 |
| joinville | expanding | 26312.78 | 22077.23 | 93.9% | 37/54 |
| guaratuba | 15_days | 2768.82 | 1233.68 | 68.4% | 19/104 |
| guaratuba | expanding | 8262.07 | 6336.52 | 90.5% | 66/104 |

No expansivo, 28/54 janelas de Joinville e 51/104 de Guaratuba são progressões aritméticas exatas. Nos 15 dias, são 2/54 e 12/104. Isso descreve um comportamento observado; não comprova sua causa interna.
O quadrado do viés médio representa 70,4% do MSE expansivo de Joinville e 58,8% de Guaratuba. Mesmo retirando as cinco piores janelas de cada cidade, o RMSE permanece alto: 24.638,54 e 7.680,30. A piora não depende de uma única resposta extrema.

### Exemplo: Joinville, origem 18/08/2024

- Resposta original: [228000, 235000, 242000, 249000, 256000, 263000, 270000].
- Real: [206367.57, 212413.20, 209247.62, 200276.56, 210469.96, 217792.58, 200553.65].
- Média histórica dos sete dias anteriores: 206.353,87; média real seguinte: 208.160,16; média prevista: 249.000. RMSE da janela: 43.694,24.
Outras janelas e respostas estão em worst_windows.csv e worst_examples.json.

## Erro confirmado: calendário no dia imputado

O CSV de Guaratuba não contém 29/02/2016. daily_grid insere esse dia e fill_past_only aplica ffill indiscriminadamente a todas as colunas, incluindo calendário. O dia 29 fica com day=28, codificação de domingo e is_weekend=1, embora fosse segunda-feira. O alvo é preenchido com o passado e o dia não é pontuado, portanto isso não é vazamento de alvo. Entretanto o calendário histórico fica incorreto.
Esse dia está no histórico de todas as 104 janelas expansivas de Guaratuba e em nenhuma janela de 15 dias. Joinville não tem esse defeito. Logo, não explica o RMSE 26 mil de Joinville. A correção deve reconstruir as covariáveis determinísticas pela data; feriados/recesso exigem suas fontes. A auditoria não alterou dados, código de inferência nem resultados.

## Disponibilidade do clima futuro — confirmação do usuário

O usuário confirmou nesta auditoria: **“São previsões disponíveis na origem”**. Sob essa informação, os valores meteorológicos dos sete dias futuros são entradas legítimas, não observações futuras vazadas. As covariáveis foram mantidas.

A confirmação é informação fornecida pelo responsável pelos dados, não uma reconstrução independente de timestamps de emissão. O repositório contém CSVs processados sem esse versionamento; é recomendável guardar a procedência para reprodução futura. Nos caminhos auditados não foi identificado vazamento de demanda futura. Não é possível inspecionar etapas externas anteriores à criação dos CSVs a partir dos arquivos disponíveis.

## Repetição isolada do prompt

O mesmo payload da janela de 18/08/2024 foi repetido após descarregar o modelo, usando o mesmo digest, Ollama 0.34.2, temperatura 0 e seed 42. A nova resposta foi [230000.0, 235000.0, 240000.0, 245000.0, 250000.0, 255000.0, 260000.0], com RMSE 38721.41. A resposta mudou, mas a superestimação e a rampa persistiram sem reaproveitar o prefixo da janela anterior.
Isso mostra que temperatura zero/seed não garantiram igualdade nessa comparação entre execução sequencial e isolada. Uma repetição não isola cache, quantização, versão ou detalhes numéricos do runtime. O cache KV q4_0 está confirmado no log atual; os metadados das execuções antigas registram a configuração pretendida, mas não a versão efetiva do servidor por janela. Não há evidência suficiente para declarar um bug do Ollama ou inocentá-lo totalmente.
Os resultados originais foram preservados; a repetição fica apenas em cold_replay_joinville_2024-08-18.json.

## Próximos controles recomendados

1. Corrigir a geração de calendário no dia imputado e registrar a procedência temporal do clima confirmada pelo usuário.
2. Fazer controles em janelas pré-definidas para separar comprimento do prompt, cache e quantização, com versão/configuração efetiva do servidor registrada.
3. Se ajustar prompts, escolher em validação cronológica anterior ao teste; não escolher o melhor prompt pelas janelas de teste desta auditoria.

Nenhuma métrica foi corrigida artificialmente, nenhuma previsão original foi substituída e o Canva não foi alterado nesta auditoria.
