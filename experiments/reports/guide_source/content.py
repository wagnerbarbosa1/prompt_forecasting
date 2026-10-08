"""Editorial source synchronized with the current text-dataset Qwen pipeline."""
INTRO = [('Como usar este livro e o que mudou',
  'Este guia mantém o formato didático da edição anterior: visão de arquitetura, uma previsão '
  'acompanhada do início ao fim, conceitos de Python aplicados, contratos de funções, código real '
  'comentado linha a linha e catálogo das APIs utilizadas. O PDF possui sumário e marcadores '
  'navegáveis; o HTML permite pesquisar nomes e trechos. Os números de linha e os hashes são extraídos '
  'dos arquivos atuais, sem reescrever suas listagens.\n'
  'A edição anterior, com 174 páginas, foi analisada antes desta atualização. Sua geração combinava '
  'conteúdo editorial com análise sintática AST do Python e composição A4 pelo ReportLab. Esta edição '
  'conserva esse método, mas atualiza inventário, contratos, comentários semânticos, configurações e '
  'comandos. Não reutiliza anotações antigas em linhas cujo significado mudou.\n'
  'As mudanças centrais são: calendário escrito em inglês; exportação e leitura direta dos CSVs '
  'textuais; prompts narrativos; contexto turístico de Guaratuba; Qwen restrito a sete ou quinze dias; '
  'execução comparável dos checkpoints 1.7B e 4B; métricas e gráficos específicos dessa avaliação. Não '
  'há treinamento dos pesos Qwen. Os novos baselines clássicos são ajustados por fold. O módulo train.py é uma '
  'extensão auxiliar.\n'
  'Dados completos, prompts de todas as janelas e respostas extensas não são reproduzidos. O guia '
  'descreve seu esquema, sua origem e onde encontrá-los. O próprio gerador documental fica fora de src '
  'para não alterar a identidade das inferências enquanto a documentação é atualizada.'),
 ('Arquitetura e responsabilidades',
  'src/data/ingestion lê os arquivos; validation exige datas únicas, ordem diária e demanda válida; '
  'preprocessing transforma calendário, expõe lacunas, preenche somente com passado e constrói prompts. '
  'features define o contrato de covariáveis e separa história de futuro. models adapta a interface '
  'local do Ollama e as bibliotecas de Chronos/TimesFM. evaluation fornece origens cronológicas e '
  'métricas determinísticas. pipelines conecta essas camadas; visualization transforma previsões já '
  'concluídas em gráficos.\n'
  'O padrão de adaptador está nas classes dos modelos, que expõem predict apesar de usar backends '
  'diferentes. create_model funciona como fábrica simples. Dataclasses como ForecastWindow e '
  'OllamaConfig agrupam contratos explícitos. Configurações JSON separam parâmetros dos algoritmos. Não '
  'há uma infraestrutura de treinamento escondida na conversão textual: trocar a representação do '
  'calendário não ajusta pesos nem aprende estatísticas do conjunto de teste.\n'
  'O benchmark tradicional continua documentado porque Chronos e TimesFM utilizam seus CSVs numéricos e '
  'histórico crescente. O experimento textual do Qwen tem entrada e diretório próprios; a configuração '
  'marca textual_calendar=true e classifica dias, estações e indicadores como categorias textuais. As '
  'duas representações não devem ser confundidas nem passadas indiscriminadamente ao mesmo adaptador.'),
 ('Mapa de chamadas: do CSV textual aos resultados',
  'A entrada recomendada é python -m src.pipelines.qwen_text_experiment. main chama prepare, que '
  'percorre as quatro combinações de checkpoint/contexto, lê os dois datasets textuais, gera os 632 '
  'prompts e verifica seu tamanho conservador. --prepare-only termina nessa etapa: não faz HTTP nem '
  'inferência.\n'
  'Sem essa opção, main chama run_dataset para cada uma das oito avaliações. run_dataset valida '
  'primeiro que qwen_history_days vale 7 ou 15; prepara dados/janelas; cria o adaptador Qwen; grava '
  'procedência; monta história/futuro em cada origem; escreve o prompt; verifica cache; chama predict '
  'quando necessário; valida a saída e só depois consulta a demanda real do horizonte para registrar '
  'y_true.\n'
  'QwenForecaster consulta /api/tags para localizar o checkpoint instalado e /api/show para conhecer '
  'seu limite. predict monta um POST para /api/generate no Ollama local. A resposta de transporte '
  'contém outra string JSON em response; parse_prediction interpreta esse JSON interno e devolve sete '
  'floats finitos e não negativos.\n'
  'Após as oito avaliações completas, summarize exige horizontes completos e alvos iguais, calcula '
  'métricas agregadas e por lead, e escreve comparações simples. --summarize-only recalcula resultados '
  'salvos sem chamar modelos. python -m src.visualization.qwen_text_results valida os resultados por '
  'summarize e gera seis gráficos. A atualização no Canva usa esses gráficos e as métricas calculadas; '
  'não existe publicação automática no código de inferência.'),
 ('Uma janela concreta e a proteção temporal',
  'Em Joinville, a primeira origem é o índice 365; o alvo é df.iloc[365:372]. São sete linhas, pois o '
  'limite direito do slice é exclusivo. Para Qwen com quinze dias, history é df.iloc[350:365]; com sete '
  'dias, df.iloc[358:365]. O alvo permanece o mesmo nas duas configurações e nos dois checkpoints. Em '
  'Guaratuba, a origem inicial é 730 e aplicam-se os mesmos recortes relativos.\n'
  'Na origem seguinte, o índice avança sete dias e o início da história do Qwen também avança sete. Seu '
  'tamanho permanece sete ou quinze; somente Chronos e TimesFM recebem todo o histórico desde o começo. '
  'context do dataset escolhe a primeira origem; qwen_history_days escolhe a quantidade de observações '
  'fornecidas ao LLM; num_ctx limita tokens. Essas três medidas têm funções diferentes.\n'
  'model_inputs constrói future selecionando timestamp e covariáveis, sem water_produced. Verifica '
  'também que a maior data histórica precede a menor data futura. build_prompt rejeita uma coluna '
  'futura de demanda e exige exatamente as datas seguintes ao histórico. Os testes alteram alvos '
  'futuros por uma sentinela e exigem que o prompt não mude. Isso testa isolamento dos alvos; não '
  'comprova a disponibilidade operacional de toda covariável.\n'
  'As configurações novas usam zero-shot. O seletor de exemplos permanece disponível: quando '
  'solicitado, recebe apenas o prefixo congelado anterior ao primeiro teste, evitando transformar alvos '
  'testados em demonstrações de treinamento. Observações de janelas anteriores podem compor o histórico '
  'observado da previsão seguinte, como definido pelo protocolo walk-forward; isso não implica ajuste '
  'de pesos do Qwen.\n'
  'Guaratuba tem uma lacuna em 29/02/2016. daily_grid a expõe; observed é calculado antes do '
  'preenchimento; fill_past_only propaga dados anteriores. Alvos artificiais são excluídos das janelas '
  'de teste. Esse preenchimento não reconstrói automaticamente o calendário: se a lacuna entrasse em um '
  'contexto usado, seria necessário conferir seus atributos. Os contextos fixos avaliados começam muito '
  'depois dessa lacuna; o guia registra a limitação sem ocultar o comportamento real.'),
 ('Calendário em palavras: exportação e leitura',
  'calendar_text.py é a única implementação da conversão. number_to_words escreve inteiros de zero a '
  '9999 sem depender do locale. date_to_text transforma 2016-01-01 em “two thousand and sixteen, day '
  'one of January”. O ano não é alterado: 2026 seria “two thousand and twenty-six”. Dias são cardinais, '
  'como “one”, conforme a representação escolhida.\n'
  'Os pares dayofweek_sin/cos e season_sin/cos são decodificados pelo ângulo atan2(seno, cosseno), '
  'normalizado ao ciclo. O índice arredondado identifica segunda a domingo ou verão, outono, inverno e '
  'primavera, em inglês. A comparação allclose com seno/cosseno reconstruídos verifica se os valores '
  'correspondem a uma categoria válida. Estações são decodificadas do valor fornecido, em vez de '
  'aproximadas somente pelo mês; isso preserva transições como 20 de março. Um par incompleto é '
  'rejeitado.\n'
  'Indicadores booleanos ou zero/um recebem expressões como “a public holiday”, “not a public holiday”, '
  '“during Carnival” e “during the Paraná school recess”. As colunas de demanda e meteorologia '
  'conservam seus números. Os pares cíclicos viram as colunas dayofweek e season; por isso o contrato '
  'textual possui colunas diferentes do contrato numérico original.\n'
  'O CLI de calendar_text escreve cópias em data/processed/calendar_text, sem sobrescrever as fontes. '
  'Para prever a partir delas, read_series recebe textual_calendar=true e aplica date_from_text somente '
  'ao timestamp. words_to_number aceita a grafia canônica emitida pelo exportador; datas inválidas são '
  'rejeitadas. A conversão da data de volta a Timestamp existe para ordenação e recortes temporais; os '
  'demais campos de calendário continuam em palavras. Ao narrar, calendar_to_text aceita os indicadores '
  'e dias já convertidos, sem convertê-los novamente em códigos numéricos.\n'
  'A exportação inteira não é um ajuste estatístico. Cada linha depende apenas de seus próprios '
  'valores. Ler o arquivo completo para administrar janelas não significa fornecê-lo completo ao LLM: '
  'somente os recortes permitidos são serializados em cada pedido.'),
 ('Estrutura narrativa dos prompts',
  'build_prompt recebe history, horizon, demonstrations opcionais, future_covariates opcionais e '
  'region. LABELS traduz nomes técnicos em descrições: temperature_mean vira average temperature e '
  'water_produced vira water production (water_produced). _narrate escreve uma sentença por variável, '
  'com valores em ordem cronológica: “From {t1} to {tobs}, the average temperature of region {region} '
  'was {valores} degrees Celsius on each day, respectively.”\n'
  'A lista de datas por extenso informa a correspondência entre posição e dia. Cada sequência tem '
  'exatamente um valor por observação; respectively deixa explícito o alinhamento. Demanda é formatada '
  'com duas casas decimais; números meteorológicos usam seis algarismos significativos. Esse '
  'arredondamento pertence ao texto do prompt, sem substituir os números usados para avaliação.\n'
  'As covariáveis futuras são narradas em bloco próprio como supplied inputs. Sua demanda não está '
  'presente. Ao fim, há uma pergunta “What is the water_produced going to be on ...?” para cada uma das '
  'sete datas futuras. A ordem das perguntas é a ordem exigida no array predictions. Não há tabela CSV '
  'embutida nem função alternativa de prompt tabular no pipeline atual.\n'
  'Quando region começa por Guaratuba, o prompt informa que a cidade é turística e tem forte '
  'sazonalidade durante férias e recessos brasileiros, especialmente de dezembro a março. Joinville não '
  'recebe essa afirmação. Esse contexto é uma instrução geral solicitada para o modelo, não um valor '
  'calculado a partir dos alvos de teste.\n'
  'Texto mais natural não garante previsão melhor. O modelo pode confundir grandezas ou produzir '
  'valores de escala inadequada e ainda respeitar o JSON. O validador verifica estrutura e domínio '
  'numérico; não altera previsões para aproximá-las da verdade observada. O desempenho precisa ser '
  'medido nos CSVs e não inferido da fluência do prompt.'),
 ('Python aplicado e contratos de funções',
  'Um argumento posicional é associado pela posição; um argumento nomeado é associado pelo nome. Nas '
  'assinaturas, * obriga os argumentos seguintes a serem nomeados, reduzindo trocas acidentais entre '
  'horizon, future_covariates e region. None é ausência de valor, mas não é permitido como '
  'qwen_history_days: o validador exige type(days) is int e days em (7, 15). bool é subclasse de int em '
  'Python; a comparação exata de tipo impede True de ser tratado como um número de dias.\n'
  'DataFrame guarda colunas com tipos distintos: demanda e meteorologia numéricas, timestamp temporal e '
  'calendário textual. .iloc utiliza posições, enquanto .loc utiliza rótulos. slice(start, stop) não '
  'materializa dados; o recorte ocorre quando aplicado à tabela. .copy evita que a apresentação textual '
  'modifique a fonte. Iterar uma Series mantém sua ordem, essencial para que valores e datas permaneçam '
  'alinhados.\n'
  'Dataclasses geram inicialização e representação; frozen=True impede reatribuir campos normais da '
  'instância, mas não torna objetos internos recursivamente imutáveis. Decoradores property expõem '
  'history e target como atributos calculados. List comprehensions criam coleções; geradores de any/all '
  'podem parar cedo. json.dumps serializa objetos; encode produz bytes; json.load lê o envelope HTTP; '
  'json.loads interpreta o texto interno do modelo.\n'
  'Exceções não são previsões. run_dataset registra uma falha em status/responses e não inventa valores '
  'substitutos. O resumo exige execuções completas para impedir comparação de subconjuntos diferentes. '
  'As chamadas de mock dos testes validam contratos sem carregar modelos; os CSVs da execução real '
  'comprovam a execução do Ollama.'),
 ('Ollama local, recursos e reprodutibilidade',
  'As quatro configurações qwen_text usam http://127.0.0.1:11435, num_ctx=16384, num_predict=256, '
  'temperature=0 e seed=42. Os checkpoints são qwen3:1.7b e qwen3:4b-instruct-2507-q4_K_M. Os pesos '
  'precisam estar instalados localmente. O código não baixa pesos e o trabalho não instala nem modifica '
  'ambientes Conda.\n'
  'num_ctx é capacidade de tokens por sequência, não número de dias; num_predict limita tokens de '
  'resposta, não a quantidade de previsões. O JSON Schema é quem exige sete números. A checagem '
  'anterior ao pedido usa bytes UTF-8 + num_predict + 512 como limite superior conservador; a contagem '
  'real de tokens pode ser menor. O adaptador também rejeita respostas incompletas e situações em que a '
  'contagem retornada alcança o limite.\n'
  'think=False desativa o modo de raciocínio configurável no pedido; stream=False solicita um envelope '
  'completo; format recebe o schema; truncate=False pede que a entrada não seja truncada; keep_alive '
  'mantém o modelo carregado entre pedidos. Essas opções não constituem treinamento. Timeout controla '
  'espera HTTP e retries controla novas tentativas após falhas de transporte ou validação. Uma previsão '
  'válida porém ruim não é tentada novamente por causa de seu erro posterior.\n'
  'As variáveis de start_ollama_context_server.sh configuram o processo servidor, incluindo Flash '
  'Attention, cache KV quantizado e paralelismo. Os campos runtime em configurações antigas descrevem '
  'essas opções; passá-los ao Python não reinicia o servidor. OllamaConfig reúne opções aceitas pelo '
  'adaptador.\n'
  'run_id incorpora configuração, hashes de dataset e código, identidade/digest do modelo e protocolo. '
  'request_id inclui run_id, janela e prompt. Cache é reutilizado apenas para o mesmo identificador e '
  'status ok. Mudança em qualquer .py de src altera source_sha256 e invalida cache; por isso o gerador '
  'do guia fica em experiments/reports/guide_source. Determinismo configurado favorece reprodução, mas '
  'hardware/backend podem influenciar detalhes numéricos.'),
 ('Métricas, gráficos, artefatos e limites',
  'Cada configuração salva predictions.csv por cidade, com origin, timestamp, lead de um a sete, '
  'y_true, y_pred, contexto e procedência. status.json registra completude; responses preserva pedido, '
  'tentativas e resposta bruta; prompts guarda o texto enviado. prompt_manifest.json descreve as 632 '
  'janelas. input_audit.json registra igualdade dos prompts entre checkpoints, invariância à mutação '
  'dos alvos futuros e correspondência das datas de teste com Chronos.\n'
  'comparison_<cidade>.csv possui somente modelo, contexto inicial, RMSE, R2, MAE e MAPE. Nesse resumo, '
  'contexto inicial é sete ou quinze dias efetivamente fornecidos ao LLM; em predictions.csv, '
  'initial_context continua sendo 365 ou 730, isto é, o limiar inicial das origens de avaliação. '
  'context registra o tamanho real da entrada. metrics.csv acrescenta cidade, sMAPE e contagens; '
  'metrics_by_horizon.csv separa os sete leads.\n'
  'MAE é a média de |previsto-real|. RMSE é a raiz da média dos erros quadráticos sobre todos os '
  'pontos, não a média dos RMSEs das janelas. R² = 1 - soma dos erros quadráticos / soma dos desvios '
  'quadráticos da verdade em relação à sua média. R² negativo é permitido e indica desempenho pior que '
  'essa referência constante; não deve ser cortado para zero. MAPE é expresso em porcentagem e exclui '
  'alvos iguais a zero, registrando sua contagem; sMAPE usa denominador |real|+|previsto| e define '
  'contribuição zero quando ambos são zero.\n'
  'São 54 janelas/378 pontos por configuração em Joinville e 104/728 em Guaratuba: oito avaliações, 632 '
  'janelas e 4.424 previsões. summarize verifica todos os sete leads e alvos idênticos entre '
  'configurações. O módulo gráfico produz quatro figuras com painéis separados para sete/quinze dias, '
  'uma por cidade/checkpoint, e duas comparações de RMSE. Não suaviza nem exclui previsões ruins.\n'
  'O adaptador barra alvos futuros nos inputs, mas as covariáveis meteorológicas futuras são fornecidas '
  'pelos CSVs e sua disponibilidade na origem não está comprovada. As métricas são condicionais a essas '
  'entradas; não demonstram desempenho operacional livre de informação futura. Não confundir ausência '
  'de treinamento com prova de causalidade de todas as entradas.\n'
  'Resultados numéricos pertencem aos CSVs gerados, não a constantes no guia. O Canva é atualizado com '
  'valores e imagens extraídos desses artefatos após conferência. Alterar o PDF não executa inferências '
  'nem atualiza automaticamente uma apresentação.'),
 ('Mini-exemplo: palavras, HTTP e avaliação',
  'Considere duas observações didáticas: 1 e 2 de janeiro de 2016, demandas 100 e 110 e temperaturas 20 '
  'e 21. O horizonte deste exemplo é dois apenas para acompanhar os objetos; a avaliação real mantém '
  'horizonte sete e contexto mínimo sete. O CSV textual guarda “two thousand and sixteen, day one of '
  'January” e o dia “one”. read_series interpreta a data para Timestamp, mantendo o dia em palavras.\n'
  'build_prompt produz, entre outras frases: “From two thousand and sixteen, day one of January to two '
  'thousand and sixteen, day two of January, the average temperature of region Joinville was 20; 21 '
  'degrees Celsius on each day, respectively.” Outra frase descreve a demanda histórica. As perguntas '
  'usam os dias três e quatro; nenhuma delas contém a demanda real dessas datas.\n'
  'prediction_schema(2) exige um objeto com predictions de dois números. json.dumps(payload).encode '
  'converte o pedido em bytes; urlopen envia o POST. Suponha response contendo a string '
  '{"predictions":[112,114]}. Primeiro json.load interpreta o envelope HTTP; depois parse_prediction '
  'interpreta essa string e devolve [112.0, 114.0]. São dois níveis de JSON, não duas previsões '
  'independentes.\n'
  'Só depois da inferência a avaliação consulta a verdade [111,117]. Os erros são +1 e -3; MAE=2 e '
  'RMSE=sqrt(5), aproximadamente 2,236. Se a resposta trouxer a string “112” em lugar do número 112, o '
  'parser rejeita; se trouxer apenas um valor, rejeita antes do zip. Se trouxer dois números válidos '
  'porém muito errados, preserva-os: o erro pertence à avaliação e não deve ser corrigido usando o '
  'alvo.')]

PURPOSE = {'src/data/ingestion/files.py': 'Lê CSV numérico ou textual conforme textual_calendar. Interpreta datas '
                                'escritas antes da validação e oferece escrita JSON atômica para '
                                'artefatos.',
 'src/data/validation/schema.py': 'Contrato do CSV: datas diárias únicas e demanda finita, não '
                                  'negativa. Chamado por read_series. Retorna cópia ordenada, '
                                  'preservando covariáveis; erro invalida o dataset antes do modelo.',
 'src/data/preprocessing/resampling.py': 'Converte datas observadas em grade diária. Chamado por '
                                         'prepare_dataset antes do preenchimento. Adiciona observed '
                                         'para distinguir alvos reais de linhas inseridas.',
 'src/data/preprocessing/missing_values.py': 'Preenchimento causal. Chamado após daily_grid. Retorna '
                                             'tabela sem lacunas iniciais de demanda; o preenchimento '
                                             'indiscriminado do calendário é um defeito confirmado, não '
                                             'uma prática recomendada.',
 'src/data/features/temporal.py': 'Geração determinística das datas-alvo a partir da última data '
                                  'histórica. Chamado pelos dois construtores de prompt; não consulta a '
                                  'demanda futura.',
 'src/data/features/covariates.py': 'Contrato comum dos modelos: nomes de covariáveis, validação, '
                                    'separação history/future e conversão para TimesFM. model_inputs é '
                                    'a fronteira que exclui estruturalmente o alvo da entrada futura.',
 'src/evaluation/cross_validation.py': 'Define ForecastWindow e origens walk-forward. prepare_dataset '
                                       'chama build_windows. validation_split é usado por '
                                       'training_partition e testes, não para ajustar os modelos neste '
                                       'benchmark.',
 'src/evaluation/metrics.py': 'Função numérica sem I/O. Recebe vetores reais e previstos já alinhados e '
                              'retorna dicionário de métricas. É chamada pelos comparadores, testes e '
                              'auditoria.',
 'src/data/preprocessing/generate_prompts.py': 'Constrói exclusivamente prompts narrativos. Recebe '
                                               'história e covariáveis futuras sem alvo, descreve todas '
                                               'as variáveis e aplica o contexto turístico somente a '
                                               'Guaratuba.',
 'src/models/promptcast.py': 'Adaptador HTTP do Qwen: configuração, schema de saída, parser, cliente '
                             'JSON e classe QwenForecaster. forecasting_pipeline chama predict; '
                             'post_json encapsula transporte, e parse_prediction valida sem corrigir '
                             'artificialmente valores.',
 'src/models/chronos.py': 'Adapter de Chronos2Pipeline. Importações pesadas ficam dentro do construtor, '
                          'permitindo importar o pipeline em outros ambientes. Usa covariáveis '
                          'passadas/futuras e devolve o quantil 0,5 como previsão pontual.',
 'src/models/timesfm.py': 'Adapter de TimesFM 2.5 com XReg. Prepara capacidade suficiente para todo o '
                          'histórico a cada origem e devolve somente os primeiros horizon valores. Não '
                          'corta a história ao contexto inicial.',
 'src/pipelines/data_pipeline.py': 'Composição de leitura, validação, grade diária, preenchimento '
                                   'causal e janelas. textual_calendar seleciona a interpretação das '
                                   'datas do CSV de palavras.',
 'src/pipelines/forecasting_pipeline.py': 'Orquestra inferência e cache. Exige Qwen com 7/15 dias, '
                                          'mantém histórias de baseline e preserva somente covariáveis '
                                          'no frame futuro. Não calcula métricas agregadas sozinho.',
 'src/pipelines/compare_models.py': 'Compara execuções completas do benchmark Qwen/Chronos/TimesFM. '
                                    'Valida contextos fixos do Qwen e rotula corretamente sete ou '
                                    'quinze dias.',
 'src/training/train.py': 'Ponto auxiliar para uma extensão de treinamento: oferece slices cronológicos '
                          'treino/validação. Não é chamado pela inferência atual nem treina pesos.',
 'src/data/preprocessing/calendar_text.py': 'Converte calendário para inglês, valida codificações '
                                            'cíclicas, exporta CSVs textuais e interpreta suas datas '
                                            'para os recortes temporais. Não treina nem consulta outras '
                                            'linhas para calcular uma categoria.',
 'src/pipelines/qwen_text_experiment.py': 'CLI das oito avaliações textuais: gera todos os prompts, '
                                          'executa checkpoint/contexto/cidade e resume métricas '
                                          'completas, alinhadas e por lead.',
 'src/visualization/qwen_text_results.py': 'Produz seis gráficos das oito avaliações textuais '
                                           'concluídas: quatro pares observado/previsto e duas '
                                           'comparações de RMSE. Usa todos os sete leads.',
 'src/visualization/forecast_comparison.py': 'Renderiza os resultados históricos preservados dos '
                                             'baselines e dos Qwen de quinze dias. É distinto dos novos '
                                             'gráficos do experimento textual.',
 'src/visualization/plots.py': 'Funções de desenho e exportação usadas pela visualização histórica, '
                               'mantendo o estilo copiado de Siaqua.',
 'src/visualization/colors.py': 'Paleta e seleção de cores das funções de visualização histórica.'}

FUNCTIONS = {'read_series': 'path é o CSV; textual_calendar=False lê datas usuais; True aplica date_from_text à '
                'coluna timestamp. Retorna DataFrame validado e ordenado. Calendário categórico '
                'continua textual.',
 'write_json': 'path: destino; value: dict serializável. Sem retorno útil. Cria pais, escreve .tmp e '
               'substitui destino.',
 'validate_series': 'df: tabela lida. Retorno: cópia ordenada. Rejeita esquema, datas e demanda '
                    'inválidos antes de qualquer modelo.',
 'daily_grid': 'df: tabela com timestamp. Retorno: grade diária com observed. Dias ausentes são '
               'expostos, não removidos.',
 'fill_past_only': 'df: grade com possíveis NaN. Retorno: cópia preenchida com passado. Não reconstrói '
                   'atributos de calendário, limitação conhecida.',
 'future_dates': 'last_observed: último Timestamp histórico; horizon: quantidade de dias. Retorna '
                 'DatetimeIndex dos próximos dias.',
 'covariate_columns': 'dataset: bloco de configuração de uma cidade. Retorna nomes numéricos seguidos '
                      'dos categóricos, após rejeitar duplicatas e nomes reservados.',
 'validate_covariates': 'df: tabela preparada; dataset: contrato de colunas. Retorna None; falha em '
                        'ausentes/NaN/números não finitos.',
 'model_inputs': 'df: série completa; window: ForecastWindow; dataset: contrato; history_days: None '
                 'para expansivo ou inteiro para últimos N dias. Retorna (history, future), excluindo '
                 'alvo do futuro.',
 'timesfm_covariates': 'history/future: tabelas separadas; dataset: listas numéricas/categóricas. '
                       'Retorna dois dicts com listas de séries cobrindo passado e futuro, sem demanda '
                       'futura.',
 'build_windows': 'df: tabela preparada; context: primeira origem; horizon=7: quantidade prevista; '
                  'step=7: avanço entre origens. Retorna lista ForecastWindow, excluindo janelas com '
                  'alvo imputado.',
 'validation_split': 'training_size: comprimento do prefixo anterior ao teste; fraction=0.2: parte '
                     'final reservada à validação. Retorna dois slices sem embaralhar.',
 'deterministic_metrics': 'ytrue/yhat: vetores de mesmo formato, não vazios e finitos. Retorna MAE, '
                          'RMSE, MAPE%, sMAPE%, R² e contagem de zeros excluídos do MAPE.',
 'select_demonstrations': 'training_values: prefixo congelado anterior ao primeiro teste; history: '
                          'demanda histórica corrente; horizon: tamanho de alvo dos exemplos; strategy: '
                          'zero/few_random/few_similar; count=3: quantidade; seed=42: sorteio. Retorna '
                          'lista de Demonstration.',
 'standardized': 'x: vetor de exemplo/consulta histórica. Retorna valores centrados e divididos por '
                 'desvio-padrão com piso 1e-8; apenas auxilia seleção por forma.',
 'format_values': 'values: números históricos. Retorna string separada por vírgulas com duas casas '
                  'decimais.',
 'build_prompt': 'history deve conter dias consecutivos e demanda finita. horizon deve ser positivo; '
                 'future_covariates não pode conter water_produced e precisa coincidir com o horizonte. '
                 'region aplica contexto de Guaratuba. Retorna uma string narrativa com uma pergunta '
                 'por dia e schema JSON.',
 'prediction_schema': 'horizon: comprimento exigido do array. Retorna JSON Schema para objeto '
                      'predictions, números não negativos e ausência de chaves extras.',
 'parse_prediction': 'text: string JSON do modelo; horizon: quantidade exata de valores. Retorna '
                     'list[float] ou lança erro; não remove prose nem aceita resposta parcial.',
 'post_json': 'url: endpoint; payload: dict do pedido; timeout: segundos de espera. Retorna dict '
              'decodificado do envelope HTTP, não ainda a lista de previsões.',
 'fingerprint': 'value: objeto serializável. Retorna SHA-256 hexadecimal do JSON com chaves ordenadas.',
 'source_fingerprint': 'Sem parâmetros. Retorna digest incremental de nomes e bytes de todos os .py de '
                       'src, em ordem fixa.',
 'create_model': 'name: qwen/chronos/timesfm; context: contexto inicial; config: configuração global; '
                 'dataset: configuração da cidade. Retorna adaptador apropriado. A fábrica não prevê '
                 'por si só.',
 'run_dataset': 'name, dataset e config identificam cidade e parâmetros; model_name seleciona '
                'adaptador; output define pasta; limit só limita testes de fumaça. Valida contexto, '
                'executa janelas, salva previsões e status; retorna número de falhas.',
 'prepare_dataset': 'path aponta o CSV; context define a primeira origem; horizon/step definem grade de '
                    'avaliação; textual_calendar escolhe interpretação de datas em inglês. Retorna '
                    'tabela preparada e lista de ForecastWindow.',
 'compare': 'output contém benchmark completo de três modelos. Verifica mesmas '
            'origens/alvos/procedência, contexto Qwen7/15 e histórias crescentes dos baselines; grava '
            'métricas.',
 'main': 'Entrada CLI do módulo correspondente. Lê argumentos ou coordena a rotina documentada no '
         'início do arquivo. Os efeitos variam: conversão, execução, métricas ou gráficos; não são a '
         'mesma função compartilhada.',
 'training_partition': 'first_test_origin: tamanho de prefixo de treino; validation_fraction=0.2: '
                       'proporção final de validação. Retorna os slices de validation_split; não faz '
                       'fit.',
 'number_to_words': 'value é um inteiro de 0 a 9999; retorna grafia inglesa canônica. Rejeita valores '
                    'fora do domínio ou não inteiros; recursão combina milhares, centenas e dezenas.',
 'words_to_number': 'text deve seguir a grafia canônica do exportador. Reconstrói o inteiro e compara '
                    'sua grafia de volta; rejeita palavras ou formas inesperadas.',
 'date_to_text': 'value é uma data ou sua representação textual canônica. Retorna ano, dia e mês em '
                 'inglês sem mudar o ano original.',
 'date_from_text': 'text é a data no formato exato do exportador. Valida com fullmatch, converte '
                   'ano/dia e retorna Timestamp; utilizado somente na organização temporal dos inputs.',
 '_cycle_labels': 'frame contém os dois componentes prefix_sin/cos; labels define a ordem do ciclo. '
                  'Decodifica o ângulo, valida seno/cosseno reconstruídos e retorna uma categoria por '
                  'linha.',
 'calendar_to_text': 'frame pode conter calendário numérico ou já convertido. Devolve cópia com '
                     'calendário inglês, mescla pares cíclicos e preserva demanda/meteorologia. Não '
                     'ajusta estatísticas nem altera o argumento.',
 '_narrate': 'frame é um recorte de história ou futuro sem alvo; region identifica a cidade; future '
             'altera os tempos verbais e indica covariáveis fornecidas. Retorna lista de frases '
             'alinhadas às datas.',
 'qwen_history_days': 'config precisa conter inteiro exato 7 ou 15. Ausência, None, strings, bool, '
                      'floats e outros comprimentos são rejeitados antes de carregar dados/modelo.',
 'configuration': 'slug identifica 1_7b ou 4b e days identifica 7/15. Lê o arquivo JSON correspondente '
                  'em configs e devolve o dicionário.',
 'prepare': 'Gera e grava todos os prompts do experimento textual, verificando covariáveis e limite '
            'conservador de entrada. Não chama Ollama. Escreve prompt_manifest.json.',
 'summarize': 'Exige as oito avaliações completas, contextos corretos, sete leads e alvos iguais. '
              'Escreve CSVs simples por cidade/execução, métricas agregadas e por lead. Não altera '
              'previsões.'}

CONFIG_KEYS = {'datasets': 'Mapeamento dos dois datasets; cada bloco fornece caminho, primeira origem e contrato de '
             'covariáveis.',
 'joinville': 'Bloco da série de Joinville. Contexto inicial de avaliação: 365 dias; total atual 54 '
              'janelas completas.',
 'guaratuba': 'Bloco da série de Guaratuba. Contexto inicial de avaliação: 730 dias; total atual 104 '
              'janelas completas.',
 'path': 'Caminho relativo à raiz do projeto. Os arquivos qwen_text apontam os CSVs dentro de '
         'data/processed/calendar_text.',
 'context': 'Tamanho inicial que define a primeira origem do dataset: 365 ou 730. Qwen recebe o recorte '
            'fixo de 7/15; baselines recebem histórico crescente.',
 'numerical_covariates': 'Colunas numéricas preservadas nos inputs. Nas configurações textuais, contém '
                         'somente meteorologia; o calendário convertido é categórico.',
 'categorical_covariates': 'Categorias e indicadores. Na entrada textual inclui day, dayofweek, season '
                           'e descrições de feriado/recesso; não são reconvertidos em códigos.',
 'horizon': 'Sete valores futuros por origem; o alvo é [origin, origin+7).',
 'step': 'Avança sete dias entre origens; com horizonte sete as janelas avaliadas não se sobrepõem.',
 'validation_fraction': 'Proporção 0,2 para validação em uma extensão de treino. A inferência zero-shot '
                        'atual não a usa para otimizar parâmetros.',
 'seed': 'Semente global transmitida ao Qwen/seleção de exemplos. Os adaptadores numéricos usam 42 '
         'internamente.',
 'qwen': 'Parâmetros que **config["qwen"] expande em OllamaConfig; chaves precisam corresponder aos '
         'campos da dataclass.',
 'model': 'Tag exata do checkpoint instalado no Ollama; o digest é consultado no servidor e guardado em '
          'run.json.',
 'base_url': 'Endereço HTTP local do Ollama. As quatro configurações qwen_text usam a porta 11435; '
             'benchmark.json conserva sua porta própria.',
 'temperature': 'Temperatura de geração enviada em options; zero não equivale a treinamento nem a '
                'garantia absoluta de determinismo.',
 'num_ctx': 'Capacidade de tokens do servidor por sequência. Configurações textuais usam 16384, '
            'verificados contra limite do modelo e tamanho conservador do prompt.',
 'num_predict': 'Limite de tokens gerados pelo Ollama. Não substitui horizon, que é quantidade de '
                'números previstos.',
 'timeout': 'Limite de espera HTTP do cliente em segundos. Pode durar mais de uma hora no caso 7200.',
 'retries': 'Tentativas adicionais depois da primeira; retries=1 permite duas chamadas para a mesma '
            'janela.',
 'strategy': 'zero retorna nenhuma demonstração; few_random/few_similar só são aceitos no layout '
             'legado.',
 'few_shot_examples': 'Quantidade de exemplos para estratégias few-shot. Ignorada efetivamente quando '
                      'strategy=zero.',
 'qwen_history_days': 'Inteiro obrigatório 7 ou 15 para Qwen. Representa histórico fixo em observações '
                      'diárias; valor nulo ou outro tamanho é rejeitado.',
 'future_covariates_policy': 'Texto de procedência armazenado com o experimento. A redação antiga diz '
                             'hipótese não verificada; posteriormente o usuário confirmou '
                             'disponibilidade na origem. Preservar o JSON preserva o hash histórico.',
 'runtime': 'Registro declarativo das opções pretendidas de servidor. forecasting_pipeline não exporta '
            'essas variáveis automaticamente; o script do servidor precisa aplicá-las.',
 'ollama_flash_attention': 'Registro da intenção de usar Flash Attention; aplicada por '
                           'OLLAMA_FLASH_ATTENTION no script do servidor.',
 'ollama_kv_cache_type': 'Registro de cache KV q4_0; diferente da quantização Q4_K_M dos pesos.',
 'ollama_num_parallel': 'Uma sequência simultânea neste experimento para limitar memória e facilitar '
                        'reutilização de prefixo.',
 'llama_arg_cache_ram': 'Zero desabilita cache adicional de snapshots; não desliga estados KV da '
                        'sequência ativa.',
 'textual_calendar': 'True ativa date_from_text na leitura do CSV para permitir os recortes temporais. '
                     'Demais covariáveis de calendário permanecem palavras.'}

NOTES = {('src/models/promptcast.py', 12): 'Nome padrão legado. Configurações do experimento 4B substituem esse '
                                   'valor ao construir OllamaConfig; alterar apenas este default não '
                                   'muda JSON já fornecido.',
 ('src/models/promptcast.py', 13): 'URL padrão legado, porta 11434. O JSON do 4B usa servidor dedicado '
                                   'local na porta 11435.',
 ('src/models/promptcast.py', 14): 'Temperatura padrão zero. É configuração de amostragem, não um '
                                   'normalizador dos dados; não garante igualdade bit a bit entre '
                                   'runtimes.',
 ('src/models/promptcast.py', 15): 'Capacidade padrão de 8192 tokens, sobrescrita para 131072 no '
                                   'experimento 4B. Não corresponde ao número de dias.',
 ('src/models/promptcast.py', 16): 'Máximo de tokens novos, contando pontuação e estrutura JSON, não '
                                   'quantidade de valores previstos.',
 ('src/models/promptcast.py', 17): 'Espera HTTP padrão de 600 segundos; configuração 4B sobrescreve '
                                   'para 7200.',
 ('src/models/promptcast.py', 18): 'Número de tentativas adicionais, não total; retries=2 permite três '
                                   'chamadas no loop.',
 ('src/models/promptcast.py', 19): 'Semente transmitida às opções de geração do Ollama.',
 ('src/models/promptcast.py', 21): 'Gancho da dataclass: executado automaticamente depois da '
                                   'inicialização gerada, para rejeitar limites inválidos.',
 ('src/models/promptcast.py', 27): 'Abre schema de objeto JSON. properties descreve a chave '
                                   'predictions, sem inserir previsões reais no pedido.',
 ('src/models/promptcast.py', 28): 'Cada item deve ser número com limite inferior zero; aqui mínimo é '
                                   'uma restrição de saída, não clipping posterior.',
 ('src/models/promptcast.py', 29): 'minItems e maxItems iguais a horizon impõem exatamente sete valores '
                                   'quando horizon=7.',
 ('src/models/promptcast.py', 30): 'required torna predictions obrigatória; additionalProperties=False '
                                   'proíbe outras chaves no objeto.',
 ('src/models/promptcast.py', 41): 'O teste type(v), diferente de isinstance(v,int), impede que '
                                   'true/false sejam aceitos como 1/0. O gerador rejeita também '
                                   'infinitos e negativos.',
 ('src/models/promptcast.py', 59): 'Filtra os modelos instalados por nome exato, incluindo '
                                   'tag/quantização. Não escolhe uma variante aproximada '
                                   'silenciosamente.',
 ('src/models/promptcast.py', 62): 'Consulta metadados do checkpoint instalado; timeout posicional de '
                                   '30 segundos só vale para esta consulta.',
 ('src/models/promptcast.py', 64): 'Obtém a arquitetura declarada e compõe a chave de metadados de '
                                   'contexto correspondente, em vez de fixar um nome de arquitetura.',
 ('src/models/promptcast.py', 65): 'Obtém a arquitetura declarada e compõe a chave de metadados de '
                                   'contexto correspondente, em vez de fixar um nome de arquitetura.',
 ('src/models/promptcast.py', 68): 'Registra modelo, digest dos pesos, contexto declarado e '
                                   'configuração efetivamente construída. Este dict participa do hash '
                                   'de cache do pipeline.',
 ('src/models/promptcast.py', 69): 'Registra modelo, digest dos pesos, contexto declarado e '
                                   'configuração efetivamente construída. Este dict participa do hash '
                                   'de cache do pipeline.',
 ('src/models/promptcast.py', 71): 'Mantém interface dos adaptadores: history e future_covariates são '
                                   'aceitos, mas este método usa o prompt já montado. horizon determina '
                                   'schema e parser; prompt deve ser nomeado após *.',
 ('src/models/promptcast.py', 76): 'Compara bytes UTF-8 do prompt + reserva de geração + 512 '
                                   'bytes/tokens de margem com num_ctx. É limite conservador, não '
                                   'tokenizer exato. Um prompt legítimo pode ser rejeitado antes do '
                                   'limite real.',
 ('src/models/promptcast.py', 78): 'Monta corpo HTTP: nome do modelo, texto completo e stream=False '
                                   'para resposta única. Isso não envia ainda a requisição.',
 ('src/models/promptcast.py', 79): 'think=False solicita geração sem modo de raciocínio; '
                                   "keep_alive='10m' controla retenção do modelo; truncate=False "
                                   'expressa intenção de não cortar; format recebe o schema. A '
                                   'auditoria do servidor continua necessária.',
 ('src/models/promptcast.py', 80): 'options agrupa configuração de amostragem e limites; são opções do '
                                   'Ollama, diferentes das opções do cliente urllib, como timeout.',
 ('src/models/promptcast.py', 81): 'options agrupa configuração de amostragem e limites; são opções do '
                                   'Ollama, diferentes das opções do cliente urllib, como timeout.',
 ('src/models/promptcast.py', 82): 'Reinicia auditoria da janela, guardando pedido exato e lista de '
                                   'tentativas. Não reutiliza uma resposta da janela anterior aqui.',
 ('src/models/promptcast.py', 83): 'range(retries+1) inclui a tentativa inicial: com retries=1, attempt '
                                   'assume 0 e 1.',
 ('src/models/promptcast.py', 84): 'Cria dict mutável por tentativa e o anexa antes do HTTP. Quando '
                                   'response/error são adicionados depois, a lista aponta para o mesmo '
                                   'dict e preserva o registro.',
 ('src/models/promptcast.py', 85): 'Cria dict mutável por tentativa e o anexa antes do HTTP. Quando '
                                   'response/error são adicionados depois, a lista aponta para o mesmo '
                                   'dict e preserva o registro.',
 ('src/models/promptcast.py', 89): "done ausente/False ou done_reason='length' rejeita resposta "
                                   'incompleta, mesmo que haja texto parseável.',
 ('src/models/promptcast.py', 91): 'Soma tokens de prompt e saída para rejeitar saturação. get(...,0) '
                                   'tolera campos ausentes, então esta checagem sozinha não certifica '
                                   'contadores válidos em toda versão.',
 ('src/models/promptcast.py', 93): 'Extrai apenas a string response do envelope e valida sete números. '
                                   'Não converte metadados do Ollama em previsão.',
 ('src/models/promptcast.py', 94): 'Captura erros de validação, transporte e timeout; não captura toda '
                                   'exceção possível dentro deste método.',
 ('src/models/promptcast.py', 97): 'Relança a última exceção preservando seu tipo; o pipeline superior '
                                   "registrará status='error'.",
 ('src/models/promptcast.py', 99): 'Guarda contra término inesperado do loop; caminhos normais retornam '
                                   'lista válida ou relançam erro antes daqui.',
 ('src/evaluation/metrics.py', 6): 'Converte ambos os vetores para floats sem remover qualquer linha. '
                                   'Mesma posição deve representar a mesma data e lead.',
 ('src/evaluation/metrics.py', 11): 'Convenção: previsão menos observado. O sinal desaparece em '
                                    'MAE/RMSE, mas é útil para viés na auditoria.',
 ('src/evaluation/metrics.py', 12): 'Máscara apenas para MAPE; os zeros continuam incluídos em '
                                    'MAE/RMSE/sMAPE/R².',
 ('src/evaluation/metrics.py', 14): 'Numerador 2*abs(erro), denominador abs(real)+abs(previsto). Onde '
                                    'ambos são zero, out inicializado com zero conserva contribuição '
                                    'zero.',
 ('src/evaluation/metrics.py', 15): 'Soma dos quadrados em relação à média real; é denominador de R², '
                                    'não variância já dividida por N.',
 ('src/evaluation/metrics.py', 16): 'MAE é média do módulo; RMSE é raiz da média dos quadrados em TODOS '
                                    'os pares fornecidos. Não é média de RMSEs por janela.',
 ('src/evaluation/metrics.py', 17): 'MAPE considera apenas alvos não zero, multiplica por 100 e retorna '
                                    'None se nenhum alvo é elegível.',
 ('src/evaluation/metrics.py', 18): 'sMAPE é apresentado em porcentagem; nesta definição com fator dois '
                                    'pode chegar a 200%.',
 ('src/evaluation/metrics.py', 19): 'R² = 1 - SSE/SST. Se os alvos forem constantes, total=0 e retorna '
                                    'None. Valores negativos são possíveis e não são erro do parser.',
 ('src/evaluation/metrics.py', 20): 'Conta zeros excluídos do MAPE para não esconder redução do '
                                    'conjunto dessa métrica.',
 ('src/data/preprocessing/missing_values.py', 7): 'Seleciona TODAS as colunas exceto data e observed. '
                                                  'Isso inclui calendário; é a origem do defeito '
                                                  'identificado na auditoria de 29/02/2016.',
 ('src/data/preprocessing/missing_values.py', 8): 'Propaga passado sem vazamento futuro, mas copia '
                                                  'day/weekday/is_weekend indevidamente para dias '
                                                  'inseridos. Preencher demanda causalmente e '
                                                  'reconstruir calendário são tarefas distintas.'}

SHELL = {'experiments/start_ollama_context_server.sh': ['Escolhe Bash pelo PATH via /usr/bin/env; o sistema usa '
                                                'essa primeira linha quando executa o script '
                                                'diretamente.',
                                                '-e encerra em falhas não tratadas; -u rejeita '
                                                'variáveis inexistentes; pipefail propaga falhas de '
                                                'componentes de pipelines shell.',
                                                'export disponibiliza a variável aos processos filhos. '
                                                'HOST escolhe interface local e porta 11435; não '
                                                'publica o modelo em uma interface externa.',
                                                'Habilita a opção de Flash Attention do servidor, '
                                                'quando suportada pelo backend.',
                                                'Configura quantização q4_0 para cache de atenção K/V; '
                                                'não é a tag Q4_K_M dos pesos.',
                                                'Limita a uma sequência simultânea para reduzir memória '
                                                'e controlar a execução sequencial.',
                                                'Desabilita RAM extra de snapshots de prompts no '
                                                'backend usado; o KV da sequência ativa continua '
                                                'existindo.',
                                                'exec substitui o processo Bash por ollama serve. O '
                                                'servidor fica ativo e atende HTTP; não chama o '
                                                'pipeline Python por conta própria.'],
 'experiments/run_benchmark.sh': ['Seleciona o interpretador Bash.',
                                  'Comentário de política: reutilizar ambientes existentes sem instalar '
                                  'ou atualizar bibliotecas.',
                                  'Configura tratamento estrito de falhas e variáveis.',
                                  'Muda à raiz do projeto usando a localização do script, não a pasta '
                                  'de onde o usuário o chamou.',
                                  'Escolhe raiz dos ambientes conda por variável opcional ou padrão; '
                                  'mantém os ambientes intactos.',
                                  'Cria diretório de logs do benchmark.',
                                  'Executa Qwen no ambiente promptcast com configuração padrão '
                                  'benchmark.json e redireciona toda saída ao log. Isso usa Qwen3 1.7B, '
                                  'não o JSON do 4B.',
                                  'Solicita ao servidor Ollama padrão descarregar o modelo 1.7B para '
                                  'liberar recursos antes das baselines. Não apaga pesos em disco.',
                                  'Executa Chronos no ambiente chronos; HF_HUB_OFFLINE=1 vale para esse '
                                  'processo e impede acesso ao Hub. Log captura saída/erros.',
                                  'Executa TimesFM no ambiente timesfm. HF_HUB_OFFLINE vale para o Hub; '
                                  'XLA_PYTHON_CLIENT_PREALLOCATE=false desabilita pré-alocação XLA se '
                                  'esse backend for usado. O caminho principal do modelo aqui é '
                                  'PyTorch.',
                                  'Invoca comparador dos três modelos após todas as inferências '
                                  'terminarem com sucesso.']}

API_TEXT = ('pd.read_csv|Lê CSV em DataFrame.|path aponta o arquivo; na ingestão textual, timestamp é interpretado '
 'por date_from_text; no gráfico parse_dates converte datas ISO dos CSVs de previsões.\n'
 'pd.DataFrame|Constrói uma tabela a partir de registros ou colunas.|data posicional pode ser lista de '
 'dicts (linhas) ou dict de listas (colunas); os comprimentos devem ser compatíveis. Colunas escalares '
 'em testes são propagadas pelo índice definido pelas demais.\n'
 "pd.to_datetime|Converte valores em datas pandas.|arg é a coluna/lista de datas; errors='raise' "
 'interrompe se algo não puder ser interpretado. Não inventa datas para entradas inválidas.\n'
 "pd.to_numeric|Converte texto/objetos em números.|arg é a Series; errors='raise' rejeita texto não "
 'numérico, em vez de convertê-lo silenciosamente para NaN.\n'
 'pd.date_range|Produz uma sequência ordenada de timestamps.|start é a primeira data; periods é a '
 "quantidade, não a data final; freq='D' incrementa um dia por elemento.\n"
 'pd.Timedelta|Representa uma duração temporal.|days=1 adiciona um dia à última observação; essa '
 'expressão desloca a primeira data prevista para o dia seguinte.\n'
 'pd.DatetimeIndex|Constrói um índice temporal para comparação.|data é a coluna timestamp futura; é '
 'comparada com dates para exigir as mesmas datas e ordem.\n'
 'pd.concat|Concatena tabelas.|objs=[história, futuro] e axis padrão 0 empilham linhas; '
 'ignore_index=True renumera o índice. Aqui só concatena covariáveis, nunca alvos futuros.\n'
 'pd.testing.assert_frame_equal|Verifica igualdade detalhada entre dois DataFrames e lança '
 'AssertionError se diferirem.|left/right são tabelas com colunas de alinhamento ou covariáveis; o '
 'código usa padrões da biblioteca para checar também índices e tipos. É uma asserção, não retorna uma '
 'tabela corrigida.\n'
 'copy|Copia uma tabela/dicionário para não alterar diretamente o objeto anterior.|DataFrame.copy() usa '
 'deep=True por padrão; para objetos Python dentro de células não significa cópia recursiva de tudo. '
 'dict.copy é superficial. Não recebe argumentos nas chamadas deste projeto.\n'
 "set_index|Transforma coluna em índice do DataFrame.|keys='timestamp'; drop padrão True remove a "
 'coluna enquanto ela é índice. É necessário para asfreq temporal.\n'
 "asfreq|Reindexa a série para uma frequência temporal regular.|freq='D' cria posições diárias entre "
 'começo e fim; dias ausentes ganham NaN. Não soma nem calcula média como resample(...).mean().\n'
 "rename_axis|Nomeia o eixo do índice.|mapper='timestamp' permite que reset_index restaure uma coluna "
 'com esse nome.\n'
 'reset_index|Transforma o índice atual em coluna ou o descarta e cria índice sequencial.|drop=True '
 'descarta o índice anterior na ordenação/comparação; sem drop, a data do índice volta a ser coluna '
 'após asfreq.\n'
 "sort_values|Ordena linhas pelos valores de colunas.|by é 'timestamp', 'rmse' ou ['origin','lead']; "
 'ascending=False coloca erros maiores primeiro. Não modifica inplace nestas chamadas: devolve um novo '
 'objeto.\n'
 'isna|Retorna máscara booleana de valores ausentes.|Sem argumentos. Em DataFrame a máscara tem o mesmo '
 'formato; em Series tem uma entrada por linha.\n'
 'notna|Retorna True onde o valor não é ausente.|Sem argumentos. Na grade diária, define observed antes '
 'de preencher water_produced, preservando quais alvos eram reais.\n'
 'any|Reduz a existência de pelo menos um True.|Series/ndarray.any() sem eixo reduz seus valores; '
 'DataFrame.any() reduz por coluna e o segundo .any() reduz as colunas. O builtin any(iterável) '
 'percorre um gerador de condições e para no primeiro True.\n'
 'all|Exige que todos os elementos sejam True.|Series/ndarray.all() reduz a máscara; builtin '
 'all(iterável) pode receber um gerador. Num conjunto vazio o resultado é True: outras verificações '
 'precisam excluir entradas vazias.\n'
 "duplicated|Marca repetições.|Em DataFrame, subset=['origin','lead'] define a chave; em Series de "
 "datas não recebe subset. keep padrão 'first' preserva a primeira ocorrência como não duplicada.\n"
 'eq|Compara igualdade elemento a elemento.|other é uma Series compatível ou escalar, como 50 no teste '
 'da redução de RMSE; retorna máscara, que .all() agrega.\n'
 'normalize|Zera o horário de cada timestamp, preservando sua data.|Series.dt.normalize() não recebe '
 'argumentos aqui. A comparação com a coluna original rejeita horários diferentes de meia-noite.\n'
 'to_numpy|Converte Series/DataFrame em array NumPy.|dtype=float força formato numérico quando '
 'fornecido; sem dtype conserva tipo inferido. Não ordena linhas e não corrige alinhamento.\n'
 'astype|Converte o tipo dos valores.|dtype=str transforma categorias em texto para TimesFM. Não faz '
 'one-hot encoding; a biblioteca recebe as categorias para processar.\n'
 'tolist|Converte array/índice/Series em lista Python.|Sem argumentos. É usado para serialização e para '
 'compatibilizar o retorno dos modelos com o pipeline.\n'
 'ffill|Propaga o último valor válido anterior para lacunas subsequentes.|Sem argumentos. Não usa uma '
 'observação futura, mas não reconstrói calendário: copiar is_weekend/day para uma nova data é '
 'incorreto.\n'
 'to_csv|Serializa DataFrame como CSV.|path_or_buf ausente retorna string; um Path grava arquivo. '
 "index=False evita coluna extra de índice; float_format='%.6g' usa seis algarismos significativos. Não "
 'muda o DataFrame original.\n'
 'map|Aplica uma função a cada valor da Series.|Na conversão aplica date_to_text, number_to_words ou '
 'uma função de indicadores; na ingestão textual aplica date_from_text somente a timestamp.\n'
 'strftime|Formata datas como texto.|%Y-%m-%d identifica origens e datas nos artefatos; o corpo '
 'narrativo usa date_to_text para datas em inglês.\n'
 'date|Extrai a parte civil da data de um Timestamp.|Sem argumentos. O retorno é date, que a f-string '
 'representa como YYYY-MM-DD, sem horário.\n'
 "groupby|Cria grupos sem calcular métricas por si só.|by='lead' separa antecedências; by='origin' "
 'separa janelas; sort=False preserva ordem encontrada na auditoria. É necessário iterar, apply ou '
 'agregar depois.\n'
 'apply|Aplica uma função por grupo/objeto.|No grupo de leads, func=lambda x: '
 'sorted(x)==list(range(1,8)) exige uma ocorrência de cada lead; .all() exige isso em toda janela.\n'
 "drop_duplicates|Retém uma linha por chave repetida.|subset='origin' retém o primeiro registro de cada "
 'janela. Serve para não somar sete vezes a duração, repetida nas sete linhas do CSV.\n'
 'nunique|Conta valores distintos.|Sem argumentos. history_start.nunique()==1 verifica que todas as '
 'janelas começam na mesma data; origin.nunique() conta janelas.\n'
 'iterrows|Itera pares (índice, Series da linha).|Sem argumentos. O pipeline ignora o índice com _ e '
 'usa observation.timestamp/water_produced para registrar cada lead.\n'
 'query|Filtra linhas por expressão pandas.|Expressões sobre dataset, slug e context selecionam '
 'métricas da figura. @city, @slug e @days referem-se às variáveis do escopo.\n'
 'head|Seleciona as primeiras n linhas.|n=5 na auditoria limita as cinco piores janelas de cada grupo '
 'após ordenação.\n'
 'tail|Seleciona as últimas n linhas.|n=len(h) reconstrói o histórico restrito; n=7 calcula a média dos '
 'sete dias recentes. Não inclui futuro, pois o objeto já é história.\n'
 'np.asarray|Representa dados como ndarray, evitando cópia quando possível.|a é o vetor/lista recebido; '
 'dtype=float converte para ponto flutuante. Não valida finitude sozinho.\n'
 'np.arange|Gera progressão de valores com fim exclusivo.|stop único produz 0...stop-1; start,stop '
 'definem origens válidas. dtype=float nos testes força floats; passo padrão é 1.\n'
 'np.isfinite|Testa finitude elemento a elemento.|x é array numérico; retorna False para NaN e '
 'infinitos. .all() transforma em condição única.\n'
 'np.mean|Calcula média aritmética.|a é o vetor; axis omitido considera todos os elementos. Em métricas '
 'deve receber erros já alinhados, nunca listas filtradas separadamente.\n'
 'np.std|Calcula desvio-padrão.|a é janela histórica; ddof padrão 0. É usado para padronizar forma na '
 'seleção por similaridade, não para transformar o alvo de teste.\n'
 "np.argsort|Retorna índices que ordenariam os valores.|a=distances; kind='stable' preserva ordem de "
 'empates. [:count] seleciona os exemplos mais semelhantes.\n'
 'np.random.default_rng|Cria um gerador pseudoaleatório local.|seed=42 torna a seleção de demonstrações '
 'repetível sem depender do estado global de np.random.\n'
 'choice|Sorteia elementos do gerador NumPy.|a=origins, size=count e replace=False selecionam origens '
 'diferentes; não sorteia do período de teste.\n'
 'np.abs|Módulo elemento a elemento.|x pode ser erro ou alvo; remove o sinal para MAE/MAPE/sMAPE, mas o '
 'viés auditado mantém o sinal.\n'
 'np.sqrt|Raiz quadrada elemento a elemento ou escalar.|x é MSE; produz RMSE nas mesmas unidades do '
 'alvo.\n'
 'np.sum|Soma elementos.|a é erro ao quadrado ou desvios do alvo; axis omitido soma tudo. Resultados '
 'NumPy são convertidos por float para JSON.\n'
 'np.divide|Divide vetores com máscara e destino explícitos.|x1=2*abs(error), x2=denom, '
 'out=zeros_like(error), where=denom!=0. Posições de denominador zero mantêm zero; evita NaN no sMAPE '
 'quando real e previsto são zero.\n'
 'np.zeros_like|Cria zeros com formato e tipo semelhantes ao array.|a=error fornece o formato do '
 'destino de divide. Não altera error.\n'
 'np.dot|Produto interno dos vetores.|a=err,b=err soma quadrados independentemente de '
 'deterministic_metrics; divisão por N e sqrt reconstroem RMSE.\n'
 'np.diff|Diferenças entre elementos consecutivos.|a=pred; n padrão 1. Sete previsões geram seis '
 'incrementos para detectar rampas.\n'
 'np.array_equal|Compara exatamente valores e formas de arrays.|a/b são alvos CSV versus dataset, ou '
 'previsão CSV versus resposta. Retorna bool e não aplica tolerância.\n'
 'np.isclose|Compara com tolerância numérica.|a/b são valores; rtol=1e-12 é tolerância relativa na '
 'auditoria do RMSE; atol padrão permanece se não informado.\n'
 'np.allclose|Verifica proximidade numérica elemento a elemento.|As componentes fornecidas são '
 'comparadas a sin/cos dos índices recuperados; atol=1e-6 tolera precisão da representação.\n'
 'np.testing.assert_allclose|Asserção numérica vetorial tolerante.|actual e desired são resultados e '
 'referência; usa tolerâncias padrão se não especificadas. Lança AssertionError e é usada para testar '
 'métricas e fidelidade do prompt.\n'
 'Path|Representa caminho de arquivo com operações portáveis.|path posicional é string ou Path. O '
 'operador / combina diretórios, não faz divisão numérica nesse contexto.\n'
 'resolve|Resolve caminho absoluto e links simbólicos.|Sem argumentos aqui. '
 'Path(__file__).resolve().parents[2] encontra a raiz a partir do módulo.\n'
 "rglob|Percorre recursivamente caminhos compatíveis com padrão.|pattern='*.py' encontra fontes dentro "
 'de src; sorted fixa a ordem antes de compor o hash.\n'
 'mkdir|Cria diretório.|parents=True cria pais ausentes; exist_ok=True aceita diretório já existente. '
 'Sem exist_ok, colisão pode lançar FileExistsError.\n'
 'read_text|Lê todo o arquivo como string.|Sem encoding explícito, usa a configuração padrão do '
 'ambiente. JSON e prompts esperam texto compatível; não retorna dict sem json.loads.\n'
 'read_bytes|Lê bytes integrais do arquivo.|Sem argumentos. O hash do dataset usa bytes originais, '
 'incluindo quebras de linha.\n'
 'write_text|Grava uma string em arquivo e retorna a quantidade de caracteres escrita.|data é a string; '
 'não cria pais automaticamente. Nestes usos, encoding não é explícito. Substitui conteúdo anterior.\n'
 "with_suffix|Produz outro Path trocando a extensão.|suffix='.tmp' transforma run.json em run.tmp, "
 'usado antes da substituição atômica.\n'
 'replace|Substitui o destino pelo arquivo representado pelo Path.|target=path final. No write_json, '
 'fonte e destino estão no mesmo diretório; reduz chance de deixar JSON parcial se o processo cair. Não '
 'é bloqueio contra dois escritores simultâneos.\n'
 'exists|Verifica existência do caminho.|Sem argumentos. Define se vale tentar ler cache já gravado.\n'
 'json.loads|Decodifica uma string JSON em objetos Python.|s=text; JSON null vira None, true vira True, '
 'arrays viram listas. Pode aceitar NaN no parser padrão, por isso parse_prediction testa math.isfinite '
 'depois.\n'
 'json.load|Decodifica JSON lido de um arquivo/response.|fp é o fluxo retornado por urlopen. Diferente '
 'de loads, não recebe diretamente a string do modelo.\n'
 'json.dumps|Serializa objeto Python em string JSON.|obj=value; sort_keys=True estabiliza hashes; '
 'ensure_ascii=False preserva acentos; indent=2 melhora leitura; allow_nan=False rejeita NaN/inf na '
 'persistência. Nem todos esses argumentos aparecem em toda chamada.\n'
 'hashlib.sha256|Cria calculador de digest SHA-256.|data opcional são bytes, não texto. Na versão '
 'incremental inicia vazio e recebe update; no hash de configuração recebe JSON codificado.\n'
 'update|Acrescenta conteúdo ao objeto mutável.|Em hash.update(data), data são bytes incorporados ao '
 'digest; em dict.update(...), argumentos nomeados adicionam/substituem campos como status e audit. O '
 'receptor distingue as duas APIs.\n'
 'hexdigest|Retorna a representação hexadecimal do digest acumulado.|Sem argumentos. É texto para '
 'comparar identidades; não criptografa conteúdo para confidencialidade.\n'
 "encode|Codifica string em bytes.|encoding='utf-8' no HTTP e orçamento do prompt; sem argumento, "
 'str.encode usa UTF-8. Bytes são usados por SHA-256 e corpo HTTP.\n'
 'urllib.request.Request|Monta uma requisição HTTP.|url é endpoint; '
 "data=json.dumps(payload).encode('utf-8') é corpo em bytes e induz POST; headers declara Content-Type "
 'application/json. Não envia a rede até urlopen.\n'
 'urllib.request.urlopen|Abre conexão HTTP e devolve resposta como fluxo.|url ou Request é primeiro '
 'argumento; timeout em segundos limita operações de espera da conexão. with fecha a resposta. GET '
 'ocorre quando recebe URL sem corpo, como /api/tags.\n'
 'math.isfinite|Confere se um único número é finito.|x=v; combina-se com checagem de tipo e sinal para '
 'rejeitar NaN, inf e valores negativos.\n'
 'time.sleep|Suspende o processo por uma duração.|secs=min(2**attempt,4) aplica espera de 1,2,4... '
 'segundos entre tentativas, limitada a quatro.\n'
 'time.monotonic|Relógio crescente para medir duração.|Sem argumentos. A diferença final-inicial mede '
 'segundos transcorridos, sem depender da data civil.\n'
 'asdict|Converte dataclass em dicionário de seus campos.|obj=config ou window. É usado na procedência '
 'e request_id; não converte uma classe qualquer sem dataclass.\n'
 'dataclass|Decora a classe para gerar métodos a partir dos campos anotados.|frozen=True impede '
 'reatribuição comum após construção. Não executa treinamento, nem valida sozinho os tipos anotados.\n'
 'argparse.ArgumentParser|Cria o parser da interface de linha de comando.|description=__doc__ utiliza a '
 'docstring do módulo como texto de ajuda; parse_args vai interpretar os argumentos depois.\n'
 "add_argument|Registra uma opção do CLI.|Primeiro argumento é '--model' etc.; choices restringe "
 'valores; default define ausência; type=Path/int converte; help descreve a opção no --help. type=Path '
 'não verifica se o arquivo existe.\n'
 'parse_args|Interpreta argumentos do processo e retorna Namespace.|Sem args explícitos, lê '
 'sys.argv[1:]. main acessa args.model/config/dataset/output/limit.\n'
 'error|Em ArgumentParser, mostra mensagem e encerra com erro de CLI.|message é o motivo, por exemplo '
 'limite negativo. Não é o erro numérico de uma previsão.\n'
 'importlib.metadata.version|Lê versão instalada de uma distribuição Python.|distribution_name é '
 "'chronos-forecasting' ou 'timesfm'; não é necessariamente o mesmo nome do import. Pode falhar se a "
 'distribuição não estiver instalada.\n'
 'torch.manual_seed|Define semente do gerador PyTorch.|seed=42. Ajuda reprodutibilidade, mas não '
 'habilita sozinho todos os modos determinísticos de GPU.\n'
 'torch.cuda.is_available|Consulta disponibilidade do backend CUDA.|Sem argumentos. Seleciona '
 "device_map='cuda' ou 'cpu' para Chronos; não instala drivers.\n"
 'torch.set_float32_matmul_precision|Configura precisão permitida para multiplicações de matrizes '
 "float32.|precision='high' escolhe política do backend para acelerar sob tolerância; não muda a dtype "
 'declarada de todos os tensores.\n'
 'from_pretrained|Carrega pesos/configuração de um checkpoint.|Primeiro argumento é ID do modelo; '
 'local_files_only=True exige cache local. Em Chronos, device_map seleciona dispositivo e '
 'torch_dtype=torch.float32 força pesos nessa precisão. Em TimesFM usa a classe própria '
 'TimesFM_2p5_200M_torch.\n'
 'predict_df|Chronos: prevê uma tabela temporal com covariáveis.|frame é histórico; future_df é tabela '
 "futura sem alvo; prediction_length=horizon; freq='D'; batch_size=1; id_column, timestamp_column e "
 'target identificam colunas; quantile_levels=[0.1,0.5,0.9]; context_length=len(frame) pede todo o '
 "histórico. Retorno é DataFrame; coluna '0.5' é mediana.\n"
 'timesfm.ForecastConfig|Configura os limites e opções de inferência do TimesFM.|max_context arredonda '
 'histórico para múltiplo de 32; max_horizon arredonda horizonte para múltiplo de 128. normalize_inputs '
 'normaliza internamente; use_continuous_quantile_head ativa cabeça contínua; force_flip_invariance '
 'estende invariância de sinal; infer_is_positive preserva não negatividade para entrada não negativa; '
 'fix_quantile_crossing corrige cruzamentos; return_backcast disponibiliza estimativas históricas '
 'usadas no caminho com regressão.\n'
 'compile|Prepara a configuração de previsão do TimesFM.|forecast_config é ForecastConfig. O nome não '
 'significa treinamento ou alteração dos pesos. É chamado novamente por origem para capacidade '
 'compatível com história crescente.\n'
 'forecast_with_covariates|Executa regressão auxiliar e TimesFM com variáveis exógenas.|inputs é lista '
 'de séries-alvo históricas; dynamic_numerical_covariates e dynamic_categorical_covariates são dicts '
 "com listas por série cobrindo passado+futuro; xreg_mode='xreg + timesfm' combina regressão e previsão "
 'residual. Retorna pontos e saídas adicionais; o segundo retorno é ignorado aqui.\n'
 'object.__new__|Cria instância sem executar seu __init__.|cls é QwenForecaster ou ChronosForecaster. '
 'Exclusivo dos testes para evitar servidor/download/GPU; é preciso preencher atributos manualmente '
 'depois.\n'
 'Mock|Cria objeto simulado para registrar chamadas.|Sem argumentos neste teste. '
 'pipeline.predict_df.return_value é atribuído ao DataFrame artificial que predict receberá.\n'
 'patch|Substitui temporariamente uma referência durante o bloco with.|target é nome completo no módulo '
 'consumidor; side_effect=answers entrega respostas sequenciais; patch(time.sleep) evita esperas nos '
 'testes. Não altera permanentemente o arquivo-fonte.\n'
 'tempfile.TemporaryDirectory|Cria diretório temporário gerenciado.|Sem argumentos; with remove ao '
 'sair. Permite simular resultados sem sobrescrever experimentos reais.\n'
 'contextlib.redirect_stdout|Redireciona print temporariamente.|new_target=io.StringIO() captura saída '
 'em memória, para testes não poluírem o terminal.\n'
 'io.StringIO|Buffer de texto em memória com interface de arquivo.|initial_value=table permite '
 'pd.read_csv sem arquivo físico; vazio serve para capturar prints.\n'
 'assertEqual|Asserção de igualdade do unittest.|first/second são valores comparados; testa contratos '
 'discretos, como tamanho, datas ou listas de previsões.\n'
 'assertTrue|Asserção de condição verdadeira.|expr é uma condição já calculada; falha se falsa. Não '
 'corrige a condição.\n'
 'assertFalse|Asserção de condição falsa.|expr verifica, por exemplo, que um dia imputado não está '
 'marcado como observado.\n'
 'assertLess|Exige first < second.|Compara a maior data histórica com a menor data de teste para '
 'detectar sobreposição.\n'
 'assertGreaterEqual|Exige first >= second.|Usada para origem nunca anteceder o contexto inicial.\n'
 'assertIn|Exige item presente em container.|Verifica datas ou chave de erro na auditoria de uma '
 'retentativa.\n'
 'assertNotIn|Exige item ausente de container.|Usada para coluna-alvo não estar em future, e sentinela '
 'futura não estar no prompt.\n'
 'assertAlmostEqual|Compara números com aproximação decimal.|first/second são métricas e valores '
 'esperados; tolerância padrão do unittest se não informada.\n'
 'assertIsNone|Exige que o resultado seja o objeto None.|Valida redução percentual indefinida para '
 'erro-base zero.\n'
 'assertRaises|Context manager que exige uma exceção.|expected_exception é classe ou tupla de classes; '
 'o trecho dentro de with precisa falhar de modo esperado.\n'
 'assertRaisesRegex|Exige exceção e mensagem compatível com expressão regular.|expected_exception e '
 'expected_regex distinguem a razão de falha, como mudança no modelo da comparação.\n'
 'subTest|Identifica um caso dentro de um loop de teste.|text=text rotula a resposta inválida corrente '
 'para diagnóstico individual.\n'
 'unittest.main|Executa testes quando o arquivo é ponto de entrada.|Sem argumentos no código; discover '
 'também pode localizar as classes sem executar esse bloco.\n'
 'len|Conta elementos.|Objeto pode ser histórico, previsões ou texto/bytes. len(prompt) não é contagem '
 'de tokens; len(history) conta linhas.\n'
 'range|Gera inteiros com limite final exclusivo.|start, stop, step definem origens; range(retries+1) '
 'inclui tentativa inicial e retentativas; range(1,8) gera sete leads.\n'
 'slice|Representa intervalo, não materializa dados.|start é inclusivo e stop exclusivo; '
 'df.iloc[slice(...)] efetua a seleção depois.\n'
 'float|Converte escalar para float Python.|x é valor numérico ou NumPy. Facilita serialização em JSON; '
 'não normaliza nem muda a unidade.\n'
 'int|Converte para inteiro, truncando parte fracionária numérica.|x é contexto, tamanho da validação '
 'ou contagem; int(training_size*fraction) arredonda para baixo para positivos.\n'
 'str|Produz representação textual.|x pode ser exceção, timestamp ou caminho. Usado em logs/hashes; não '
 'faz validação de conteúdo.\n'
 'list|Materializa um iterável em lista.|iterável pode ser colunas, range ou tupla; pode também '
 'representar tipo em isinstance.\n'
 'tuple|Materializa sequência imutável.|iterável é histórico/alvo de demonstração. A imutabilidade da '
 'tupla ajuda o contrato da dataclass.\n'
 'set|Cria conjunto de elementos únicos.|iterável são colunas/chaves. Subtração detecta ausentes, '
 'interseção detecta nomes proibidos; não preserva ordem como contrato.\n'
 'dict|Cria dicionário.|Pode receber mapeamento posicional, campos nomeados e **outro_dict. '
 'dict(config) é cópia superficial; pop remove só a chave nessa cópia.\n'
 'sorted|Devolve uma lista ordenada.|iterável são caminhos, origens ou leads; key/reverse alteram '
 'critério/ordem quando fornecidos. Estabilidade ajuda hash e ordem de demonstrações.\n'
 'enumerate|Percorre iterável com contador.|iterable e start=1 nos leads/janelas; o contador é separado '
 'do índice temporal do dataset.\n'
 'zip|Emparelha iteráveis pela posição.|truth.iterrows() e previsões precisam da mesma ordem. zip '
 'padrão para no menor comprimento, então tamanho é validado antes.\n'
 'min|Retorna menor valor.|Recebe limites para validar positividade, tempos de espera ou timestamps. '
 'Método Series.min opera sobre a coluna.\n'
 'max|Retorna maior valor.|max(context_start,origin-history_days) impede começo negativo; max(std,1e-8) '
 'evita divisão por zero; método de Series reduz valores.\n'
 'sum|Soma elementos de iterável.|No builtin, conta condições True como 1; no método de Series soma '
 'duração ou máscara. Não calcula média por si só.\n'
 'bool|Converte condição escalar em booleano Python.|Usado para serializar resultados de reduções '
 'NumPy; não usar em array com vários elementos sem .all/.any.\n'
 'print|Escreve representação textual na saída padrão.|flush=True força envio imediato ao log de '
 'progresso; index=False pertence a to_string, não a print.\n'
 'isinstance|Verifica se um objeto é instância de um tipo ou tupla de tipos.|parsed,dict e values,list '
 'validam formato do JSON. Para números o código prefere type(v) para excluir bool.\n'
 'type|Obtém a classe concreta do valor.|type(v) not in (int,float) rejeita bool apesar de herdar de '
 'int; type(exc).__name__ registra nome da exceção.\n'
 'append|Adiciona um elemento ao final da lista.|item pode ser dicionário de previsão, auditoria ou '
 'bloco textual. Modifica a lista e retorna None.\n'
 'extend|Adiciona cada elemento de um iterável à lista.|iterable é lista de blocos de prompt. Diferente '
 'de append, não cria uma lista aninhada.\n'
 "get|Busca chave de dicionário com padrão se ausente.|key é done/response/status; default é 0 ou '' "
 'quando informado. Evita KeyError, mas padrão 0 nos contadores não prova que servidor os forneceu.\n'
 'pop|Remove uma chave de dicionário e devolve seu valor.|key identifica a entrada; default opcional '
 'evita KeyError. Não representa uma transformação dos dados temporais.\n'
 'items|Percorre pares (chave, valor) do dicionário.|Sem argumentos. main obtém name,dataset da '
 'configuração.\n'
 'values|Percorre valores do dicionário.|Sem argumentos. Comparador percorre procedências; testes '
 'percorrem datasets/covariáveis.\n'
 'issubset|Verifica se todos os elementos estão contidos em outro conjunto/iterável.|other são colunas '
 'do DataFrame; valida presença do contrato sem exigir que não existam colunas extras.\n'
 "join|Une textos com o separador que recebe o método.|'\\\\n'.join(parts) organiza prompt; "
 "','.join(...) organiza datas/valores; '|'.join(...) registra nomes de covariáveis. Todos os elementos "
 'devem ser strings.\n'
 'strip|Remove espaços e quebras nas bordas do texto.|Sem argumentos. Tira a quebra final de to_csv '
 'para manter o layout previsível, sem alterar os campos internos.\n'
 'to_string|Formata tabela como texto para terminal/log.|index=False omite índice auxiliar, conservando '
 'colunas. Não grava CSV.\n'
 'startswith|Testa prefixo textual exato.|Os argumentos concretos usados nesta edição são descritos '
 'junto a cada chamada na listagem do arquivo. Consulte o tipo do objeto receptor para distinguir '
 'métodos homônimos.\n'
 "split|Divide uma string por separador.|sep='\\\\nEND HISTORY:' ou marcador CSV identifica segmentos "
 'do prompt no teste; a indexação [0]/[1] escolhe trecho esperado.\n'
 'mean|Calcula média dos valores do objeto receptor.|Sem argumentos nas Series de duração/demanda e '
 'arrays de erro; o receptor já foi filtrado para a janela ou subconjunto correto.\n'
 'std|Desvio-padrão do receptor quando usado como método.|Neste projeto a padronização principal usa '
 'np.std; não confundir com pandas.std, cujo ddof padrão pode diferir.\n'
 'ValueError|Cria exceção para valor/formato inválido.|A string explica a violação: datas, dimensões, '
 'covariáveis, JSON ou contexto. Só é lançada quando precedida por raise ou usada por teste de '
 'exceção.\n'
 'RuntimeError|Cria exceção para estado de execução inesperado.|Mensagem sinaliza caminho que deveria '
 'ser inalcançável depois das tentativas.\n'
 'SystemExit|Solicita encerramento do processo Python.|code=0 indica sucesso; 1 sinaliza falhas para '
 'scripts shell. Usada com raise no CLI.\n'
 'ForecastWindow|Constrói dataclass de uma origem temporal.|context_start=0, origin=posição do primeiro '
 'alvo, horizon=sete dias. history/target são properties que devolvem slices.\n'
 'Demonstration|Constrói dataclass de exemplo few-shot.|history e target são tuplas de números; '
 'target_end é posição final exclusiva dentro do prefixo autorizado.\n'
 'OllamaConfig|Constrói configuração tipada do cliente Qwen.|Campos do JSON qwen e seed são argumentos '
 'nomeados. Os defaults legados só valem para campos omitidos; __post_init__ valida limites.\n'
 'QwenForecaster|Constrói adaptador Qwen e consulta servidor local.|config é OllamaConfig. Valida '
 'modelo/contexto e cria identity; não chama geração até predict.\n'
 'ChronosForecaster|Constrói adaptador Chronos2 com pesos locais.|context é contexto inicial '
 'registrado; dataset fornece nomes de covariáveis. predict depois recebe história crescente de '
 'verdade.\n'
 'TimesFMForecaster|Constrói adaptador TimesFM2.5 local.|context é contexto inicial registrado; dataset '
 'define covariáveis. Previsão usa XReg + TimesFM.\n'
 'check|Registra uma checagem da auditoria sem interromper o laço.|condition vira bool; name identifica '
 'teste; **where fornece cidade, política e origem. Lista checks preserva evidência de sucesso/falha.\n'
 'predict|Interface comum dos adaptadores do projeto.|history é tabela histórica; horizon é quantidade '
 'prevista; future_covariates é tabela sem alvo; prompt é texto já construído para Qwen. Retorna lista '
 'de floats; cada classe escolhe backend.\n'
 'Path.cwd|Obtém diretório atual.|Sem argumentos; pode diferir de ROOT, derivado da localização do '
 'módulo.\n'
 'equals|Compara conteúdo e estrutura do receptor com outro objeto pandas.|other é DataFrame ou '
 'DatetimeIndex compatível. Retorna bool; datas/ordem precisam coincidir. Não alinha ou reordena '
 'automaticamente para fazer as tabelas parecerem iguais.\n'
 "drop|Retorna tabela sem as linhas/colunas indicadas.|columns='water_produced' nos testes remove o "
 'alvo do futuro; inplace padrão False mantém o original intacto.\n'
 'relative_to|Calcula caminho relativo a um ancestral de um Path.|other=ROOT; o resultado entra no hash '
 'de fonte para considerar nome/local do arquivo sem depender do prefixo absoluto da máquina.\n'
 'itertuples|Itera linhas como tuplas nomeadas.|As métricas fornecem row.model, row.context e row.rmse '
 'para rótulos/gráficos; a ordem acompanha a tabela.\n'
 're.fullmatch|Exige correspondência da string inteira com expressão regular.|O padrão possui três '
 'grupos: ano escrito, dia escrito e nome do mês; sem correspondência, retorna None.\n'
 'pd.Timestamp|Cria instante temporal pandas.|A ingestão textual fornece year, month e day inteiros; '
 'valores calendáricos inválidos lançam erro. Não usa a demanda para descobrir a data.\n'
 'np.arctan2|Calcula ângulo a partir de seno e cosseno.|Primeiro argumento é seno, segundo cosseno; '
 'preserva o quadrante e permite recuperar a posição no ciclo.\n'
 'np.mod|Calcula resto elemento a elemento.|O divisor 2*pi normaliza ângulos para um ciclo completo; '
 'não ajusta parâmetros a partir de teste.\n'
 'np.rint|Arredonda valores ao inteiro mais próximo.|A posição angular multiplicada pelo número de '
 'categorias produz índices de dia/estação.\n'
 'np.sin|Calcula seno em radianos.|Recebe ângulos do ciclo reconstruído para validar a categoria.\n'
 'np.cos|Calcula cosseno em radianos.|Recebe ângulos do ciclo reconstruído para validar a categoria.\n'
 'divmod|Devolve quociente inteiro e resto.|Usado com bases 10, 100 ou 1000 na escrita de números; '
 'preserva o valor original.\n'
 'plt.subplots|Cria figura e eixos Matplotlib.|2,1 cria dois painéis; sharex=True compartilha datas; '
 'figsize define polegadas; constrained_layout ajusta margens.\n'
 'plot|Desenha sequência de pontos/linhas no eixo.|x recebe timestamps e y recebe y_true ou y_pred; cor '
 'distingue observado e previsto; nenhum filtro de erro é aplicado.\n'
 'bar|Desenha barras de comparação.|Rótulos identificam checkpoint/contexto e alturas são RMSEs '
 'calculados.\n'
 'bar_label|Escreve valores sobre as barras.|labels contém RMSE formatado; padding afasta texto da '
 'barra.\n'
 'savefig|Exporta uma figura para arquivo.|Path define destino; dpi=150 controla resolução dos PNGs. '
 'Não salva previsões ou altera métricas.\n'
 'plt.close|Libera recursos da figura.|fig identifica a figura concluída, evitando acumular memória '
 'entre exportações.\n'
 'mdates.DateFormatter|Formata rótulos temporais do eixo.|%m/%Y apresenta mês/ano no gráfico; não '
 'modifica timestamps dos dados.\n'
 'plt.rcParams.update|Define padrões visuais da sessão Matplotlib.|Dicionário ajusta fonte e bordas; '
 'não muda números ou seleção de pontos.\n'
 'matplotlib.use|Seleciona backend de renderização.|Agg permite exportar PNG sem janela gráfica; '
 'precisa ser definido antes de importar pyplot.\n'
 'set_title|Define título de eixo.|Texto identifica contexto e métricas; loc e fontsize controlam '
 'posição/estilo.\n'
 'set_ylabel|Define unidade/rótulo vertical.|Produção é expressa em m³; nas barras, o eixo representa '
 'RMSE em m³.\n'
 'set_ylim|Define limites verticais do eixo.|Nas barras, limite superior reserva espaço acima do maior '
 'RMSE; nenhuma barra é recortada.\n'
 'grid|Exibe linhas de referência.|alpha define transparência e axis pode limitar ao eixo y.\n'
 'legend|Mostra identificação das séries.|loc e ncol controlam posição e colunas de Real/Previsto.\n'
 'suptitle|Define título geral da figura.|Identifica cidade, checkpoint e uso do dataset textual.\n'
 'set_major_formatter|Define formatação dos rótulos principais.|DateFormatter modifica a apresentação '
 'do eixo de datas.')

# Explicit updates for the current execution profile and newly used library calls.
NOTES[('src/models/promptcast.py', 15)] = 'Padrão do adaptador: 8192 tokens. As quatro configurações qwen_text e o benchmark atual sobrescrevem para 16384; a configuração histórica de 4B/15 dias usa 131072. Isso não representa dias de histórico.'
API_TEXT += r"""
index|Localiza a posição de um valor numa sequência.|MONTHS.index recebe o nome do mês; o resultado começa em zero e soma-se um para construir Timestamp.
set_color|Define cor de uma borda do eixo.|Recebe código de cor da paleta; afeta aparência, não valores.
set_visible|Mostra ou oculta uma borda do eixo.|False remove as bordas superior/direita do gráfico.
tick_params|Ajusta estilo dos marcadores e rótulos dos eixos.|Os argumentos concretos controlam cores e tamanhos, sem mudar os pontos previstos.
endswith|Verifica sufixo de string.|A tupla ('_sin','_cos') identifica componentes cíclicos; retorna bool.
title|Converte texto para capitalização de título.|Usado para exibir o nome da cidade, sem modificar o dataset.
capitalize|Capitaliza a primeira letra do texto.|Converte o identificador de cidade em nome usado no prompt.
casefold|Normaliza caixa de texto para comparação.|Permite identificar Guaratuba independentemente das letras maiúsculas/minúsculas.
lower|Converte letras para minúsculas.|Normaliza indicadores booleanos recebidos como texto; os rótulos válidos são reconhecidos antes.
count|Conta ocorrências na sequência.|No teste, conta perguntas para exigir uma por dia do horizonte.
rename|Renomeia colunas de um DataFrame.|columns mapeia y_true para Real e y_pred para Predito, sem mudar os valores das previsões.
pd.testing.assert_series_equal|Exige igualdade de duas Series pandas.|Compara timestamps originais e recuperados do CSV textual; erro interrompe o teste.
assert_not_called|Verifica que um mock não foi chamado.|Os testes exigem rejeitar contexto Qwen inválido antes de leitura de dados ou criação do modelo.
plt.show|Solicita exibição da figura no backend ativo.|Na rotina histórica é opcional; exportação para PNG usa savefig e independe de janela gráfica.
plt.style.use|Aplica um estilo visual Matplotlib.|O estilo altera parâmetros de apresentação, não dados.
plt.tight_layout|Ajusta espaçamento entre eixos e rótulos.|Sem argumentos nas chamadas listadas; previne sobreposição na figura histórica.
"""

# Extensão editorial: comparação histórica e baselines solicitados pelo usuário.
INTRO.extend([
('Comparação entre prompt numérico e linguagem natural',
 'As previsões numéricas anteriores foram preservadas: Qwen 1.7B em experiments/benchmark e Qwen 4B em experiments/qwen_15_days. As novas avaliações estão em experiments/qwen_text. comparison_results.py exige execuções completas, igualdade de origin, timestamp, lead e y_true e o mesmo digest do checkpoint. A comparação pareada usa quinze dias; a tabela também mostra linguagem natural com sete dias, sem inventar uma execução numérica de sete dias.\n'
 'A redução percentual é 100 × (RMSE anterior − RMSE natural) / RMSE anterior. Para 4B, foi 5,16% em Joinville e 7,86% em Guaratuba. Para 1.7B, foi −522,33% e −400,04%, respectivamente: erro maior, não ganho. Previsões extremas permanecem nos resultados e nos gráficos, cujos painéis usam escalas verticais automáticas.\n'
 'Esta é uma comparação histórica descritiva. Além da redação, mudaram calendário, conteúdo turístico e a capacidade da janela de contexto num_ctx (não o limite de tokens gerados): 8192 para 16384 no 1.7B e 131072 para 16384 no 4B. Não isola causalmente o efeito da linguagem natural. Covariáveis meteorológicas futuras continuam condicionadas à disponibilidade suposta no dataset.\n'
 'experiments/prompt_comparison contém metrics.csv, prompt_change.csv, comparison_<cidade>.csv, quatro figuras Qwen e duas figuras de baselines. canva_update.json registra a apresentação atualizada: tabela clássica no slide 22, tabelas Qwen nos slides 27 e 28 e gráficos nos slides 20, 21 e 23 a 26. Esses números descrevem o estado salvo, podendo mudar se a apresentação for reorganizada.'),
('Baselines clássicos: seleção, reajuste e janela expansiva',
 'Por confirmação explícita do usuário, 355 dias de treino, 30 de validação e 30 de teste aplicam-se somente a regressão linear, ARIMA e KNN. Qwen continua com contexto fixo 7/15 e horizonte/passo 7. Chronos e TimesFM conservam o protocolo anterior. As métricas de horizonte 30 não são diretamente comparáveis às de horizonte 7.\n'
 'A divisão é cronológica, sem embaralhamento. Fold 1: treino [0,355), validação [355,385), teste [385,415). Fold 2: treino [0,385), validação [385,415), teste [415,445). Fold 3: treino [0,415), validação [415,445), teste [445,475). O limite direito é exclusivo. O passo é 30 dias e o início do treino permanece zero.\n'
 'Cada candidato ajusta somente o treino e prevê a validação inteira. O menor RMSE de validação escolhe os parâmetros; empates mantêm o primeiro candidato da grade. Depois há reajuste com treino + validação, ambos anteriores ao teste: 385 observações no primeiro fold, 415 no segundo. Por isso training_days e fit_days não são sinônimos. initial_context é 355 nos CSVs simples.\n'
 'Durante os 30 dias de teste, nenhum valor real desse bloco entra no ajuste ou nos atrasos. Ao terminar o bloco, seus 30 dados reais tornam-se passado disponível na origem seguinte; na divisão adotada, ocupam primeiro a nova validação. Isso é avaliação temporal expansiva, não validação cruzada aleatória. select_and_forecast nem aceita alvos de teste como argumento.\n'
 'Só folds completos são avaliados: 12 em Joinville e 35 em Guaratuba por modelo. A cauda de 26 dias de Guaratuba fica fora e é registrada. A preparação usa grade diária e preenchimento com passado; a construção dos folds rejeita validação/teste com demanda originalmente ausente. Foram concluídas seis execuções, 141 folds e 4230 previsões, sem falhas. O registro audit.json verifica limites e seleção das execuções salvas.'),
('Como funcionam regressão linear, KNN e ARIMA neste código',
 'Os três baselines são univariados: usam somente water_produced histórico, sem clima futuro ou calendário textual. lagged_samples cria exemplos supervisionados: com atrasos 3, [10,11,12,13] fornece X=[10,11,12] e y=13. Cada atributo precede seu alvo. A grade usa atrasos 7, 15 ou 30.\n'
 'Regressão linear ajusta uma combinação linear dos atrasos com intercepto. KNN busca exemplos históricos próximos no espaço desses atrasos; testa 3, 5 ou 10 vizinhos e pesos uniformes ou inversamente proporcionais à distância. StandardScaler e o regressor ficam na mesma Pipeline, ajustada exclusivamente nos exemplos do prefixo autorizado. São três candidatos lineares e dezoito candidatos KNN por fold.\n'
 'recursive_forecast prevê um dia, acrescenta a própria previsão ao histórico temporário e repete até completar 30 dias. Não usa teacher forcing com valores reais do horizonte. Essa recursão vale tanto para validação quanto para teste e pode acumular erro.\n'
 'ARIMA(p,d,q) combina termos autorregressivos, diferenciação e médias móveis dos erros. Não há componente sazonal SARIMA nesta implementação. A grade tem oito ordens: (1,0,0), (2,0,0), (1,0,1), (2,0,1), (1,1,0), (2,1,0), (0,1,1), (1,1,1). Média e desvio são calculados somente no prefixo de ajuste; o desvio mínimo é 1e-8. A previsão de 30 passos é transformada de volta à escala original. O ajuste permite até 200 iterações; aviso de não convergência é tratado como falha de candidato.\n'
 'Falhas ficam registradas em trials; se todos os candidatos falham, o fold falha. Valores não finitos são rejeitados. Não há substituição por previsões inventadas, recorte de negativos ou remoção de extremos nos baselines. run.json identifica configuração, hashes e limites; folds/*.json registra seleção, erros, previsões e duração. status.json aponta completude. predictions.csv contém origem, data, lead, verdade, previsão e tamanhos de treino/reajuste.\n'
 'RMSE agregado em Joinville: regressão linear 13634,16; ARIMA 14719,54; KNN 14135,82. Em Guaratuba: regressão linear 2670,04; ARIMA 2920,12; KNN 2967,14. A regressão teve o menor RMSE entre estes três candidatos de modelo no protocolo executado. Os números completos estão em experiments/classical_baselines/metrics.csv. Os testes cobrem limites dos folds, escala ajustada somente no treino, seleção/reajuste sem alvos de teste e recursão real dos regressores.')])
PURPOSE.update({
 'src/data/features/autoregressive.py':'Constrói atrasos anteriores ao alvo e previsão recursiva sem observações do horizonte.',
 'src/models/classical.py':'Adapta regressão linear, KNN e ARIMA ao contrato fit(history)/predict(horizon).',
 'src/training/classical.py':'Seleciona parâmetros pelo RMSE de validação e reajusta somente o prefixo anterior ao teste.',
 'src/pipelines/classical_baselines.py':'Executa o protocolo exclusivo 355/30/30, salva procedência, auditorias e previsões.',
 'src/visualization/comparison_results.py':'Compara os prompts Qwen históricos e gera gráficos separados dos baselines.',
 'configs/classical_baselines.json':'Define séries, protocolo e grades dos três baselines univariados.',
 'tests/test_classical_baselines.py':'Verifica isolamento temporal, seleção, escala e previsão recursiva.'})
FUNCTIONS.update({
 'lagged_samples':'values é o histórico autorizado; lags é o número de atrasos. Devolve X com lags colunas e y com o próximo valor, sem atributos posteriores ao alvo.',
 'recursive_forecast':'Recebe estimador ajustado, histórico permitido, horizonte e atrasos. Acrescenta apenas previsões durante o horizonte e devolve um vetor finito.',
 'select_and_forecast':'Recebe nome, treino, validação, horizonte e candidatos, nunca alvos de teste. Seleciona pelo RMSE de validação, reajusta treino+validação e devolve previsões e auditoria.',
 'create_classical_model':'Cria o adaptador ARIMA ou LagRegressor conforme name e os parâmetros selecionados.',
 'expanding_validation_folds':'Gera treino expansivo desde zero, validação móvel e teste completo, com avanço igual ao tamanho do teste. Rejeita alvos ausentes nos blocos pontuados.',
 'prompt_comparison':'Confere completude, mesmos alvos e checkpoint; calcula métricas e redução histórica de RMSE e gera quatro gráficos de contexto 15.',
 'baseline_plots':'Lê os seis resultados clássicos, exige mesmos alvos entre modelos e desenha três painéis por cidade.'})
CONFIG_KEYS.update({
 'initial_training':'Treino inicial de seleção: 355 dias, exclusivo dos baselines clássicos.',
 'validation_days':'Bloco cronológico de 30 dias para seleção de parâmetros.',
 'test_days':'Horizonte de teste de 30 dias, sem atualização com verdade interna.',
 'step_days':'Avanço 30, obrigatoriamente igual ao teste neste pipeline.',
 'refit_train_validation':'Exige reajuste final no prefixo treino+validação anterior ao teste.',
 'candidates':'Grade explícita selecionada exclusivamente pela validação de cada fold.',
 'lags':'Número de demandas passadas por exemplo supervisionado.',
 'neighbors':'Número de vizinhos do KNN.',
 'weights':'uniform atribui pesos iguais; distance pondera por distância.',
 'order':'Ordem (p,d,q) do ARIMA não sazonal.'})
API_TEXT += '''
ARIMA|Constrói o modelo autorregressivo integrado de médias móveis.|endog é a demanda normalizada do prefixo permitido; order=(p,d,q) define autorregressão, diferenciação e médias móveis. Não configura sazonalidade.
KNeighborsRegressor|Regressor por vizinhos mais próximos.|n_neighbors escolhe quantos exemplos históricos usar; weights é uniform ou distance; n_jobs=1 limita paralelismo.
LinearRegression|Regressão linear por mínimos quadrados.|Sem argumentos explícitos; usa intercepto padrão. Ajusta coeficientes nos atrasos padronizados.
StandardScaler|Padroniza atributos por média e desvio do ajuste.|Sem argumentos explícitos; é ajustado dentro da Pipeline apenas nas linhas de atributos do histórico autorizado.
make_pipeline|Encadeia transformações e estimador.|Recebe StandardScaler e regressor em ordem; fit aprende a escala antes de ajustar o regressor e predict reutiliza essa escala.
reshape|Muda a forma de um array.|(1,-1) cria uma única linha de atributos com todos os atrasos para prever o próximo dia.
np.concatenate|Concatena arrays.|[train,valid] forma o prefixo autorizado para reajuste; nenhum alvo do teste é concatenado.
np.lib.stride_tricks.sliding_window_view|Cria janelas consecutivas sobre um array.|values é a série histórica; lags+1 inclui atributos anteriores e alvo seguinte. O código copia X e y antes de retorná-los.
np.repeat|Repete valores em um array.|Usado no estimador simulado dos testes para produzir previsões de comprimento determinado.
np.testing.assert_array_equal|Exige igualdade exata entre arrays.|Compara resultados nos testes; divergência de forma ou valores falha a asserção.
warnings.catch_warnings|Isola temporariamente a política de avisos.|O bloco with restaura a política anterior ao terminar o ajuste ARIMA.
warnings.simplefilter|Define tratamento de uma categoria de avisos.|action=error e ConvergenceWarning transformam não convergência em exceção registrada como falha.
'''
