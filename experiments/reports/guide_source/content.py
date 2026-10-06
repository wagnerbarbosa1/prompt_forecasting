"""Editorial source for the complete project guide; not imported by forecasting."""
INTRO = [
('Como usar este livro', '''Este guia documenta o código existente, não uma arquitetura idealizada. Cada arquivo executável do pipeline, cada teste, os scripts de execução, as configurações e a auditoria são percorridos com numeração de linha. Imports e separadores recebem comentários curtos; as operações que transformam dados, montam prompts, fazem HTTP ou calculam métricas recebem explicações específicas. Os parâmetros apresentados são os usados no projeto: isto não é uma reprodução de todos os parâmetros possíveis de cada biblioteca.
O percurso recomendado é: arquitetura → rastreamento de uma previsão → conceitos de Python → arquivos em ordem de execução → referências das bibliotecas → testes e auditoria. O sumário e os marcadores do PDF permitem abrir diretamente um arquivo. O HTML acompanhante permite procurar nomes com facilidade. Os números de linha e hashes pertencem ao retrato de código indicado no manifesto; se o código mudar, uma linha pode mudar de posição.
Arquivos CSV de dados e milhares de respostas JSON são artefatos, não programas: o guia explica seu esquema e exemplifica sua relação com o código, sem reproduzir todo o dataset. Arquivos vazios __init__.py são listados individualmente. Metadados gerados por instalação, .git, ambientes conda e o próprio gerador deste livro não fazem parte do pipeline de previsão.'''),
('Arquitetura: camadas, responsabilidades e padrões', '''A árvore src organiza responsabilidades: ingestion cuida de I/O; validation verifica o contrato dos dados; preprocessing constrói a grade diária e os textos; features separa covariáveis; evaluation define janelas e métricas; models adapta as bibliotecas; pipelines coordena a execução. Configurações ficam em configs; entradas em data/processed; saídas auditáveis em experiments; regressões em tests.
Esse desenho é uma arquitetura em camadas e um pipeline procedural. Não é correto chamar a árvore de MVC ou Clean Architecture completa: não há controladores web, repositórios abstratos nem injeção de dependência por framework. Há separação de responsabilidades e dependências explícitas entre módulos.
create_model funciona como uma fábrica simples: escolhe um adaptador pelo nome qwen, chronos ou timesfm. QwenForecaster, ChronosForecaster e TimesFMForecaster exercem o papel de Adapter: expõem predict, identity e last_audit, escondendo HTTP, predict_df ou forecast_with_covariates. É uma interface por convenção, ou duck typing; não existe classe abstrata obrigando esse contrato.
A escolha de strategy em select_demonstrations é uma estratégia implementada por condicionais, sem hierarquia formal de classes. As dataclasses ForecastWindow, Demonstration e OllamaConfig são objetos de dados com significado explícito. O cache por hash é um mecanismo de reprodutibilidade; não é treinamento nem cache de aprendizado do modelo.
A modularização evita que uma rotina de leitura precise conhecer Ollama, ou que a fórmula do RMSE precise abrir CSV. Em contrapartida, forecasting_pipeline importa vários serviços e conhece o esquema inteiro: é o ponto de orquestração. compare_models importa ROOT desse pipeline; há acoplamento a essa constante. Não existe um módulo de domínio independente de pandas.
api.py, database.py, cleaning.py e moirai.py foram sugeridos na árvore inicial, mas não existem como implementações ativas no retrato atual. Não inventamos chamadas para módulos inexistentes. training/train.py existe como suporte à divisão treino/validação, mas não é chamado no benchmark zero-shot.'''),
('Mapa de chamadas: de qwen_forecast ao CSV', '''pyproject.toml registra qwen_forecast = src.pipelines.forecasting_pipeline:main. O comando instalado chama main() desse módulo. Executar python -m src.pipelines.forecasting_pipeline percorre o mesmo módulo e entra no bloco if __name__ == '__main__'. Não há run_qwen.py ativo. O antigo run_qwen_forecasts.py não é o ponto de entrada atual.
experiments/run_benchmark.sh → Python do ambiente promptcast → forecasting_pipeline.main(--model qwen) → run_dataset. Em seguida o shell executa Chronos no ambiente chronos, TimesFM no ambiente timesfm e compare_models. Já run_qwen_context_comparison.sh percorre 15_days e expanding no mesmo checkpoint Qwen3 4B e termina em compare_qwen_contexts. start_ollama_context_server.sh é iniciado separadamente: ele abre o servidor local; não calcula métricas.
main → ArgumentParser/parse_args → JSON de configuração → loop dos datasets → run_dataset. run_dataset → prepare_dataset → read_series → pandas.read_csv → validate_series. Ao voltar: daily_grid → fill_past_only → build_windows. validate_covariates verifica o contrato das colunas.
run_dataset → create_model → OllamaConfig → QwenForecaster.__init__. O construtor consulta /api/tags e /api/show para confirmar instalação, digest e contexto declarado; não faz ainda uma previsão. A identidade resultante participa da procedência e dos hashes de cache.
Para cada ForecastWindow: model_inputs separa history/future; select_demonstrations retorna [] no modo zero; build_tabular_prompt ou build_prompt constrói uma string. O pipeline grava essa string e calcula request_id. Se uma resposta ok tiver o mesmo hash, reaproveita-a. Caso contrário, QwenForecaster.predict → prediction_schema → post_json → urllib.request.Request → urlopen → Ollama /api/generate.
A resposta HTTP contém metadados e uma string response. parse_prediction converte apenas essa string em lista de números, após validar chave, tipo, quantidade, finitude e sinal. O pipeline grava resposta bruta e duração. Só depois seleciona truth = df.iloc[window.target] e emparelha cada número com uma data/lead.
Ao terminar cada dataset: predictions.csv e status.json. O CLI de inferência não calcula automaticamente métricas agregadas: o shell chama um comparador depois. compare_qwen_contexts valida as duas configurações e chama deterministic_metrics; compare_models compara os três modelos do benchmark antigo. A comparação simples com seis colunas foi exportada pelo driver de retomada, que era temporário, e não é uma saída nativa do CLI geral.'''),
('Uma janela concreta, sem saltar etapas', '''Em Joinville, a primeira origem é o índice 365: history = df.iloc[0:365], alvo = df.iloc[365:372]. O intervalo da direita é exclusivo: existem 365 observações históricas e sete futuras. As datas históricas vão de 17/06/2023 a 15/06/2024; a primeira previsão é para 16/06/2024. Na origem seguinte, 372, o histórico expansivo passa a df.iloc[0:372]. O começo não muda.
Com qwen_history_days=15, a origem não muda: apenas history vira df.iloc[350:365] na primeira janela. future continua nos mesmos sete dias. Com null no JSON, json.loads produz None e o código não corta o início. context do dataset escolhe a primeira origem; qwen_history_days escolhe quanto desse passado o Qwen recebe; num_ctx é capacidade de tokens do Ollama. São três grandezas distintas.
Em Guaratuba, 730 dias de calendário levam à primeira origem em 31/12/2017. O CSV não contém 29/02/2016: daily_grid materializa o dia. O preenchimento usa passado, mas a auditoria encontrou um problema: fill_past_only também copia atributos de calendário. Esse defeito está documentado nas anotações e não foi corrigido silenciosamente para escrever o guia.
O prompt histórico contém timestamp, water_produced e covariáveis. O futuro contém timestamp e covariáveis, sem water_produced. O usuário confirmou que as covariáveis meteorológicas futuras são previsões disponíveis na origem. A validação estrutural impede demanda futura explícita; a disponibilidade temporal externa dos dados não pode ser deduzida apenas de nomes de colunas.
No walk-forward, alvos de uma janela antiga tornam-se passado observado nas seguintes. Isso é próprio da avaliação expansiva solicitada, não acesso antecipado aos sete alvos da janela atual. Não há ajuste dos pesos do Qwen/Chronos/TimesFM; TimesFM com xreg ajusta regressão auxiliar sobre histórico. A política de 20% para validação está implementada como divisão cronológica, sem busca de hiperparâmetros nesta execução.'''),
('Python aplicado: argumentos, objetos e coleções', '''Parâmetro é o nome declarado na assinatura; argumento é o valor fornecido ao chamar. Em model_inputs(df, window, dataset, *, history_days=None), os três primeiros podem ser posicionais; o asterisco isolado obriga history_days a ser nomeado. Ele não coleta argumentos. Já *args numa assinatura coletaria posicionais em uma tupla, e **kwargs coletaria nomeados num dicionário. Nenhuma dessas duas variáveis mágicas é necessária para o padrão funcionar.
O código usa **config['qwen'] na chamada de OllamaConfig: é desempacotamento de um dicionário em argumentos nomeados. {'model':'qwen3:4b...', 'num_ctx':131072} equivale a model='qwen3:4b...', num_ctx=131072. Chave desconhecida ou duplicada com seed=... gera TypeError. **score dentro de dict(...) mescla campos de métricas. Nos mocks dos testes, call.args e call.kwargs são os argumentos capturados, não variáveis do programa real.
args = parser.parse_args() é apenas uma variável que recebe argparse.Namespace: args.config e args.model são atributos; não é *args. Anotações como list[float], str | Path e -> dict descrevem intenção e ajudam ferramentas; Python não valida automaticamente todos esses tipos.
@dataclass(frozen=True) gera __init__, representação e igualdade com base nos campos, e bloqueia reatribuição usual dos campos após construir. __post_init__ executa validações adicionais após o __init__ gerado. frozen não torna recursivamente imutáveis quaisquer objetos internos; aqui os campos centrais são escalares e tuplas. asdict transforma os campos em um dicionário serializável.
@property permite escrever window.history em vez de window.history(). O método calcula e retorna slice, sem criar uma cópia de dados sozinho. self é a instância: self.config persiste entre chamadas, enquanto config local é uma referência para esse objeto. __init__ inicializa a instância. object.__new__ nos testes cria a instância sem executar o construtor que faria I/O ou carregaria pesos.
DataFrame guarda tabela com linhas e colunas; Series representa uma coluna; ndarray é um vetor NumPy. df.iloc usa posições inteiras; df.loc usa rótulos/máscaras. df.iloc[a:b] inclui a e exclui b. Já df.loc[20:, ...] num índice inteiro inclui o rótulo 20. df[['timestamp', 'water_produced']] preserva uma tabela; df['water_produced'] retorna Series. copy evita modificar involuntariamente o objeto de origem.
Compreensões como [c for c in columns if ...] constroem listas filtradas; {c: valor for c in columns} constrói dicionário. any(...) para ao encontrar verdade; all(...) exige todas as condições. Em arrays, & e | combinam máscaras elemento a elemento; and/or combinam condições Python escalares. ~ inverte máscara booleana. Comparações vetoriais não devem ser usadas como um único bool sem redução.
with fecha arquivos/conexões ao sair, inclusive com erro. try/except captura falhas previstas. raise sem argumento relança a exceção corrente; return encerra a função; continue passa à próxima janela. O pipeline captura falhas de inferência para auditá-las e segue, mas um comparador rejeita execução incompleta. time.monotonic mede intervalo sem depender do relógio civil.
Tuplas permitem retorno múltiplo: history, future = model_inputs(...). enumerate(..., 1) numera leads a partir de um. zip emparelha as linhas verdadeiras e previsões; sozinho truncaria ao menor comprimento, por isso o código valida o tamanho antes. O nome _ recebe o índice de iterrows que não será usado.
F-strings avaliam expressões entre chaves: :.2f fixa duas casas, e %.6g na serialização pandas usa seis algarismos significativos. Chaves duplicadas {{ e }} representam chaves literais dentro de f-string. '\\n'.join(parts) une blocos com quebras reais; não transmite barras e n como texto literal. JSON é formato de intercâmbio; um dict é o objeto Python em memória.'''),
('Ollama: onde cada parâmetro age', '''Ollama é um servidor local separado do processo Python. O ambiente conda fornece pandas/NumPy e o cliente HTTP padrão; o modelo não é carregado pela biblioteca transformers do cliente. base_url aponta para 127.0.0.1:11435 na comparação 4B; 11434 é o padrão antigo. O servidor precisa ter o modelo instalado. O script de previsão não baixa automaticamente o checkpoint.
model escolhe a variante instalada; temperature=0 reduz a aleatoriedade de amostragem; seed=42 fixa uma semente, mas o replay da auditoria mostrou que execução sequencial e isolada não necessariamente produzem números idênticos. num_ctx=131072 limita o contexto em tokens; não dias. num_predict=256 limita tokens gerados, não sete valores. Sete números em JSON consomem vários tokens cada.
timeout=7200 pertence à espera HTTP do cliente, em segundos; retries=1 significa até duas tentativas. keep_alive='10m' pede retenção do modelo carregado após a requisição; não é timeout. stream=False pede uma resposta JSON completa em vez de fluxo de partes. think=False é enviado para desativar o modo de raciocínio onde aplicável. format recebe JSON Schema para orientar a estrutura; parse_prediction continua necessário para verificar a resposta recebida.
truncate=False é enviado como intenção de não cortar entrada; não se deve tratar um campo enviado como prova de que toda versão do servidor o aplica. Há também checagem conservadora de tamanho antes do HTTP, inspeção dos contadores na resposta e registros de runtime na auditoria. Essa implementação não tokeniza exatamente no cliente.
OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE=q4_0, OLLAMA_NUM_PARALLEL=1 e LLAMA_ARG_CACHE_RAM=0 são variáveis do processo servidor. Não são parâmetros de pandas ou de OllamaConfig. O cache KV guarda estados da atenção; os arquivos responses/*.json guardam resultados do experimento. Desabilitar snapshots extras não desabilita toda reutilização de prefixo do modelo.
O schema exige objeto com predictions, array de sete números não negativos e nenhuma chave adicional. O envelope HTTP tem response como string: primeiro json.load lê o envelope; depois json.loads lê o JSON gerado pelo modelo. done/done_reason verificam término; prompt_eval_count/eval_count ajudam a auditar limite. O parser rejeita bool explicitamente, pois bool é subtipo de int em Python.'''),
('Artefatos, reprodutibilidade e limites atuais', '''Cada pasta <execução>/<dataset>/<modelo> guarda run.json (configuração, modelo, hashes, política temporal), responses/<origem>.json (pedido, resposta, auditoria e números), predictions.csv (uma linha por data prevista) e status.json (contagens e falhas). Os prompts ficam em <execução>/<dataset>/prompts. A previsão de uma janela é armazenada antes de gerar o CSV final do dataset.
predictions.csv relaciona origin (data de início da previsão), timestamp (data-alvo), lead (1 a 7), y_true, y_pred, initial_context (origem inicial da avaliação), context (linhas efetivamente vistas), history_start, horizon, duração e run_id. Contexto inicial da avaliação não é sempre o contexto fornecido ao Qwen: pode haver initial_context=365 e context=15.
Os hashes verificam compatibilidade, não provam ausência de leakage. source_fingerprint percorre todo src/*.py: modificar qualquer fonte invalida caches mesmo que o arquivo não participe daquela execução. Escrever documentação fora de src preserva os hashes. O hash atual não inclui a versão efetiva do servidor por janela nem todos os scripts externos; runtime no JSON é configuração declarada.
As métricas juntam todos os pares dos sete leads e só então calculam RMSE. Não calculam a média dos RMSEs por janela. R² pode ser negativo; MAPE usa apenas alvos diferentes de zero e retorna porcentagem. sMAPE trata o caso ambos zero como contribuição zero. O comparador por lead usa os mesmos dados agrupados pela antecedência.
Limitações reais: compare_models assume contexto fixo do Qwen e rotula last 15 days; não é o comparador correto para Qwen expansivo nem uma configuração de sete dias. compare_qwen_contexts foi escrito especificamente para 15_days versus expanding. Não existe configuração pronta de sete dias no retrato, embora AGENTS.md a solicite; não confunda suporte de model_inputs(history_days=7) com uma execução feita.
training_partition é suporte não utilizado na inferência. O pipeline não faz treinamento do Qwen. fill_past_only precisa de tratamento específico para calendário em dias inseridos. Dados meteorológicos foram confirmados pelo usuário como previsões disponíveis na origem. A comparação simples de seis colunas por cidade existe nos resultados da retomada, mas os comparadores permanentes ainda exportam também metadados adicionais.
Este guia é documentação, não alteração de comportamento. Não corrige o bug de calendário, não modifica prompts, não reexecuta benchmarks e não muda as métricas publicadas.''')]

PURPOSE = {
'src/data/ingestion/files.py': 'Entrada/saída de arquivos. Chamado por data_pipeline e forecasting_pipeline. read_series retorna DataFrame validado; write_json grava objetos serializáveis por substituição de arquivo temporário.',
'src/data/validation/schema.py': 'Contrato do CSV: datas diárias únicas e demanda finita, não negativa. Chamado por read_series. Retorna cópia ordenada, preservando covariáveis; erro invalida o dataset antes do modelo.',
'src/data/preprocessing/resampling.py': 'Converte datas observadas em grade diária. Chamado por prepare_dataset antes do preenchimento. Adiciona observed para distinguir alvos reais de linhas inseridas.',
'src/data/preprocessing/missing_values.py': 'Preenchimento causal. Chamado após daily_grid. Retorna tabela sem lacunas iniciais de demanda; o preenchimento indiscriminado do calendário é um defeito confirmado, não uma prática recomendada.',
'src/data/features/temporal.py': 'Geração determinística das datas-alvo a partir da última data histórica. Chamado pelos dois construtores de prompt; não consulta a demanda futura.',
'src/data/features/covariates.py': 'Contrato comum dos modelos: nomes de covariáveis, validação, separação history/future e conversão para TimesFM. model_inputs é a fronteira que exclui estruturalmente o alvo da entrada futura.',
'src/evaluation/cross_validation.py': 'Define ForecastWindow e origens walk-forward. prepare_dataset chama build_windows. validation_split é usado por training_partition e testes, não para ajustar os modelos neste benchmark.',
'src/evaluation/metrics.py': 'Função numérica sem I/O. Recebe vetores reais e previstos já alinhados e retorna dicionário de métricas. É chamada pelos comparadores, testes e auditoria.',
'src/data/preprocessing/generate_prompts.py': 'Transforma história/covariáveis em texto. forecasting_pipeline usa select_demonstrations e escolhe build_prompt (legado) ou build_tabular_prompt (comparação 4B). Não chama Ollama e não calcula métricas.',
'src/models/promptcast.py': 'Adaptador HTTP do Qwen: configuração, schema de saída, parser, cliente JSON e classe QwenForecaster. forecasting_pipeline chama predict; post_json encapsula transporte, e parse_prediction valida sem corrigir artificialmente valores.',
'src/models/chronos.py': 'Adapter de Chronos2Pipeline. Importações pesadas ficam dentro do construtor, permitindo importar o pipeline em outros ambientes. Usa covariáveis passadas/futuras e devolve o quantil 0,5 como previsão pontual.',
'src/models/timesfm.py': 'Adapter de TimesFM 2.5 com XReg. Prepara capacidade suficiente para todo o histórico a cada origem e devolve somente os primeiros horizon valores. Não corta a história ao contexto inicial.',
'src/pipelines/data_pipeline.py': 'Composição de leitura, grade diária, preenchimento e janelas. Chamado por run_dataset, testes e auditoria. Retorna o par (DataFrame, lista de ForecastWindow).',
'src/pipelines/forecasting_pipeline.py': 'Orquestrador principal e CLI. Conecta dados, fábrica de modelos, prompt, cache, inferência e gravação. Executa modelos; não produz sozinho os CSVs agregados de métricas.',
'src/pipelines/compare_models.py': 'Comparador do benchmark qwen/chronos/timesfm. Rejeita execuções incompletas e datas incompatíveis; sua lógica específica de contexto do Qwen não cobre o braço expansivo.',
'src/pipelines/compare_qwen_contexts.py': 'Comparador controlado Qwen 15 dias versus expansivo. Exige mesmos alvos, fontes, modelo e configuração exceto qwen_history_days. Produz métricas, melhorias e detalhamento por lead.',
'src/training/train.py': 'Ponto auxiliar para uma extensão de treinamento: oferece slices cronológicos treino/validação. Não é chamado pela inferência atual nem treina pesos.',
'experiments/qwen_context_comparison/audit/audit_run.py': 'Auditoria offline das respostas já salvas. Reconstrói prompts, hashes e alvos; aplica mutação sentinela nos alvos futuros; recalcula erros e produz relatórios por janela. Não altera os resultados originais nem chama Ollama.',
}
# Descriptions apply to the actual calls in this repository, not all possible APIs.
API_TEXT = r'''
pd.read_csv|Lê CSV em um DataFrame, inferindo tipos; datas só são convertidas posteriormente no validador.|Primeiro argumento: caminho Path/string ou buffer StringIO. Nesta base não há parse_dates na leitura; timestamp inicialmente pode ser texto.
pd.DataFrame|Constrói uma tabela a partir de registros ou colunas.|data posicional pode ser lista de dicts (linhas) ou dict de listas (colunas); os comprimentos devem ser compatíveis. Colunas escalares em testes são propagadas pelo índice definido pelas demais.
pd.to_datetime|Converte valores em datas pandas.|arg é a coluna/lista de datas; errors='raise' interrompe se algo não puder ser interpretado. Não inventa datas para entradas inválidas.
pd.to_numeric|Converte texto/objetos em números.|arg é a Series; errors='raise' rejeita texto não numérico, em vez de convertê-lo silenciosamente para NaN.
pd.date_range|Produz uma sequência ordenada de timestamps.|start é a primeira data; periods é a quantidade, não a data final; freq='D' incrementa um dia por elemento.
pd.Timedelta|Representa uma duração temporal.|days=1 adiciona um dia à última observação; essa expressão desloca a primeira data prevista para o dia seguinte.
pd.DatetimeIndex|Constrói um índice temporal para comparação.|data é a coluna timestamp futura; é comparada com dates para exigir as mesmas datas e ordem.
pd.concat|Concatena tabelas.|objs=[história, futuro] e axis padrão 0 empilham linhas; ignore_index=True renumera o índice. Aqui só concatena covariáveis, nunca alvos futuros.
pd.testing.assert_frame_equal|Verifica igualdade detalhada entre dois DataFrames e lança AssertionError se diferirem.|left/right são tabelas com colunas de alinhamento ou covariáveis; o código usa padrões da biblioteca para checar também índices e tipos. É uma asserção, não retorna uma tabela corrigida.
copy|Copia uma tabela/dicionário para não alterar diretamente o objeto anterior.|DataFrame.copy() usa deep=True por padrão; para objetos Python dentro de células não significa cópia recursiva de tudo. dict.copy é superficial. Não recebe argumentos nas chamadas deste projeto.
set_index|Transforma coluna em índice do DataFrame.|keys='timestamp'; drop padrão True remove a coluna enquanto ela é índice. É necessário para asfreq temporal.
asfreq|Reindexa a série para uma frequência temporal regular.|freq='D' cria posições diárias entre começo e fim; dias ausentes ganham NaN. Não soma nem calcula média como resample(...).mean().
rename_axis|Nomeia o eixo do índice.|mapper='timestamp' permite que reset_index restaure uma coluna com esse nome.
reset_index|Transforma o índice atual em coluna ou o descarta e cria índice sequencial.|drop=True descarta o índice anterior na ordenação/comparação; sem drop, a data do índice volta a ser coluna após asfreq.
sort_values|Ordena linhas pelos valores de colunas.|by é 'timestamp', 'rmse' ou ['origin','lead']; ascending=False coloca erros maiores primeiro. Não modifica inplace nestas chamadas: devolve um novo objeto.
isna|Retorna máscara booleana de valores ausentes.|Sem argumentos. Em DataFrame a máscara tem o mesmo formato; em Series tem uma entrada por linha.
notna|Retorna True onde o valor não é ausente.|Sem argumentos. Na grade diária, define observed antes de preencher water_produced, preservando quais alvos eram reais.
any|Reduz a existência de pelo menos um True.|Series/ndarray.any() sem eixo reduz seus valores; DataFrame.any() reduz por coluna e o segundo .any() reduz as colunas. O builtin any(iterável) percorre um gerador de condições e para no primeiro True.
all|Exige que todos os elementos sejam True.|Series/ndarray.all() reduz a máscara; builtin all(iterável) pode receber um gerador. Num conjunto vazio o resultado é True: outras verificações precisam excluir entradas vazias.
duplicated|Marca repetições.|Em DataFrame, subset=['origin','lead'] define a chave; em Series de datas não recebe subset. keep padrão 'first' preserva a primeira ocorrência como não duplicada.
eq|Compara igualdade elemento a elemento.|other é uma Series compatível ou escalar, como 50 no teste da redução de RMSE; retorna máscara, que .all() agrega.
normalize|Zera o horário de cada timestamp, preservando sua data.|Series.dt.normalize() não recebe argumentos aqui. A comparação com a coluna original rejeita horários diferentes de meia-noite.
to_numpy|Converte Series/DataFrame em array NumPy.|dtype=float força formato numérico quando fornecido; sem dtype conserva tipo inferido. Não ordena linhas e não corrige alinhamento.
astype|Converte o tipo dos valores.|dtype=str transforma categorias em texto para TimesFM. Não faz one-hot encoding; a biblioteca recebe as categorias para processar.
tolist|Converte array/índice/Series em lista Python.|Sem argumentos. É usado para serialização e para compatibilizar o retorno dos modelos com o pipeline.
ffill|Propaga o último valor válido anterior para lacunas subsequentes.|Sem argumentos. Não usa uma observação futura, mas não reconstrói calendário: copiar is_weekend/day para uma nova data é incorreto.
to_csv|Serializa DataFrame como CSV.|path_or_buf ausente retorna string; um Path grava arquivo. index=False evita coluna extra de índice; float_format='%.6g' usa seis algarismos significativos. Não muda o DataFrame original.
map|Aplica uma função a cada valor da Series.|arg=lambda value: f'{float(value):.2f}' converte somente a representação da demanda em texto com duas casas. A demanda original fora dessa cópia conserva precisão.
strftime|Formata datas como texto.|format='%Y-%m-%d' produz ano-mês-dia e é usado em nomes de arquivos, datas dos prompts e CSVs. Pode ser método de Timestamp ou de índice/Series.dt.
date|Extrai a parte civil da data de um Timestamp.|Sem argumentos. O retorno é date, que a f-string representa como YYYY-MM-DD, sem horário.
groupby|Cria grupos sem calcular métricas por si só.|by='lead' separa antecedências; by='origin' separa janelas; sort=False preserva ordem encontrada na auditoria. É necessário iterar, apply ou agregar depois.
apply|Aplica uma função por grupo/objeto.|No grupo de leads, func=lambda x: sorted(x)==list(range(1,8)) exige uma ocorrência de cada lead; .all() exige isso em toda janela.
drop_duplicates|Retém uma linha por chave repetida.|subset='origin' retém o primeiro registro de cada janela. Serve para não somar sete vezes a duração, repetida nas sete linhas do CSV.
nunique|Conta valores distintos.|Sem argumentos. history_start.nunique()==1 verifica que todas as janelas começam na mesma data; origin.nunique() conta janelas.
iterrows|Itera pares (índice, Series da linha).|Sem argumentos. O pipeline ignora o índice com _ e usa observation.timestamp/water_produced para registrar cada lead.
query|Filtra linhas por expressão textual pandas.|expr contém nomes de colunas; @dataset/@policy referem-se às variáveis Python do escopo. Não é consulta SQL nem eval arbitrário de dados do modelo.
head|Seleciona as primeiras n linhas.|n=5 na auditoria limita as cinco piores janelas de cada grupo após ordenação.
tail|Seleciona as últimas n linhas.|n=len(h) reconstrói o histórico restrito; n=7 calcula a média dos sete dias recentes. Não inclui futuro, pois o objeto já é história.
np.asarray|Representa dados como ndarray, evitando cópia quando possível.|a é o vetor/lista recebido; dtype=float converte para ponto flutuante. Não valida finitude sozinho.
np.arange|Gera progressão de valores com fim exclusivo.|stop único produz 0...stop-1; start,stop definem origens válidas. dtype=float nos testes força floats; passo padrão é 1.
np.isfinite|Testa finitude elemento a elemento.|x é array numérico; retorna False para NaN e infinitos. .all() transforma em condição única.
np.mean|Calcula média aritmética.|a é o vetor; axis omitido considera todos os elementos. Em métricas deve receber erros já alinhados, nunca listas filtradas separadamente.
np.std|Calcula desvio-padrão.|a é janela histórica; ddof padrão 0. É usado para padronizar forma na seleção por similaridade, não para transformar o alvo de teste.
np.argsort|Retorna índices que ordenariam os valores.|a=distances; kind='stable' preserva ordem de empates. [:count] seleciona os exemplos mais semelhantes.
np.random.default_rng|Cria um gerador pseudoaleatório local.|seed=42 torna a seleção de demonstrações repetível sem depender do estado global de np.random.
choice|Sorteia elementos do gerador NumPy.|a=origins, size=count e replace=False selecionam origens diferentes; não sorteia do período de teste.
np.abs|Módulo elemento a elemento.|x pode ser erro ou alvo; remove o sinal para MAE/MAPE/sMAPE, mas o viés auditado mantém o sinal.
np.sqrt|Raiz quadrada elemento a elemento ou escalar.|x é MSE; produz RMSE nas mesmas unidades do alvo.
np.sum|Soma elementos.|a é erro ao quadrado ou desvios do alvo; axis omitido soma tudo. Resultados NumPy são convertidos por float para JSON.
np.divide|Divide vetores com máscara e destino explícitos.|x1=2*abs(error), x2=denom, out=zeros_like(error), where=denom!=0. Posições de denominador zero mantêm zero; evita NaN no sMAPE quando real e previsto são zero.
np.zeros_like|Cria zeros com formato e tipo semelhantes ao array.|a=error fornece o formato do destino de divide. Não altera error.
np.dot|Produto interno dos vetores.|a=err,b=err soma quadrados independentemente de deterministic_metrics; divisão por N e sqrt reconstroem RMSE.
np.diff|Diferenças entre elementos consecutivos.|a=pred; n padrão 1. Sete previsões geram seis incrementos para detectar rampas.
np.array_equal|Compara exatamente valores e formas de arrays.|a/b são alvos CSV versus dataset, ou previsão CSV versus resposta. Retorna bool e não aplica tolerância.
np.isclose|Compara com tolerância numérica.|a/b são valores; rtol=1e-12 é tolerância relativa na auditoria do RMSE; atol padrão permanece se não informado.
np.allclose|Exige proximidade em todos os elementos.|a=diffs,b=diffs[0], atol=1e-6,rtol=0 detectam progressão aritmética tolerando arredondamento absoluto mínimo.
np.testing.assert_allclose|Asserção numérica vetorial tolerante.|actual e desired são resultados e referência; usa tolerâncias padrão se não especificadas. Lança AssertionError e é usada para testar métricas e fidelidade do prompt.
Path|Representa caminho de arquivo com operações portáveis.|path posicional é string ou Path. O operador / combina diretórios, não faz divisão numérica nesse contexto.
resolve|Resolve caminho absoluto e links simbólicos.|Sem argumentos aqui. Path(__file__).resolve().parents[2] encontra a raiz a partir do módulo.
rglob|Percorre recursivamente caminhos compatíveis com padrão.|pattern='*.py' encontra fontes dentro de src; sorted fixa a ordem antes de compor o hash.
mkdir|Cria diretório.|parents=True cria pais ausentes; exist_ok=True aceita diretório já existente. Sem exist_ok, colisão pode lançar FileExistsError.
read_text|Lê todo o arquivo como string.|Sem encoding explícito, usa a configuração padrão do ambiente. JSON e prompts esperam texto compatível; não retorna dict sem json.loads.
read_bytes|Lê bytes integrais do arquivo.|Sem argumentos. O hash do dataset usa bytes originais, incluindo quebras de linha.
write_text|Grava uma string em arquivo e retorna a quantidade de caracteres escrita.|data é a string; não cria pais automaticamente. Nestes usos, encoding não é explícito. Substitui conteúdo anterior.
with_suffix|Produz outro Path trocando a extensão.|suffix='.tmp' transforma run.json em run.tmp, usado antes da substituição atômica.
replace|Substitui o destino pelo arquivo representado pelo Path.|target=path final. No write_json, fonte e destino estão no mesmo diretório; reduz chance de deixar JSON parcial se o processo cair. Não é bloqueio contra dois escritores simultâneos.
exists|Verifica existência do caminho.|Sem argumentos. Define se vale tentar ler cache já gravado.
json.loads|Decodifica uma string JSON em objetos Python.|s=text; JSON null vira None, true vira True, arrays viram listas. Pode aceitar NaN no parser padrão, por isso parse_prediction testa math.isfinite depois.
json.load|Decodifica JSON lido de um arquivo/response.|fp é o fluxo retornado por urlopen. Diferente de loads, não recebe diretamente a string do modelo.
json.dumps|Serializa objeto Python em string JSON.|obj=value; sort_keys=True estabiliza hashes; ensure_ascii=False preserva acentos; indent=2 melhora leitura; allow_nan=False rejeita NaN/inf na persistência. Nem todos esses argumentos aparecem em toda chamada.
hashlib.sha256|Cria calculador de digest SHA-256.|data opcional são bytes, não texto. Na versão incremental inicia vazio e recebe update; no hash de configuração recebe JSON codificado.
update|Acrescenta conteúdo ao objeto mutável.|Em hash.update(data), data são bytes incorporados ao digest; em dict.update(...), argumentos nomeados adicionam/substituem campos como status e audit. O receptor distingue as duas APIs.
hexdigest|Retorna a representação hexadecimal do digest acumulado.|Sem argumentos. É texto para comparar identidades; não criptografa conteúdo para confidencialidade.
encode|Codifica string em bytes.|encoding='utf-8' no HTTP e orçamento do prompt; sem argumento, str.encode usa UTF-8. Bytes são usados por SHA-256 e corpo HTTP.
urllib.request.Request|Monta uma requisição HTTP.|url é endpoint; data=json.dumps(payload).encode('utf-8') é corpo em bytes e induz POST; headers declara Content-Type application/json. Não envia a rede até urlopen.
urllib.request.urlopen|Abre conexão HTTP e devolve resposta como fluxo.|url ou Request é primeiro argumento; timeout em segundos limita operações de espera da conexão. with fecha a resposta. GET ocorre quando recebe URL sem corpo, como /api/tags.
math.isfinite|Confere se um único número é finito.|x=v; combina-se com checagem de tipo e sinal para rejeitar NaN, inf e valores negativos.
time.sleep|Suspende o processo por uma duração.|secs=min(2**attempt,4) aplica espera de 1,2,4... segundos entre tentativas, limitada a quatro.
time.monotonic|Relógio crescente para medir duração.|Sem argumentos. A diferença final-inicial mede segundos transcorridos, sem depender da data civil.
asdict|Converte dataclass em dicionário de seus campos.|obj=config ou window. É usado na procedência e request_id; não converte uma classe qualquer sem dataclass.
dataclass|Decora a classe para gerar métodos a partir dos campos anotados.|frozen=True impede reatribuição comum após construção. Não executa treinamento, nem valida sozinho os tipos anotados.
argparse.ArgumentParser|Cria o parser da interface de linha de comando.|description=__doc__ utiliza a docstring do módulo como texto de ajuda; parse_args vai interpretar os argumentos depois.
add_argument|Registra uma opção do CLI.|Primeiro argumento é '--model' etc.; choices restringe valores; default define ausência; type=Path/int converte; help descreve a opção no --help. type=Path não verifica se o arquivo existe.
parse_args|Interpreta argumentos do processo e retorna Namespace.|Sem args explícitos, lê sys.argv[1:]. main acessa args.model/config/dataset/output/limit.
error|Em ArgumentParser, mostra mensagem e encerra com erro de CLI.|message é o motivo, por exemplo limite negativo. Não é o erro numérico de uma previsão.
importlib.metadata.version|Lê versão instalada de uma distribuição Python.|distribution_name é 'chronos-forecasting' ou 'timesfm'; não é necessariamente o mesmo nome do import. Pode falhar se a distribuição não estiver instalada.
torch.manual_seed|Define semente do gerador PyTorch.|seed=42. Ajuda reprodutibilidade, mas não habilita sozinho todos os modos determinísticos de GPU.
torch.cuda.is_available|Consulta disponibilidade do backend CUDA.|Sem argumentos. Seleciona device_map='cuda' ou 'cpu' para Chronos; não instala drivers.
torch.set_float32_matmul_precision|Configura precisão permitida para multiplicações de matrizes float32.|precision='high' escolhe política do backend para acelerar sob tolerância; não muda a dtype declarada de todos os tensores.
from_pretrained|Carrega pesos/configuração de um checkpoint.|Primeiro argumento é ID do modelo; local_files_only=True exige cache local. Em Chronos, device_map seleciona dispositivo e torch_dtype=torch.float32 força pesos nessa precisão. Em TimesFM usa a classe própria TimesFM_2p5_200M_torch.
predict_df|Chronos: prevê uma tabela temporal com covariáveis.|frame é histórico; future_df é tabela futura sem alvo; prediction_length=horizon; freq='D'; batch_size=1; id_column, timestamp_column e target identificam colunas; quantile_levels=[0.1,0.5,0.9]; context_length=len(frame) pede todo o histórico. Retorno é DataFrame; coluna '0.5' é mediana.
timesfm.ForecastConfig|Configura os limites e opções de inferência do TimesFM.|max_context arredonda histórico para múltiplo de 32; max_horizon arredonda horizonte para múltiplo de 128. normalize_inputs normaliza internamente; use_continuous_quantile_head ativa cabeça contínua; force_flip_invariance estende invariância de sinal; infer_is_positive preserva não negatividade para entrada não negativa; fix_quantile_crossing corrige cruzamentos; return_backcast disponibiliza estimativas históricas usadas no caminho com regressão.
compile|Prepara a configuração de previsão do TimesFM.|forecast_config é ForecastConfig. O nome não significa treinamento ou alteração dos pesos. É chamado novamente por origem para capacidade compatível com história crescente.
forecast_with_covariates|Executa regressão auxiliar e TimesFM com variáveis exógenas.|inputs é lista de séries-alvo históricas; dynamic_numerical_covariates e dynamic_categorical_covariates são dicts com listas por série cobrindo passado+futuro; xreg_mode='xreg + timesfm' combina regressão e previsão residual. Retorna pontos e saídas adicionais; o segundo retorno é ignorado aqui.
object.__new__|Cria instância sem executar seu __init__.|cls é QwenForecaster ou ChronosForecaster. Exclusivo dos testes para evitar servidor/download/GPU; é preciso preencher atributos manualmente depois.
Mock|Cria objeto simulado para registrar chamadas.|Sem argumentos neste teste. pipeline.predict_df.return_value é atribuído ao DataFrame artificial que predict receberá.
patch|Substitui temporariamente uma referência durante o bloco with.|target é nome completo no módulo consumidor; side_effect=answers entrega respostas sequenciais; patch(time.sleep) evita esperas nos testes. Não altera permanentemente o arquivo-fonte.
tempfile.TemporaryDirectory|Cria diretório temporário gerenciado.|Sem argumentos; with remove ao sair. Permite simular resultados sem sobrescrever experimentos reais.
contextlib.redirect_stdout|Redireciona print temporariamente.|new_target=io.StringIO() captura saída em memória, para testes não poluírem o terminal.
io.StringIO|Buffer de texto em memória com interface de arquivo.|initial_value=table permite pd.read_csv sem arquivo físico; vazio serve para capturar prints.
assertEqual|Asserção de igualdade do unittest.|first/second são valores comparados; testa contratos discretos, como tamanho, datas ou listas de previsões.
assertTrue|Asserção de condição verdadeira.|expr é uma condição já calculada; falha se falsa. Não corrige a condição.
assertFalse|Asserção de condição falsa.|expr verifica, por exemplo, que um dia imputado não está marcado como observado.
assertLess|Exige first < second.|Compara a maior data histórica com a menor data de teste para detectar sobreposição.
assertGreaterEqual|Exige first >= second.|Usada para origem nunca anteceder o contexto inicial.
assertIn|Exige item presente em container.|Verifica datas ou chave de erro na auditoria de uma retentativa.
assertNotIn|Exige item ausente de container.|Usada para coluna-alvo não estar em future, e sentinela futura não estar no prompt.
assertAlmostEqual|Compara números com aproximação decimal.|first/second são métricas e valores esperados; tolerância padrão do unittest se não informada.
assertIsNone|Exige que o resultado seja o objeto None.|Valida redução percentual indefinida para erro-base zero.
assertRaises|Context manager que exige uma exceção.|expected_exception é classe ou tupla de classes; o trecho dentro de with precisa falhar de modo esperado.
assertRaisesRegex|Exige exceção e mensagem compatível com expressão regular.|expected_exception e expected_regex distinguem a razão de falha, como mudança no modelo da comparação.
subTest|Identifica um caso dentro de um loop de teste.|text=text rotula a resposta inválida corrente para diagnóstico individual.
unittest.main|Executa testes quando o arquivo é ponto de entrada.|Sem argumentos no código; discover também pode localizar as classes sem executar esse bloco.
len|Conta elementos.|Objeto pode ser histórico, previsões ou texto/bytes. len(prompt) não é contagem de tokens; len(history) conta linhas.
range|Gera inteiros com limite final exclusivo.|start, stop, step definem origens; range(retries+1) inclui tentativa inicial e retentativas; range(1,8) gera sete leads.
slice|Representa intervalo, não materializa dados.|start é inclusivo e stop exclusivo; df.iloc[slice(...)] efetua a seleção depois.
float|Converte escalar para float Python.|x é valor numérico ou NumPy. Facilita serialização em JSON; não normaliza nem muda a unidade.
int|Converte para inteiro, truncando parte fracionária numérica.|x é contexto, tamanho da validação ou contagem; int(training_size*fraction) arredonda para baixo para positivos.
str|Produz representação textual.|x pode ser exceção, timestamp ou caminho. Usado em logs/hashes; não faz validação de conteúdo.
list|Materializa um iterável em lista.|iterável pode ser colunas, range ou tupla; pode também representar tipo em isinstance.
tuple|Materializa sequência imutável.|iterável é histórico/alvo de demonstração. A imutabilidade da tupla ajuda o contrato da dataclass.
set|Cria conjunto de elementos únicos.|iterável são colunas/chaves. Subtração detecta ausentes, interseção detecta nomes proibidos; não preserva ordem como contrato.
dict|Cria dicionário.|Pode receber mapeamento posicional, campos nomeados e **outro_dict. dict(config) é cópia superficial; pop remove só a chave nessa cópia.
sorted|Devolve uma lista ordenada.|iterável são caminhos, origens ou leads; key/reverse alteram critério/ordem quando fornecidos. Estabilidade ajuda hash e ordem de demonstrações.
enumerate|Percorre iterável com contador.|iterable e start=1 nos leads/janelas; o contador é separado do índice temporal do dataset.
zip|Emparelha iteráveis pela posição.|truth.iterrows() e previsões precisam da mesma ordem. zip padrão para no menor comprimento, então tamanho é validado antes.
min|Retorna menor valor.|Recebe limites para validar positividade, tempos de espera ou timestamps. Método Series.min opera sobre a coluna.
max|Retorna maior valor.|max(context_start,origin-history_days) impede começo negativo; max(std,1e-8) evita divisão por zero; método de Series reduz valores.
sum|Soma elementos de iterável.|No builtin, conta condições True como 1; no método de Series soma duração ou máscara. Não calcula média por si só.
bool|Converte condição escalar em booleano Python.|Usado para serializar resultados de reduções NumPy; não usar em array com vários elementos sem .all/.any.
print|Escreve representação textual na saída padrão.|flush=True força envio imediato ao log de progresso; index=False pertence a to_string, não a print.
isinstance|Verifica se um objeto é instância de um tipo ou tupla de tipos.|parsed,dict e values,list validam formato do JSON. Para números o código prefere type(v) para excluir bool.
type|Obtém a classe concreta do valor.|type(v) not in (int,float) rejeita bool apesar de herdar de int; type(exc).__name__ registra nome da exceção.
append|Adiciona um elemento ao final da lista.|item pode ser dicionário de previsão, auditoria ou bloco textual. Modifica a lista e retorna None.
extend|Adiciona cada elemento de um iterável à lista.|iterable é lista de blocos de prompt. Diferente de append, não cria uma lista aninhada.
get|Busca chave de dicionário com padrão se ausente.|key é done/response/status; default é 0 ou '' quando informado. Evita KeyError, mas padrão 0 nos contadores não prova que servidor os forneceu.
pop|Remove e devolve valor de chave.|key='qwen_history_days' é removida da cópia da configuração antes de comparar os outros campos. Sem default, chave ausente lança KeyError.
items|Percorre pares (chave, valor) do dicionário.|Sem argumentos. main obtém name,dataset da configuração.
values|Percorre valores do dicionário.|Sem argumentos. Comparador percorre procedências; testes percorrem datasets/covariáveis.
issubset|Verifica se todos os elementos estão contidos em outro conjunto/iterável.|other são colunas do DataFrame; valida presença do contrato sem exigir que não existam colunas extras.
join|Une textos com o separador que recebe o método.|'\\n'.join(parts) organiza prompt; ','.join(...) organiza datas/valores; '|'.join(...) registra nomes de covariáveis. Todos os elementos devem ser strings.
strip|Remove espaços e quebras nas bordas do texto.|Sem argumentos. Tira a quebra final de to_csv para manter o layout previsível, sem alterar os campos internos.
to_string|Formata tabela como texto para terminal/log.|index=False omite índice auxiliar, conservando colunas. Não grava CSV.
'''

FUNCTIONS = {
'read_series':'path: caminho CSV; retorno: DataFrame validado. Encadeia pandas.read_csv e validate_series.',
'write_json':'path: destino; value: dict serializável. Sem retorno útil. Cria pais, escreve .tmp e substitui destino.',
'validate_series':'df: tabela lida. Retorno: cópia ordenada. Rejeita esquema, datas e demanda inválidos antes de qualquer modelo.',
'daily_grid':'df: tabela com timestamp. Retorno: grade diária com observed. Dias ausentes são expostos, não removidos.',
'fill_past_only':'df: grade com possíveis NaN. Retorno: cópia preenchida com passado. Não reconstrói atributos de calendário, limitação conhecida.',
'future_dates':'last_observed: último Timestamp histórico; horizon: quantidade de dias. Retorna DatetimeIndex dos próximos dias.',
'covariate_columns':'dataset: bloco de configuração de uma cidade. Retorna nomes numéricos seguidos dos categóricos, após rejeitar duplicatas e nomes reservados.',
'validate_covariates':'df: tabela preparada; dataset: contrato de colunas. Retorna None; falha em ausentes/NaN/números não finitos.',
'model_inputs':'df: série completa; window: ForecastWindow; dataset: contrato; history_days: None para expansivo ou inteiro para últimos N dias. Retorna (history, future), excluindo alvo do futuro.',
 'timesfm_covariates':'history/future: tabelas separadas; dataset: listas numéricas/categóricas. Retorna dois dicts com listas de séries cobrindo passado e futuro, sem demanda futura.',
'build_windows':'df: tabela preparada; context: primeira origem; horizon=7: quantidade prevista; step=7: avanço entre origens. Retorna lista ForecastWindow, excluindo janelas com alvo imputado.',
'validation_split':'training_size: comprimento do prefixo anterior ao teste; fraction=0.2: parte final reservada à validação. Retorna dois slices sem embaralhar.',
'deterministic_metrics':'ytrue/yhat: vetores de mesmo formato, não vazios e finitos. Retorna MAE, RMSE, MAPE%, sMAPE%, R² e contagem de zeros excluídos do MAPE.',
'select_demonstrations':'training_values: prefixo congelado anterior ao primeiro teste; history: demanda histórica corrente; horizon: tamanho de alvo dos exemplos; strategy: zero/few_random/few_similar; count=3: quantidade; seed=42: sorteio. Retorna lista de Demonstration.',
'standardized':'x: vetor de exemplo/consulta histórica. Retorna valores centrados e divididos por desvio-padrão com piso 1e-8; apenas auxilia seleção por forma.',
'format_values':'values: números históricos. Retorna string separada por vírgulas com duas casas decimais.',
'build_prompt':'history: DataFrame histórico; horizon: dias; demonstrations: lista ou None; future_covariates: tabela sem alvo ou None. Retorna prompt legado com lista de demanda e tabelas de covariáveis.',
'build_tabular_prompt':'history: DataFrame histórico; horizon: dias; future_covariates: obrigatório e nomeado. Retorna CSV histórico unificado e tabela futura, com prefixo estável para história expansiva.',
'prediction_schema':'horizon: comprimento exigido do array. Retorna JSON Schema para objeto predictions, números não negativos e ausência de chaves extras.',
'parse_prediction':'text: string JSON do modelo; horizon: quantidade exata de valores. Retorna list[float] ou lança erro; não remove prose nem aceita resposta parcial.',
'post_json':'url: endpoint; payload: dict do pedido; timeout: segundos de espera. Retorna dict decodificado do envelope HTTP, não ainda a lista de previsões.',
'fingerprint':'value: objeto serializável. Retorna SHA-256 hexadecimal do JSON com chaves ordenadas.',
'source_fingerprint':'Sem parâmetros. Retorna digest incremental de nomes e bytes de todos os .py de src, em ordem fixa.',
'create_model':'name: qwen/chronos/timesfm; context: contexto inicial; config: configuração global; dataset: configuração da cidade. Retorna adaptador apropriado. A fábrica não prevê por si só.',
'run_dataset':'name: cidade; dataset: bloco da cidade; config: parâmetros globais; model_name: adaptador; output: raiz de resultados; limit=0: todas as janelas. Retorna número de falhas e grava artefatos.',
'prepare_dataset':'path: CSV; context: primeira origem; horizon=7 e step=7 nomeados. Retorna df e windows depois da preparação.',
'compare':'output: pasta com resultados dos três modelos/cidades. Sem retorno útil; valida e grava métricas globais e por lead.',
'relative_reduction':'short: erro de 15 dias; long: erro expansivo. Retorna 100*(short-long)/short ou None se baseline zero. Positivo é melhoria, negativo piora.',
'load_completed_run':'root: raiz de comparação Qwen; dataset: cidade; label: 15_days ou expanding. Retorna (frame ordenado, run) após validar completude e história.',
'compare_contexts':'output: pasta que contém os dois braços. Sem retorno útil. Exige comparação controlada e grava quatro CSVs.',
'main':'Sem parâmetros explícitos. Lê argumentos do processo via argparse ou executa entry point do módulo. O corpo de cada arquivo determina qual CLI será iniciado.',
 'training_partition':'first_test_origin: tamanho de prefixo de treino; validation_fraction=0.2: proporção final de validação. Retorna os slices de validation_split; não faz fit.',
'check':'condition: condição auditada; name: nome da checagem; **where: dataset, policy e origin. Acrescenta resultado à lista checks; falha não interrompe o restante da auditoria.',
}
NOTES={}
def notes(path, text):
    for row in text.strip().splitlines():
        key, value=row.split('|',1)
        lo,_,hi=key.partition('-')
        for number in range(int(lo),int(hi or lo)+1):
            NOTES[(path,number)]=value

notes('src/models/promptcast.py', '''
12|Nome padrão legado. Configurações do experimento 4B substituem esse valor ao construir OllamaConfig; alterar apenas este default não muda JSON já fornecido.
13|URL padrão legado, porta 11434. O JSON do 4B usa servidor dedicado local na porta 11435.
14|Temperatura padrão zero. É configuração de amostragem, não um normalizador dos dados; não garante igualdade bit a bit entre runtimes.
15|Capacidade padrão de 8192 tokens, sobrescrita para 131072 no experimento 4B. Não corresponde ao número de dias.
16|Máximo de tokens novos, contando pontuação e estrutura JSON, não quantidade de valores previstos.
17|Espera HTTP padrão de 600 segundos; configuração 4B sobrescreve para 7200.
18|Número de tentativas adicionais, não total; retries=2 permite três chamadas no loop.
19|Semente transmitida às opções de geração do Ollama.
21|Gancho da dataclass: executado automaticamente depois da inicialização gerada, para rejeitar limites inválidos.
27|Abre schema de objeto JSON. properties descreve a chave predictions, sem inserir previsões reais no pedido.
28|Cada item deve ser número com limite inferior zero; aqui mínimo é uma restrição de saída, não clipping posterior.
29|minItems e maxItems iguais a horizon impõem exatamente sete valores quando horizon=7.
30|required torna predictions obrigatória; additionalProperties=False proíbe outras chaves no objeto.
41|O teste type(v), diferente de isinstance(v,int), impede que true/false sejam aceitos como 1/0. O gerador rejeita também infinitos e negativos.
59|Filtra os modelos instalados por nome exato, incluindo tag/quantização. Não escolhe uma variante aproximada silenciosamente.
62|Consulta metadados do checkpoint instalado; timeout posicional de 30 segundos só vale para esta consulta.
64-65|Obtém a arquitetura declarada e compõe a chave de metadados de contexto correspondente, em vez de fixar um nome de arquitetura.
68-69|Registra modelo, digest dos pesos, contexto declarado e configuração efetivamente construída. Este dict participa do hash de cache do pipeline.
71|Mantém interface dos adaptadores: history e future_covariates são aceitos, mas este método usa o prompt já montado. horizon determina schema e parser; prompt deve ser nomeado após *.
76|Compara bytes UTF-8 do prompt + reserva de geração + 512 bytes/tokens de margem com num_ctx. É limite conservador, não tokenizer exato. Um prompt legítimo pode ser rejeitado antes do limite real.
78|Monta corpo HTTP: nome do modelo, texto completo e stream=False para resposta única. Isso não envia ainda a requisição.
79|think=False solicita geração sem modo de raciocínio; keep_alive='10m' controla retenção do modelo; truncate=False expressa intenção de não cortar; format recebe o schema. A auditoria do servidor continua necessária.
80-81|options agrupa configuração de amostragem e limites; são opções do Ollama, diferentes das opções do cliente urllib, como timeout.
82|Reinicia auditoria da janela, guardando pedido exato e lista de tentativas. Não reutiliza uma resposta da janela anterior aqui.
83|range(retries+1) inclui a tentativa inicial: com retries=1, attempt assume 0 e 1.
84-85|Cria dict mutável por tentativa e o anexa antes do HTTP. Quando response/error são adicionados depois, a lista aponta para o mesmo dict e preserva o registro.
89|done ausente/False ou done_reason='length' rejeita resposta incompleta, mesmo que haja texto parseável.
91|Soma tokens de prompt e saída para rejeitar saturação. get(...,0) tolera campos ausentes, então esta checagem sozinha não certifica contadores válidos em toda versão.
93|Extrai apenas a string response do envelope e valida sete números. Não converte metadados do Ollama em previsão.
94|Captura erros de validação, transporte e timeout; não captura toda exceção possível dentro deste método.
97|Relança a última exceção preservando seu tipo; o pipeline superior registrará status='error'.
99|Guarda contra término inesperado do loop; caminhos normais retornam lista válida ou relançam erro antes daqui.
''')
notes('src/data/preprocessing/generate_prompts.py','''
11|Tupla com o histórico de um exemplo few-shot; ... na anotação significa comprimento variável.
12|Tupla com valores de saída do exemplo histórico, anterior ao teste; não é o alvo da janela corrente.
13|Fim exclusivo do alvo do exemplo; permite verificar que o exemplo termina dentro do prefixo de treinamento.
16-18|training_values e history são posicionais; strategy, count e seed são nomeados. O retorno é lista de objetos Demonstration. history define o tamanho de cada exemplo.
26-27|Zero-shot não inclui demonstrações: retorna lista vazia antes de construir origens. Esse é o caminho efetivamente usado no experimento 4B.
32|Uma demonstração precisa de width pontos passados e horizon pontos posteriores ainda dentro do prefixo permitido. O +1 faz a última origem válida entrar no arange de fim exclusivo.
39|Centraliza e escala o vetor para comparar forma; max(...,1e-8) evita divisão por desvio zero. Não aplica normalização à previsão entregue pelo modelo.
41|Para cada origem candidata, mede MSE entre a janela histórica padronizada e a consulta padronizada. O alvo do exemplo não participa da escolha por similaridade.
43-44|Empacota histórico, saída e posição final dos exemplos escolhidos, em ordem cronológica; converte arrays para tuplas e índice para int nativo.
52-54|Prompt legado aceita demonstrações e covariáveis opcionais; tabela completa nunca é parâmetro. A responsabilidade de passar apenas história pertence ao chamador.
63|Instrução da tarefa e unidade esperada. É texto dirigido ao LLM; não converte a unidade dos números no CSV.
64|Pede inferência de nível, tendência e sazonalidade; isso orienta o modelo, mas não executa um estimador estatístico explícito.
65|Interpolação de horizon informa número exato de saídas e chave JSON. É reforçada pelo schema do adaptador e pelo parser.
66|Instrui a não explicar nem simplesmente copiar a história; não impede matematicamente uma previsão constante ou repetida.
68|O operador or substitui None por lista vazia; nenhuma iteração é feita sem demonstrações.
69-70|Adiciona um par histórico entrada/saída ao texto por demonstração. json.dumps formata o exemplo de resposta no mesmo envelope predictions.
71-77|Acrescenta metadados, lista de demanda, datas-alvo e lembrete de schema. extend acrescenta os blocos separadamente; chaves duplas no f-string representam chaves literais.
79-80|Rejeita tabela futura com coluna water_produced, mesmo que os valores sejam nulos. Defesa estrutural contra passagem acidental do alvo.
81|Confere quantidade e igualdade ordenada das datas; não basta ter sete linhas quaisquer.
87-89|Rótulos que explicam ao modelo a utilização das covariáveis e marcam a tabela histórica. Não são cálculos nem chamadas de biblioteca.
90|No layout legado esta tabela contém data e covariáveis; a demanda já foi enviada na lista Values acima. index=False impede enviar índice pandas como variável extra.
91-92|Tabela futura é serializada sem demanda; %.6g reduz apenas representação textual de floats. strip elimina a quebra final do CSV.
97|Layout tabular usado em AMBOS os braços do 4B. future_covariates é obrigatório e nomeado; não há parâmetros de demonstração aqui.
103-104|Rejeita horizonte não positivo, história vazia ou alvo na tabela futura. Esta função não verifica sozinha toda a origem externa dos dados.
111|Ordem explícita das colunas garante que water_produced apareça junto à respectiva data antes das covariáveis. A cópia protege a tabela original.
114|Converte a coluna da cópia em strings com duas casas. map aplica a lambda em cada célula. As demais colunas numéricas usam seis algarismos no to_csv.
116|Declara que o histórico enviado é completo para a entrada recebida; no braço curto significa todos os 15 dias selecionados, não todo o dataset.
117|Pede combinação de demanda, clima e calendário; não há seleção automática de variáveis ou pesos definidos pelo Python.
118-119|Reforça formato e restrições de resposta, idênticos nos dois contextos do experimento 4B.
120|Marcador literal que delimita o início do CSV histórico e também é usado no teste de preservação do prefixo.
121|Insere todas as linhas de history na string, em ordem cronológica. Não amostra nem resume valores.
122|Metadados que mudam por janela vêm DEPOIS da tabela. Isso mantém o prefixo histórico anterior igual entre janelas expansivas, favorecendo reutilização de cache KV.
123-124|Marca e serializa as sete linhas futuras de covariáveis. As datas aparecem aqui, mas os sete valores water_produced não.
125|Define explicitamente a associação por posição entre números do array e datas; o modelo não devolve timestamps individuais.
126|Lembrete final do schema; number é um marcador textual de formato, não um número que deva ser copiado na saída.
''')
notes('src/pipelines/forecasting_pipeline.py','''
15|ROOT é a raiz obtida a partir da posição deste arquivo; os caminhos de dados e padrões do CLI são ancorados nela.
19|O JSON de value recebe chaves ordenadas, vira bytes e depois SHA-256 hexadecimal. Ordem de listas permanece significativa: trocar covariáveis de posição muda o hash.
24-28|O hash de fonte inclui caminho relativo e bytes de cada .py em src, em ordem lexicográfica. Qualquer alteração de fonte pode invalidar todos os caches; relatórios fora de src não entram.
33|Importa apenas o adaptador necessário ao ramo Qwen; dependências dos outros modelos não precisam estar no ambiente promptcast.
34|**config['qwen'] expande chaves como model/base_url; seed vem do nível global. A dataclass valida limites no __post_init__.
36-39|Importações tardias isolam dependências de Chronos e TimesFM; context e dataset são passados a cada adaptador.
44|Executa um único par dataset/modelo. output é a raiz de saída; limit=0 significa execução completa, valor positivo limita as janelas para smoke test.
45|Resolve caminho do CSV relativamente à raiz do projeto, independentemente de qual nome de modelo será executado.
46-47|prepare_dataset retorna tabela preparada e todas as janelas elegíveis, com horizonte e passo da configuração comum.
49-50|Um limite positivo retém só as primeiras janelas; o valor entra na procedência para comparadores recusarem métricas parciais como completas.
52-60|Procedência reúne configuração, hashes, identidade do adaptador, datas imputadas e política de covariáveis. É compartilhada pelas janelas desse dataset/modelo.
61|run_id depende de toda a procedência; um cache de outra configuração não deve passar na checagem mesmo que o nome do arquivo coincida.
64|Comentário de segurança: demonstrações few-shot ficam no prefixo ANTERIOR ao primeiro teste, mesmo quando a história operacional cresce depois.
65|Congela valores para seleção de demonstrações. A série completa continua em df, mas esta variável não inclui alvos da primeira janela nem posteriores.
67|Percorre janelas em ordem. index é contador humano começando em 1; window.origin é posição da origem no dataset, outra grandeza.
68-69|Apenas Qwen pode receber corte history_days do JSON; None mantém expansão. Chronos e TimesFM recebem None sempre.
70|A origem é convertida em YYYY-MM-DD para identificar janela, prompt e resposta; não é data de término do horizonte.
71|Um JSON por origem dentro de responses permite retomar execução sem repetir inferências válidas.
74-75|Seleciona exemplos exclusivamente de training_values congelado; zero-shot retorna [] e ignora count na prática.
76-80|Layout tabular recusa demonstrações não vazias para manter a comparação controlada zero-shot. O legado é o outro ramo.
83-85|Persiste o prompt antes de chamar o modelo. Se a execução cair no HTTP, pode haver arquivo de prompt sem resposta válida: isso não significa janela concluída.
86|request_id combina procedência, definição exata da janela e texto completo. Até mudança de espaço no prompt invalida a resposta em cache.
87-88|Só reutiliza resposta com request_id idêntico E status ok. Um JSON de erro anterior será tentado novamente numa retomada.
90|Relógio monotônico começa antes da inferência e inclui tentativas/esperas; cache não ganha tempo zero, conserva a duração original armazenada.
91-93|Registro inicial guarda posição, limites de história e lista de covariáveis, antes de prever.
94|Chamada comum ao adaptador. Para Qwen, prompt já contém os dados; para Chronos/TimesFM, history e future_covariates alimentam suas APIs numéricas.
95-96|Exige exatamente horizon números finitos. Impede zip de truncar silenciosamente previsões curtas depois.
97|Marca sucesso somente após validar forma e finitude.
98-99|Captura exceções de inferência e registra classe/mensagem, permitindo continuar outras janelas. Não transforma erro em previsão zero.
100-101|Acrescenta duração e last_audit do adaptador e grava arquivo por substituição temporária. A resposta bruta do Ollama permanece auditável.
102-103|Progresso inclui cidade, modelo, contador, origem e status; sufixo cached indica reaproveitamento. flush=True atualiza log prontamente.
104-106|Janela com falha entra na lista failures e não recebe linhas de previsão. Comparadores rejeitam esse resultado incompleto.
108|Só depois da inferência acessa a tabela de alvos para avaliação; history/future já foram construídos antes sem essa coluna futura.
109|iterrows fornece (índice, linha); zip associa previsão na mesma posição; enumerate numera lead 1...7; _ descarta o índice pandas.
110-116|Uma linha de resultado por lead. initial_context é origem inicial da avaliação; context é o histórico efetivamente fornecido. A duração da janela repete nas sete linhas e precisa ser desduplicada ao somar.
117|Grava o CSV completo do dataset depois do loop. Durante uma execução interrompida, as respostas por janela são a fonte de retomada, não necessariamente esse CSV.
118-119|status resume janelas esperadas, concluídas e falhas. successful_windows usa divisão inteira do número de linhas pelo horizonte.
120|Retorna quantidade de falhas, permitindo main definir o código de saída do processo.
125|--model escolhe adaptador; padrão qwen. Não escolhe automaticamente o tamanho do checkpoint, que vem do JSON.
126|--config vira Path e por padrão usa benchmark.json (Qwen1.7B). Para 4B é indispensável fornecer o JSON correspondente.
127|--dataset all executa as duas cidades; uma cidade filtra o loop sem alterar seus parâmetros.
128|--output escolhe pasta de artefatos; separar experimentos evita sobrepor configurações incompatíveis.
129|--limit serve para teste rápido; 0 é todas as janelas. Não use métricas desse recorte como benchmark completo.
133|read_text retorna string; json.loads a transforma em dict. Sem esse segundo passo, acessar config['datasets'] não funcionaria.
135-137|Itera entradas de datasets, aplica filtro all/nome e acumula número de janelas que falharam em run_dataset.
138|Código 0 indica sucesso; 1 indica alguma falha. Em shells com set -e isso impede executar comparação como se todos os modelos tivessem terminado.
141-142|Guarda de entrada: executa main apenas como programa/módulo -m, não quando outro arquivo importa ROOT ou funções.
''')
notes('src/evaluation/metrics.py','''
6|Converte ambos os vetores para floats sem remover qualquer linha. Mesma posição deve representar a mesma data e lead.
11|Convenção: previsão menos observado. O sinal desaparece em MAE/RMSE, mas é útil para viés na auditoria.
12|Máscara apenas para MAPE; os zeros continuam incluídos em MAE/RMSE/sMAPE/R².
14|Numerador 2*abs(erro), denominador abs(real)+abs(previsto). Onde ambos são zero, out inicializado com zero conserva contribuição zero.
15|Soma dos quadrados em relação à média real; é denominador de R², não variância já dividida por N.
16|MAE é média do módulo; RMSE é raiz da média dos quadrados em TODOS os pares fornecidos. Não é média de RMSEs por janela.
17|MAPE considera apenas alvos não zero, multiplica por 100 e retorna None se nenhum alvo é elegível.
18|sMAPE é apresentado em porcentagem; nesta definição com fator dois pode chegar a 200%.
19|R² = 1 - SSE/SST. Se os alvos forem constantes, total=0 e retorna None. Valores negativos são possíveis e não são erro do parser.
20|Conta zeros excluídos do MAPE para não esconder redução do conjunto dessa métrica.
''')
notes('src/data/preprocessing/missing_values.py','''
7|Seleciona TODAS as colunas exceto data e observed. Isso inclui calendário; é a origem do defeito identificado na auditoria de 29/02/2016.
8|Propaga passado sem vazamento futuro, mas copia day/weekday/is_weekend indevidamente para dias inseridos. Preencher demanda causalmente e reconstruir calendário são tarefas distintas.
''')
notes('src/pipelines/compare_models.py','''
25-26|Limitação: assume que todo Qwen tem história fixa igual a qwen_history_days. Se for None para expansivo, espera uma lista de None e rejeita comprimentos reais. Use compare_qwen_contexts para o experimento expansivo existente.
41|Rótulo hardcoded last 15 days: não descreve corretamente um futuro teste Qwen com sete dias. Este guia expõe a limitação sem modificar o comparador.
''')
notes('src/pipelines/compare_qwen_contexts.py','''
27|Validação especializada nos nomes 15_days/expanding; uma terceira configuração de sete dias exigirá generalização explícita, não só mudar o nome da pasta.
58|A cópia superficial de config permite remover qwen_history_days para comparar todo o resto; não altera o dict original de run. pop exige que a chave exista.
''')

CONFIG_KEYS = {
'datasets':'Mapeamento dos dois datasets; cada bloco fornece caminho, primeira origem e contrato de covariáveis.',
'joinville':'Bloco da série de Joinville. Contexto inicial de avaliação: 365 dias; total atual 54 janelas completas.',
'guaratuba':'Bloco da série de Guaratuba. Contexto inicial de avaliação: 730 dias; total atual 104 janelas completas.',
'path':'Caminho relativo à raiz do projeto para CSV já processado. A extensão não define sozinho esquema; validate_series o verifica.',
'context':'Quantidade inicial de dias antes da primeira origem de teste. Expansivo cresce após essa origem; não é limite fixo em todas as janelas.',
'numerical_covariates':'Lista ordenada de colunas numéricas. Nomes passam intactos para prompt/adaptadores; também entram na identidade da execução.',
'categorical_covariates':'Lista ordenada de categorias/binários. TimesFM converte para strings; Qwen recebe valores na tabela textual.',
'horizon':'Sete valores futuros por origem; o alvo é [origin, origin+7).',
'step':'Avança sete dias entre origens; com horizonte sete as janelas avaliadas não se sobrepõem.',
'validation_fraction':'Proporção 0,2 para validação em uma extensão de treino. A inferência zero-shot atual não a usa para otimizar parâmetros.',
'seed':'Semente global transmitida ao Qwen/seleção de exemplos. Os adaptadores numéricos usam 42 internamente.',
'qwen':'Parâmetros que **config["qwen"] expande em OllamaConfig; chaves precisam corresponder aos campos da dataclass.',
'model':'Tag exata do checkpoint instalado no Ollama; o digest é consultado no servidor e guardado em run.json.',
'base_url':'Endpoint local do servidor Ollama. 11435 separa o servidor do 4B; o cliente adiciona /api/tags, /api/show e /api/generate.',
 'temperature':'Temperatura de geração enviada em options; zero não equivale a treinamento nem a garantia absoluta de determinismo.',
'num_ctx':'Capacidade em tokens do contexto do servidor por sequência. Precisa comportar prompt, template e resposta.',
'num_predict':'Limite de tokens gerados pelo Ollama. Não substitui horizon, que é quantidade de números previstos.',
 'timeout':'Limite de espera HTTP do cliente em segundos. Pode durar mais de uma hora no caso 7200.',
'retries':'Tentativas adicionais depois da primeira; retries=1 permite duas chamadas para a mesma janela.',
'strategy':'zero retorna nenhuma demonstração; few_random/few_similar só são aceitos no layout legado.',
'few_shot_examples':'Quantidade de exemplos para estratégias few-shot. Ignorada efetivamente quando strategy=zero.',
'qwen_history_days':'15 seleciona últimos 15 dias; null vira None e mantém todo o histórico desde o início da série.',
'future_covariates_policy':'Texto de procedência armazenado com o experimento. A redação antiga diz hipótese não verificada; posteriormente o usuário confirmou disponibilidade na origem. Preservar o JSON preserva o hash histórico.',
'runtime':'Registro declarativo das opções pretendidas de servidor. forecasting_pipeline não exporta essas variáveis automaticamente; o script do servidor precisa aplicá-las.',
'ollama_flash_attention':'Registro da intenção de usar Flash Attention; aplicada por OLLAMA_FLASH_ATTENTION no script do servidor.',
'ollama_kv_cache_type':'Registro de cache KV q4_0; diferente da quantização Q4_K_M dos pesos.',
'ollama_num_parallel':'Uma sequência simultânea neste experimento para limitar memória e facilitar reutilização de prefixo.',
'llama_arg_cache_ram':'Zero desabilita cache adicional de snapshots; não desliga estados KV da sequência ativa.',
'prompt_layout':'tabular escolhe build_tabular_prompt; se chave ausente, o pipeline usa legacy e build_prompt.',
}
NOTES[('src/pipelines/compare_qwen_contexts.py',59)]=NOTES.pop(('src/pipelines/compare_qwen_contexts.py',58))
API_TEXT += r'''
startswith|Testa prefixo textual exato.|prefix é a parte histórica do prompt anterior mais uma quebra de linha. Confirma que o novo prompt expansivo começa com os mesmos bytes textuais.
split|Divide uma string por separador.|sep='\\nEND HISTORY:' ou marcador CSV identifica segmentos do prompt no teste; a indexação [0]/[1] escolhe trecho esperado.
mean|Calcula média dos valores do objeto receptor.|Sem argumentos nas Series de duração/demanda e arrays de erro; o receptor já foi filtrado para a janela ou subconjunto correto.
std|Desvio-padrão do receptor quando usado como método.|Neste projeto a padronização principal usa np.std; não confundir com pandas.std, cujo ddof padrão pode diferir.
ValueError|Cria exceção para valor/formato inválido.|A string explica a violação: datas, dimensões, covariáveis, JSON ou contexto. Só é lançada quando precedida por raise ou usada por teste de exceção.
RuntimeError|Cria exceção para estado de execução inesperado.|Mensagem sinaliza caminho que deveria ser inalcançável depois das tentativas.
SystemExit|Solicita encerramento do processo Python.|code=0 indica sucesso; 1 sinaliza falhas para scripts shell. Usada com raise no CLI.
ForecastWindow|Constrói dataclass de uma origem temporal.|context_start=0, origin=posição do primeiro alvo, horizon=sete dias. history/target são properties que devolvem slices.
Demonstration|Constrói dataclass de exemplo few-shot.|history e target são tuplas de números; target_end é posição final exclusiva dentro do prefixo autorizado.
OllamaConfig|Constrói configuração tipada do cliente Qwen.|Campos do JSON qwen e seed são argumentos nomeados. Os defaults legados só valem para campos omitidos; __post_init__ valida limites.
QwenForecaster|Constrói adaptador Qwen e consulta servidor local.|config é OllamaConfig. Valida modelo/contexto e cria identity; não chama geração até predict.
ChronosForecaster|Constrói adaptador Chronos2 com pesos locais.|context é contexto inicial registrado; dataset fornece nomes de covariáveis. predict depois recebe história crescente de verdade.
TimesFMForecaster|Constrói adaptador TimesFM2.5 local.|context é contexto inicial registrado; dataset define covariáveis. Previsão usa XReg + TimesFM.
check|Registra uma checagem da auditoria sem interromper o laço.|condition vira bool; name identifica teste; **where fornece cidade, política e origem. Lista checks preserva evidência de sucesso/falha.
predict|Interface comum dos adaptadores do projeto.|history é tabela histórica; horizon é quantidade prevista; future_covariates é tabela sem alvo; prompt é texto já construído para Qwen. Retorna lista de floats; cada classe escolhe backend.
'''
SHELL={
'experiments/start_ollama_context_server.sh':[
'Escolhe Bash pelo PATH via /usr/bin/env; o sistema usa essa primeira linha quando executa o script diretamente.',
'-e encerra em falhas não tratadas; -u rejeita variáveis inexistentes; pipefail propaga falhas de componentes de pipelines shell.',
'export disponibiliza a variável aos processos filhos. HOST escolhe interface local e porta 11435; não publica o modelo em uma interface externa.',
'Habilita a opção de Flash Attention do servidor, quando suportada pelo backend.',
'Configura quantização q4_0 para cache de atenção K/V; não é a tag Q4_K_M dos pesos.',
'Limita a uma sequência simultânea para reduzir memória e controlar a execução sequencial.',
'Desabilita RAM extra de snapshots de prompts no backend usado; o KV da sequência ativa continua existindo.',
'exec substitui o processo Bash por ollama serve. O servidor fica ativo e atende HTTP; não chama o pipeline Python por conta própria.'
],
'experiments/run_qwen_context_comparison.sh':[
'Seleciona Bash para interpretar o arquivo.',
'Ativa saída em erro, rejeição de variáveis não definidas e propagação de falhas em pipes.',
'dirname "$0" extrai a pasta do script; $(...) substitui pela saída; /.. sobe à raiz; cd muda o diretório de trabalho para caminhos relativos funcionarem.',
'${CONDA_ENVS:-...} conserva variável fornecida ou usa o caminho padrão. Não ativa/modifica ambientes; escolhe executáveis existentes.',
'Define raiz de resultados da comparação, sem espaços em volta de = na atribuição shell.',
'mkdir -p cria a pasta de logs e pais se necessário, sem falhar por já existir.',
'Loop shell com duas políticas: primeiro 15_days, depois expanding. Não inclui sete dias automaticamente.',
'Invoca Python do ambiente promptcast com -m para localizar o módulo. Barra invertida prolonga o comando na próxima linha.',
'Passa --model qwen e JSON cujo nome contém a política atual; a expansão ${CONTEXT_POLICY} ocorre no shell.',
'--output separa resultados dos braços; > grava stdout no log, sobrescrevendo o log anterior; 2>&1 manda stderr ao mesmo destino. Respostas salvas com hash continuam reutilizáveis.',
'Encerra o corpo do loop. set -e impede chegar à comparação se alguma execução retornar código não zero.',
'Executa compare_qwen_contexts após os dois braços completos. O servidor Ollama precisa ter sido iniciado em outro processo.'
],
'experiments/run_benchmark.sh':[
'Seleciona o interpretador Bash.',
'Comentário de política: reutilizar ambientes existentes sem instalar ou atualizar bibliotecas.',
'Configura tratamento estrito de falhas e variáveis.',
'Muda à raiz do projeto usando a localização do script, não a pasta de onde o usuário o chamou.',
'Escolhe raiz dos ambientes conda por variável opcional ou padrão; mantém os ambientes intactos.',
'Cria diretório de logs do benchmark.',
'Executa Qwen no ambiente promptcast com configuração padrão benchmark.json e redireciona toda saída ao log. Isso usa Qwen3 1.7B, não o JSON do 4B.',
'Solicita ao servidor Ollama padrão descarregar o modelo 1.7B para liberar recursos antes das baselines. Não apaga pesos em disco.',
'Executa Chronos no ambiente chronos; HF_HUB_OFFLINE=1 vale para esse processo e impede acesso ao Hub. Log captura saída/erros.',
'Executa TimesFM no ambiente timesfm. HF_HUB_OFFLINE vale para o Hub; XLA_PYTHON_CLIENT_PREALLOCATE=false desabilita pré-alocação XLA se esse backend for usado. O caminho principal do modelo aqui é PyTorch.',
'Invoca comparador dos três modelos após todas as inferências terminarem com sucesso.'
]}
API_TEXT += r'''
Path.cwd|Obtém o diretório de trabalho atual do processo.|Sem argumentos. Na auditoria, pressupõe execução a partir da raiz do repositório; diferente de ROOT derivado de __file__, depende da pasta de lançamento.
equals|Compara conteúdo e estrutura do receptor com outro objeto pandas.|other é DataFrame ou DatetimeIndex compatível. Retorna bool; datas/ordem precisam coincidir. Não alinha ou reordena automaticamente para fazer as tabelas parecerem iguais.
drop|Retorna tabela sem as linhas/colunas indicadas.|columns='water_produced' nos testes remove o alvo do futuro; inplace padrão False mantém o original intacto.
relative_to|Calcula caminho relativo a um ancestral de um Path.|other=ROOT; o resultado entra no hash de fonte para considerar nome/local do arquivo sem depender do prefixo absoluto da máquina.
itertuples|Itera linhas como tuplas nomeadas.|Sem argumentos, inclui índice e atributos pelos nomes das colunas válidos. Na auditoria permite r.dataset, r.policy e r.origin ao identificar piores janelas.
'''
INTRO.append(('Mini-exemplo: objetos, texto, HTTP e avaliação', '''Exemplo didático reduzido: suponha uma história de dois dias, com demandas 100 e 110, covariável temperature de 20 e 21, e feriado igual a zero. O horizonte fictício deste exemplo é dois, apenas para mostrar a transformação; os experimentos reais usam sete.
model_inputs entrega history com timestamp, water_produced, temperature e holiday. future contém as próximas duas datas e apenas temperature/holiday. Não contém os dois valores reais de demanda que serão usados depois. DataFrame é objeto tabular; build_tabular_prompt o converte em uma única string. Serializar não treina nem ajusta parâmetros do modelo.
Na cópia frame, map(lambda value: f'{float(value):.2f}') transforma 100 em '100.00' e 110 em '110.00'. to_csv(index=False, float_format='%.6g') retorna texto começando por timestamp,water_produced,temperature,holiday, seguido de uma linha por dia. index=False evita um contador 0,1 virar coluna extra. strip remove só o espaço/quebra das bordas. join junta esse CSV com as instruções, delimitadores e tabela futura.
prediction_schema(2) devolve um dict descrevendo minItems=2/maxItems=2. Ele vai no campo format, não dentro de predictions. payload inclui prompt, model e options. json.dumps(payload) transforma o dict em texto JSON; encode('utf-8') transforma esse texto em bytes para a rede. urllib.request.Request monta o POST e urlopen efetivamente envia.
Suponha que o envelope recebido contenha response='{"predictions":[112,114]}', done=True e os contadores de tokens. json.load(response_HTTP) lê o envelope. parse_prediction aplica json.loads na string interna, encontra a lista de dois números, testa finitude e devolve [112.0,114.0]. Ou seja, existem duas decodificações JSON em níveis diferentes: transporte e saída do LLM.
Somente depois o pipeline consulta a demanda real das duas datas: suponha [111,117]. zip associa 112 a 111 e 114 a 117; enumerate(...,1) atribui leads 1 e 2. Os erros são +1 e -3; MAE=(1+3)/2=2; RMSE=sqrt((1+9)/2)=sqrt(5), aproximadamente 2,236. O modelo não recebeu os valores 111 e 117 para produzir a previsão.
Se uma previsão vier como string '112', parse_prediction rejeita em vez de convertê-la permissivamente. Se vierem só um número, rejeita antes do zip. Se a resposta original completa for guardada com request_id válido, uma retomada pode reutilizá-la sem novo HTTP. Esses passos conectam biblioteca, contrato de dados e proteção temporal em um mesmo fluxo.'''))
