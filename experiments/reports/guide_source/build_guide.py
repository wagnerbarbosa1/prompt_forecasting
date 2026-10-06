"""Build a source-synchronized Portuguese manual, PDF and searchable HTML."""
import ast
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import textwrap
from content import INTRO, PURPOSE, NOTES, API_TEXT, FUNCTIONS, CONFIG_KEYS, SHELL
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    PageBreak, Spacer, KeepTogether, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'experiments/reports'
API={}
for line in API_TEXT.strip().splitlines():
    if line.strip():
        name, description, params=line.split('|',2); API[name]=(description,params)

def short(node):
    return ast.unparse(node) if node is not None else 'None'

def api_key(name):
    if name in API: return name
    leaf=name.split('.')[-1]
    return leaf if leaf in API else None

ORDER=[
'pyproject.toml', 'experiments/start_ollama_context_server.sh',
'experiments/run_qwen_context_comparison.sh','experiments/run_benchmark.sh',
'configs/benchmark.json','configs/qwen_context_15_days.json','configs/qwen_context_expanding.json',
'src/pipelines/forecasting_pipeline.py','src/pipelines/data_pipeline.py',
'src/data/ingestion/files.py','src/data/validation/schema.py','src/data/preprocessing/resampling.py',
'src/data/preprocessing/missing_values.py','src/evaluation/cross_validation.py',
'src/data/features/temporal.py','src/data/features/covariates.py',
'src/data/preprocessing/generate_prompts.py','src/models/promptcast.py',
'src/models/chronos.py','src/models/timesfm.py','src/evaluation/metrics.py',
'src/pipelines/compare_models.py','src/pipelines/compare_qwen_contexts.py','src/training/train.py',
 'tests/test_forecasting.py','tests/test_qwen_context_comparison.py','tests/test_seven_day_metrics.py',
 'experiments/qwen_context_comparison/audit/audit_run.py']
blocks=[]; manifest=[]; used_api=set(); unresolved=set(); line_coverage=[]
def add(kind,text,key=None): blocks.append((kind,text,key))
def paras(text):
    for p in text.split('\n'):
        if p.strip(): add('p',p.strip())
def chapter(title,key): add('h1',title,key)

chapter('Qwen Forecast\nCódigo, arquitetura e inferência\nGuia completo comentado','capa')
paras('Edição de 6 de outubro de 2026. Leitura linha a linha do projeto atual, com foco na construção de prompts e na inferência local pelo Ollama.\nInclui módulos de dados, três adaptadores de modelos, métricas, CLIs, configurações, scripts, testes e auditoria.\nA documentação preserva o comportamento atual e evidencia limitações. Não houve alteração do pipeline ou de resultados para produzir este livro.')
add('p','Retrato Git: '+subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip()+'. Fontes atuais também incluem a auditoria local; hashes individuais constam no manifesto acompanhante.')
add('toc','Sumário navegável','sumario')
for i,(title,text) in enumerate(INTRO,1):
    chapter(f'{i}. {title}',f'intro{i}'); paras(text)
    if i==3:
        add('code', '''qwen_forecast (pyproject.toml)
  -> forecasting_pipeline.main()
     -> run_dataset()
        -> prepare_dataset()
           -> read_series() -> pd.read_csv() -> validate_series()
           -> daily_grid() -> fill_past_only() -> build_windows()
        -> create_model() -> OllamaConfig -> QwenForecaster
           -> /api/tags + /api/show
        -> para cada janela:
           model_inputs() -> history + future SEM alvo
           build_tabular_prompt() -> string -> prompt.txt
           hash -> cache valido OU predict()
           predict() -> post_json() -> /api/generate
           parse_prediction() -> sete floats
           zip(truth, predictions) -> registros por lead
        -> predictions.csv + status.json
  -> comparador (invocado separadamente pelo shell)
     -> deterministic_metrics() -> CSVs de metricas''')
chapter('9. Inventário e dependências entre módulos','inventory')
for rel in ORDER:
    p=ROOT/rel
    add('p',f'{rel} — {len(p.read_text().splitlines())} linhas. '+PURPOSE.get(rel, 'Configuração declarativa.' if p.suffix in ('.json','.toml') else 'Script de execução.' if p.suffix=='.sh' else 'Testes automatizados de contratos do pipeline.'))
empty=[p for p in sorted((ROOT/'src').rglob('__init__.py')) if not p.read_text().strip() and '.egg-info' not in str(p)]
for p in empty:
    add('p',str(p.relative_to(ROOT))+' — vazio; identifica/organiza um pacote Python, sem função, efeito de inicialização ou chamada de inferência a explicar.')

# Project call index extracted from the exact source text.
project_defs={}
for rel in ORDER:
    if rel.endswith('.py'):
        tree=ast.parse((ROOT/rel).read_text())
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
                project_defs.setdefault(n.name,[]).append((rel,n.lineno))

def signature(n):
    a=n.args; result=[]; positional=a.posonlyargs+a.args
    offset=len(positional)-len(a.defaults)
    for j,arg in enumerate(positional):
        s=arg.arg+(': '+short(arg.annotation) if arg.annotation else '')
        if j>=offset:s+=' = '+short(a.defaults[j-offset])
        result.append(s)
    if a.vararg: result.append('*'+a.vararg.arg)
    elif a.kwonlyargs: result.append('*')
    for arg,default in zip(a.kwonlyargs,a.kw_defaults):
        result.append(arg.arg+(': '+short(arg.annotation) if arg.annotation else '')+(' = '+short(default) if default else ' [obrigatório]'))
    if a.kwarg: result.append('**'+a.kwarg.arg)
    return n.name+'('+', '.join(result)+')'+(' → '+short(n.returns) if n.returns else '')

ARTIFACT_FIELDS={'origin':'data de início do horizonte previsto','lead':'posição de 1 a 7 dentro do horizonte','y_true':'demanda observada usada somente na avaliação','y_pred':'valor retornado pelo modelo para a data correspondente','history_start':'primeira data histórica efetivamente fornecida','initial_context':'origem inicial do protocolo de avaliação','context':'número de linhas históricas efetivamente fornecidas','run_id':'identificador derivado da procedência da execução','elapsed_seconds_window':'duração da janela, repetida em seus sete registros','status':'estado ok/error de execução','failures':'lista de janelas com falha','predictions':'lista ordenada de números retornados','prompt':'texto integral enviado ao modelo','request_id':'hash da execução, janela e prompt','audit':'pedido, tentativas e respostas brutas para verificação','dataset_sha256':'hash dos bytes do CSV de entrada','source_sha256':'hash de todos os .py sob src','model':'identidade do checkpoint/adaptador','expected_windows':'quantidade de janelas que deveriam terminar','successful_windows':'quantidade de janelas com previsões válidas','history_rows':'comprimento da história','history_end':'última data conhecida antes da origem','covariates':'nomes das variáveis auxiliares','mae':'média do erro absoluto','rmse':'raiz do erro quadrático médio','r2':'coeficiente de determinação','mape_percent':'erro percentual absoluto médio em porcentagem','smape_percent':'erro percentual absoluto simétrico em porcentagem'}

def describe_statement(n):
    if isinstance(n,ast.Assign):
        return 'Atribui a '+', '.join(short(t) for t in n.targets)+' o resultado de '+short(n.value)+'. A expressão à direita é avaliada antes da atribuição.'
    if isinstance(n,ast.AnnAssign): return 'Declara o campo '+short(n.target)+' com tipo esperado '+short(n.annotation)+(' e valor inicial '+short(n.value) if n.value else ', a ser fornecido na construção')+'. A anotação não faz conversão automática.'
    if isinstance(n,ast.AugAssign):return 'Atualiza '+short(n.target)+' acumulando o resultado de '+short(n.value)+'.'
    if isinstance(n,ast.Return):return 'Encerra esta função e devolve '+short(n.value)+' ao chamador; não imprime o resultado por si só.'
    if isinstance(n,ast.If):return 'Avalia a condição '+short(n.test)+'. O bloco indentado seguinte só é executado quando ela é verdadeira.'
    if isinstance(n,ast.For):return 'Percorre '+short(n.iter)+'; em cada iteração associa o item a '+short(n.target)+'. O corpo indentado aplica a operação a esse item/grupo.'
    if isinstance(n,ast.With):return 'Abre o contexto gerenciado '+', '.join(short(i.context_expr)+(' como '+short(i.optional_vars) if i.optional_vars else '') for i in n.items)+'. Ao sair, libera o recurso ou restaura a simulação/redirecionamento, inclusive se ocorrer erro.'
    if isinstance(n,ast.Raise):return ('Interrompe o caminho atual lançando '+short(n.exc)+'. O chamador precisa capturar a exceção ou ela encerra a execução.' if n.exc else 'Relança a exceção corrente, conservando sua informação para o nível chamador.')
    if isinstance(n,ast.Expr):return 'Executa '+short(n.value)+'. O resultado não é atribuído; o efeito pode ser gravar arquivo, modificar lista, imprimir ou validar uma condição.'
    if isinstance(n,ast.Try):return 'Inicia região protegida de execução. Exceções compatíveis serão tratadas pelo except correspondente; sucesso continua normalmente.'
    if isinstance(n,ast.Continue):return 'Avança imediatamente à próxima iteração; esta janela com falha não recebe linhas de previsão.'
    if isinstance(n,ast.Pass):return 'Corpo vazio intencional: não executa operação.'
    return 'Integra a instrução iniciada na linha '+str(n.lineno)+', dentro do fluxo descrito pela função.'

for idx,rel in enumerate(ORDER,10):
    path=ROOT/rel; source=path.read_text(); lines=source.splitlines(); key='file'+str(idx)
    chapter(f'{idx}. {rel}',key)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.append(dict(file=rel,lines=len(lines),sha256=digest))
    add('p',PURPOSE.get(rel,'Testes de regressão. São executados por unittest discover, não durante a inferência.' if rel.startswith('tests/') else 'Arquivo de configuração/execução que conecta parâmetros e módulos.'))
    add('small','SHA-256: '+digest)
    tree=ast.parse(source) if path.suffix=='.py' else None
    nodes=list(ast.walk(tree)) if tree else []
    doc_lines=set()
    for n in nodes:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
            doc_lines.update(range(n.lineno,n.end_lineno+1))
    fn_defs=[n for n in nodes if isinstance(n,ast.FunctionDef)]
    if fn_defs:
        add('h2','Funções, contratos e pontos de conexão',None)
        for n in sorted(fn_defs,key=lambda x:x.lineno):
            add('code',f'L{n.lineno}: '+signature(n))
            contract=FUNCTIONS.get(n.name)
            if n.name=='__init__':contract='Construtor da classe: self recebe a instância e os demais argumentos inicializam configuração, backend ou estado. Confira o corpo abaixo para I/O e carregamento de modelo.'
            elif n.name=='predict':contract=API['predict'][1]+' O método pertence à classe adaptadora deste arquivo.'
            elif n.name=='__post_init__':contract='Valida limites após a inicialização automática da dataclass; não é chamado manualmente no pipeline.'
            elif n.name in ('history','target'):contract='Property calculada de ForecastWindow: retorna slice, sem materializar a tabela até o chamador aplicá-lo com .iloc.'
            elif n.name.startswith('test_'):contract='Caso de teste: '+n.name[5:].replace('_',' ')+'. self é a instância de unittest.TestCase; as asserções descritas abaixo definem o contrato exigido.'
            add('p',contract or 'Rotina auxiliar deste arquivo; as entradas são as da assinatura e os retornos/efeitos estão detalhados nas linhas abaixo.')
            calls=sorted({short(c.func) for c in ast.walk(n) if isinstance(c,ast.Call)})
            project=[c for c in calls if c.split('.')[-1] in project_defs and c.split('.')[-1]!=n.name]
            if project:
                links=[]
                for call in project:
                    for dst,num in project_defs[call.split('.')[-1]]:
                        if not dst.startswith('tests/'):
                            links.append(call+' → '+dst+':'+str(num))
                if links:add('small','Chamadas do projeto: '+'; '.join(links))
    file_api=set()
    for n in nodes:
        if isinstance(n,ast.Call):
            name=short(n.func); ak=api_key(name)
            if ak:file_api.add(ak);used_api.add(ak)
            elif name.split('.')[-1] not in project_defs:unresolved.add(name)
    if file_api:
        add('h2','APIs externas usadas neste arquivo',None)
        add('p',', '.join(sorted(file_api))+'. O catálogo final explica finalidade e parâmetros; junto a cada chamada abaixo constam os argumentos concretos.')
    add('h2','Código comentado, linha a linha',None)
    for number,line in enumerate(lines,1):
        if not line.strip():
            # Non-executable separators remain documented without spending one box per blank.
            line_coverage.append(dict(file=rel,line=number,kind='blank',explained=True));continue
        text=line.strip(); explanation=NOTES.get((rel,number))
        if explanation is None and tree:
            same=[n for n in nodes if getattr(n,'lineno',0)==number]
            if text.startswith(('import ','from ')):
                n=next((n for n in same if isinstance(n,(ast.Import,ast.ImportFrom))),None)
                explanation='Importa '+(', '.join(a.name+(' como '+a.asname if a.asname else '') for a in n.names) if n else text)+(' de '+n.module if isinstance(n,ast.ImportFrom) else '')+'. Disponibiliza os nomes para as chamadas deste módulo; não executa a previsão.'
            elif number in doc_lines:explanation='Documentação textual do módulo/função. Descreve o contrato; não é instrução de cálculo ou acesso ao dataset.'
            elif text.startswith('#'):explanation='Comentário de implementação: '+text.lstrip('# ')+'. Não é executado; as operações efetivas são as linhas seguintes.'
            elif text.startswith('@'):explanation='Aplica o decorador '+text[1:]+(' à classe declarada em seguida; gera comportamento de dataclass.' if 'dataclass' in text else ' ao método seguinte; permite acessá-lo como atributo calculado.')
            elif text.startswith('def '):
                n=next((n for n in same if isinstance(n,ast.FunctionDef)),None)
                explanation='Define '+(signature(n) if n else text)+'. O corpo só executa quando essa função/método é chamado. Seu contrato e chamadores estão na abertura deste arquivo.'
            elif text.startswith('class '): explanation='Declara a classe '+text.removeprefix('class ').removesuffix(':')+'. Seus métodos compartilham estado por self; a declaração não cria uma previsão.'
            elif text.startswith('except '): explanation='Trata as exceções indicadas na linha e vincula a falha a exc quando declarado. O registro permite diagnosticar a tentativa sem inventar previsões.'
            elif text=='else:':explanation='Ramo alternativo da condição imediatamente anterior: executa quando o teste anterior não foi satisfeito.'
            elif text in (']',')','}','])','))',']):'):explanation='Fecha a coleção/chamada iniciada acima. Não inicia uma nova transformação de dados; a indentação seguinte determina a continuidade do bloco.'
            else:
                stmt=[n for n in same if isinstance(n,ast.stmt) and not isinstance(n,(ast.FunctionDef,ast.ClassDef))]
                if stmt:explanation=' '.join(describe_statement(n) for n in stmt)
                else:
                    kv=[]
                    for n in nodes:
                        if isinstance(n,ast.Dict):
                            for k,v in zip(n.keys,n.values):
                                if k is not None and getattr(k,'lineno',0)==number:
                                    kn=short(k); meaning=ARTIFACT_FIELDS.get(k.value if isinstance(k,ast.Constant) else '', 'campo do registro/coleção em construção')
                                    kv.append(kn+': '+meaning+'; recebe '+short(v))
                    kws=[n for n in same if isinstance(n,ast.keyword)]
                    if kv:explanation='Define '+ '; '.join(kv)+'. Os valores são avaliados nesta janela/iteração.'
                    elif kws:explanation='Continua os argumentos nomeados: '+'; '.join((k.arg or '**')+' = '+short(k.value) for k in kws)+'. Esses valores pertencem à chamada iniciada acima; consulte seu contrato de parâmetros.'
                    elif text.startswith('for '):explanation='Cláusula de compreensão: percorre os valores indicados para construir a coleção da instrução anterior, mantendo a ordem desse percurso.'
                    elif text.startswith('if '):explanation='Filtro da compreensão iniciada acima; apenas elementos que atendem à condição são incluídos.'
                    else:
                        enclosing=[n for n in nodes if isinstance(n,ast.stmt) and getattr(n,'lineno',0)<number<=getattr(n,'end_lineno',0)]
                        n=min(enclosing,key=lambda n:n.end_lineno-n.lineno) if enclosing else None
                        explanation='Continuação da '+('instrução na linha '+str(n.lineno) if n else 'expressão anterior')+': fornece os valores/expressões mostrados no código, na mesma chamada ou coleção. '+(FUNCTIONS.get(n.name,'') if isinstance(n,ast.FunctionDef) else '')
        elif explanation is None and path.suffix=='.sh': explanation=SHELL[rel][number-1]
        elif explanation is None and path.suffix=='.json':
            match=re.match(r'"([^"]+)"\s*:',text)
            if match:explanation=CONFIG_KEYS.get(match[1],'Campo declarativo do arquivo JSON.')+' JSON é lido por json.loads; não executa chamadas por si só.'
            elif text.startswith('"'):
                name=text.rstrip(',').strip('"'); explanation='Nome de covariável '+name+'. O CSV precisa conter exatamente esta coluna; ela é preservada na história e no futuro sem o alvo.'
            else:explanation='Delimitador JSON: abre/fecha um objeto ou lista e organiza os campos do bloco. As vírgulas separam elementos; null representa ausência de corte histórico.'
        elif explanation is None:
            toml={1:'Seção que configura a ferramenta de construção do pacote.',2:'Dependência de build setuptools>=61; não é comando executado no benchmark.',3:'Backend setuptools.build_meta usado para construir/instalar o pacote.',5:'Metadados do projeto instalável.',6:'Nome de distribuição, diferente do pacote importado src.',7:'Versão declarada do projeto.',8:'Descrição textual do pacote.',9:'Versão mínima de Python declarada. A sintaxe de união | exige Python 3.10+.',10:'Dependências comuns; PyTorch/Chronos/TimesFM pertencem aos ambientes especializados, não são instalados por esta lista.',12:'Tabela de comandos de console que a instalação disponibiliza.',13:'Cria qwen_forecast apontando para main de forecasting_pipeline. Não procura arquivo qwen_forecast.py.',14:'Cria forecast-compare apontando para main de compare_models.',16:'Configura descoberta de pacotes do setuptools.',17:'Busca pacotes a partir da raiz atual.',18:'Inclui pacotes cujo nome começa com src. A estrutura real, não nomes de arquivos antigos, determina os imports.'}
            explanation=toml.get(number,'Delimitação/separação da configuração TOML; não executa previsão.')
        # Calls starting on this exact physical line: operation + actual arguments.
        details=[]
        for call in [n for n in nodes if isinstance(n,ast.Call) and n.lineno==number]:
            name=short(call.func); ak=api_key(name)
            desc=API[ak][0] if ak else FUNCTIONS.get(name.split('.')[-1],'Chamada de rotina do projeto ou construtor descrito no mapa de funções.')
            actual=[f'posicional {i+1}: {short(a)}' for i,a in enumerate(call.args)]
            actual += [(k.arg or '**desempacotamento')+': '+short(k.value) for k in call.keywords]
            details.append(name+' — '+desc+' Argumentos usados: '+('; '.join(actual) if actual else 'nenhum argumento explícito')+'.')
        add('line',(number,line,explanation,details),key+'L'+str(number))
        line_coverage.append(dict(file=rel,line=number,kind='code',explained=bool(explanation)))
    add('small','Linhas em branco são separadores sem execução; seus números foram preservados nos saltos da listagem. As linhas de código e comentários acima seguem o arquivo integral, sem omissão de trechos.')

chapter('Catálogo de funções e métodos de biblioteca','api')
paras('Esta referência explica as APIs efetivamente chamadas. Em cada ocorrência na listagem, os argumentos concretos foram mostrados, inclusive os posicionais. Aqui estão seu significado e os parâmetros relevantes. Métodos homônimos (como update de dict e de hash) dependem do objeto receptor. O catálogo não deve ser lido como lista de todos os parâmetros opcionais que a biblioteca admite.')
for name in sorted(used_api):
    add('h2',name,None);add('p',API[name][0]);add('p','Parâmetros e uso neste projeto: '+API[name][1])
chapter('Como executar e localizar o resultado','commands')
for code,desc in [
('bash experiments/start_ollama_context_server.sh','Em um terminal, inicia o servidor local separado; use ambientes já disponíveis, sem instalá-los ou atualizá-los.'),
('conda run -n promptcast python -m src.pipelines.forecasting_pipeline \\\n  --model qwen --config configs/qwen_context_expanding.json \\\n  --output experiments/qwen_context_comparison/expanding','Executa Qwen3 4B expansivo nas duas cidades. Reutiliza somente caches com identidade exata. Não execute duplicatas escrevendo na mesma pasta.'),
('conda run -n promptcast python -m src.pipelines.compare_qwen_contexts \\\n  --output experiments/qwen_context_comparison','Depois dos braços 15_days e expanding completos, gera métricas e comparação controlada.'),
('conda run -n promptcast python -m unittest discover -s tests -v','Executa testes de contrato sem baixar modelos; testes mockados não substituem execução real do Ollama.'),
('PYTHONPATH=. conda run -n promptcast python \\\n  experiments/qwen_context_comparison/audit/audit_run.py','Audita respostas salvas e escreve checks/windows/summary na pasta audit. Não faz uma nova inferência.')]:
    add('code',code);add('p',desc)
paras('O ambiente selecionado precisa conter dependências já instaladas. O usuário exige consulta antes de modificar ambientes conda. O comando instalado qwen_forecast depende de instalação do pacote naquele ambiente; python -m funciona pela localização de src quando executado a partir da raiz.\nPara estudar, não é necessário executar novamente centenas de janelas. Abra uma resposta em responses, compare audit.request.prompt com o .txt correspondente e associe predictions[i] à linha lead=i+1 do CSV. Para mudar código, use validação anterior ao teste; não escolha uma formulação de prompt por desempenho em janelas finais já inspecionadas.')
chapter('Cobertura, fontes e reprodução do documento','coverage')
paras(f'Foram incluídos {len(ORDER)} arquivos de código/configuração/script e {len(empty)} inicializadores vazios. Todas as {sum(len((ROOT/p).read_text().splitlines()) for p in ORDER)} linhas físicas estão cobertas: linhas com conteúdo recebem código e anotação; linhas em branco são separadores registrados no manifesto. A extração usa os arquivos reais, não trechos reescritos.\nA referência primária é o próprio repositório. Para semântica específica de TimesFM, também foi consultada a implementação instalada de timesfm/configs.py no ambiente timesfm, incluindo a docstring de ForecastConfig. Bibliotecas possuem versões instaladas distintas por ambiente; os parâmetros documentados correspondem às chamadas deste código, não a uma promessa para qualquer versão futura.\nArquivos gerados: guia_qwen_forecast.pdf, guia_qwen_forecast.html, guide_source/manifest.json e guide_source/line_coverage.json. O conteúdo editorial e o gerador estão em guide_source/content.py e guide_source/build_guide.py, fora de src para não invalidar hashes de inferência. O gerador documenta o produto, não integra sua execução.\nPara reconstruir neste ambiente, execute python experiments/reports/guide_source/build_guide.py com o Python de sistema, que já possui ReportLab. Isso não altera ambientes conda. O manifesto é a referência para confirmar a qual versão de cada arquivo uma explicação pertence.')
(OUT/'guide_source/manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
(OUT/'guide_source/line_coverage.json').write_text(json.dumps(line_coverage,indent=2,ensure_ascii=False)+'\n')
(OUT/'guide_source/unresolved_calls.json').write_text(json.dumps(sorted(unresolved),indent=2)+'\n')

for name,file in [('Guide','DejaVuSans.ttf'),('GuideBold','DejaVuSans-Bold.ttf'),('GuideMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
styles={
'p':ParagraphStyle('P',fontName='Guide',fontSize=9.2,leading=13.3,spaceAfter=7),
'small':ParagraphStyle('Small',fontName='Guide',fontSize=7.6,leading=10.5,textColor=colors.HexColor('#475569'),spaceAfter=6),
'h1':ParagraphStyle('H1',fontName='GuideBold',fontSize=19,leading=24,spaceAfter=16,splitLongWords=True),
'h2':ParagraphStyle('H2',fontName='GuideBold',fontSize=11.5,leading=16,spaceBefore=11,spaceAfter=7,keepWithNext=True),
'code':ParagraphStyle('Code',fontName='GuideMono',fontSize=7.3,leading=10,backColor=colors.HexColor('#edf3f7'),borderPadding=6,spaceBefore=5,spaceAfter=6),
}
E=lambda x:html.escape(str(x),quote=False)
def code_flow(s):
    wrapped='\n'.join(textwrap.fill(line,width=99,subsequent_indent='    ',replace_whitespace=False,drop_whitespace=False) if line else '' for line in s.splitlines())
    return XPreformatted(E(wrapped),styles['code'])
class Doc(BaseDocTemplate):
    def afterFlowable(self,f):
        if getattr(f,'guide_key',None):
            self.canv.bookmarkPage(f.guide_key)
            title=getattr(f,'guide_title',f.getPlainText())
            self.canv.addOutlineEntry(title,f.guide_key,0)
            self.notify('TOCEntry',(0,title,self.page,f.guide_key))
def page_footer(c,doc):
    c.saveState();c.setFont('Guide',7);c.setFillColor(colors.HexColor('#64748b'))
    c.drawString(42,24,'Qwen Forecast • Guia completo do código • 06/10/2026')
    c.drawRightString(A4[0]-42,24,str(doc.page));c.restoreState()
doc=Doc(str(OUT/'guia_qwen_forecast.pdf'),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=42,title='Qwen Forecast — Guia completo linha a linha',author='Documentação técnica do projeto')
doc.addPageTemplates(PageTemplate(id='normal',frames=[Frame(42,42,A4[0]-84,A4[1]-84,id='body')],onPage=page_footer))
story=[]; html_parts=['<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Guia completo Qwen Forecast</title><style>body{font:16px/1.55 sans-serif;max-width:1040px;margin:40px auto;padding:0 24px;color:#173044}h1{margin-top:60px;border-bottom:3px solid #167c91;padding-bottom:12px}h2{color:#116c7d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f7;padding:14px;font-size:13px}small{color:#64748b}.line{border-top:1px solid #d8e3ea;padding:8px 0}.detail{font-size:14px;color:#3b5263}a{color:#116c7d}</style><body>']
first=True
for kind,value,key in blocks:
    if kind=='h1':
        if not first:story.append(PageBreak())
        first=False
        p=Paragraph(E(value).replace('\n','<br/>'),styles['h1']);p.guide_key=key;p.guide_title=value.replace('\n',' — ');story.append(p)
        html_parts.append(f'<h1 id="{key}">{E(value).replace(chr(10),"<br>")}</h1>')
    elif kind=='toc':
        story.append(PageBreak());story.append(Paragraph(value,styles['h1']))
        toc=TableOfContents();toc.levelStyles=[ParagraphStyle('TOC',fontName='Guide',fontSize=9,leading=14,spaceBefore=3)]
        story.append(toc)
        html_parts.append('<nav><h1>Sumário</h1><ol>'+''.join(f'<li><a href="#{k}">{E(v)}</a></li>' for t,v,k in blocks if t=='h1')+'</ol></nav>')
    elif kind=='line':
        num,line,explain,details=value
        c=code_flow(f'{num:04d}  {line}')
        p=Paragraph(E(explain),styles['p'])
        # Keep the code with the first explanation; detailed call descriptions may flow naturally.
        story.append(KeepTogether([c,p]))
        html_parts.append(f'<div class="line" id="{key}"><pre>{E(str(num).zfill(4)+"  "+line)}</pre><p>{E(explain)}</p>')
        for detail in details:
            story.append(Paragraph(E(detail),styles['small']));html_parts.append('<p class="detail">'+E(detail)+'</p>')
        html_parts.append('</div>')
    elif kind=='code':story.append(code_flow(value));html_parts.append('<pre>'+E(value)+'</pre>')
    else:
        story.append(Paragraph(E(value),styles[kind]));html_parts.append(f'<{kind if kind in ("h2","p","small") else "p"}>{E(value)}</{kind if kind in ("h2","p","small") else "p"}>')
html_parts.append('</body></html>')
(OUT/'guia_qwen_forecast.html').write_text('\n'.join(html_parts))
doc.multiBuild(story)
print(json.dumps(dict(files=len(ORDER),empty_init=len(empty),lines=len(line_coverage),api_entries=len(used_api),unresolved=sorted(unresolved)),ensure_ascii=False))
