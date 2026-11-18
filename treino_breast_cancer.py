###############################################################################
# MÓDULO DE TREINAMENTO - BASE BREAST CANCER (UCI, id 14)
# Disciplina: Data Science - Avaliação Semestral
#
# OBJETIVO
#   Construir uma árvore de decisão que, a partir das características clínicas
#   de uma paciente, indique o valor do atributo "Class". O script executa
#   todos os procedimentos vistos em sala e salva o modelo de MELHOR ACURÁCIA
#   no conjunto de teste.
#
# O QUE SIGNIFICA "Class"?
#   ATENÇÃO: nesta base, Class NÃO diz se o tumor é maligno ou benigno.
#   TODAS as 286 pacientes tiveram câncer de mama. Class diz se o câncer
#   VOLTOU (recorrência) depois do tratamento:
#     - no-recurrence-events -> o câncer NÃO voltou (201 pacientes, ~70%)
#     - recurrence-events    -> o câncer VOLTOU     ( 85 pacientes, ~30%)
#
# ROTEIRO
#   1. Carregar a base
#   2. Tratar os valores ausentes
#   3. Separar atributos e classe
#   4. Tratar atributos quantitativos e categóricos
#      (inclui a análise monovariada dos quantitativos)
#   5. Separar treino e teste (e calcular a acurácia de referência)
#   6. Treinar dois cenários, com e sem balanceamento:
#        6.1 balancear  6.2 normalizar  6.3 hiperparametrizar
#        6.4 treinar    6.5 avaliar
#   7. Escolher o melhor cenário e salvar o modelo e o normalizador
#
# ARQUIVOS GERADOS (usados depois por inferencia_breast_cancer.py)
#   breast_cancer_tree.model         -> árvore de decisão treinada
#   normalizador_breast_cancer.model -> MinMaxScaler ajustado no treino
###############################################################################

import pandas as pd
import numpy as np
from pickle import dump
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from scipy import stats

# Configuração só de exibição: mostra todas as colunas das tabelas impressas
# sem cortar com "..." e sem quebrar a linha cedo. Não altera nenhum dado.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)


###############################################################################
# 1. CARREGAR A BASE
###############################################################################
# sep=','       -> no arquivo, as colunas são separadas por vírgula.
# na_values='?' -> nesta base, a informação que não foi registrada aparece
#                  como "?". Com na_values, o pandas transforma todo "?" em
#                  NaN ("Not a Number"), o marcador padrão de valor ausente.
#                  Sem isso, "?" seria lido como uma categoria válida
#                  (node-caps teria 3 valores: "yes", "no" e "?"), e o
#                  get_dummies criaria uma coluna sem sentido para ele.
dados = pd.read_csv('breast-cancer.csv', sep=',', na_values='?')

print('=' * 70)
print('1. CARREGAMENTO DA BASE')
print('=' * 70)
print(f'Registros (pacientes): {dados.shape[0]}')
print(f'Colunas: {dados.shape[1]}')
print()
print(dados.head())
print()


###############################################################################
# 2. TRATAR OS VALORES AUSENTES
###############################################################################
# Primeiro olhamos QUANTOS valores faltam e EM QUAIS colunas.
# isna() marca True onde há NaN; sum() conta os True de cada coluna.
# Filtramos (> 0) para mostrar só as colunas que têm algum ausente.
ausentes = dados.isna().sum()

print('=' * 70)
print('2. VALORES AUSENTES')
print('=' * 70)
print('Ausentes por coluna (só as que têm):')
print(ausentes[ausentes > 0])
print()

# POR QUE PREENCHER E NÃO APAGAR?
#   São 9 pacientes com algum ausente (8 em node-caps e 1 em breast-quad).
#   Numa base pequena de 286 registros, apagar essas linhas jogaria fora
#   informação útil das outras 8 colunas de cada uma delas.
#
# POR QUE A MODA?
#   node-caps e breast-quad são CATEGÓRICAS (textos como "yes"/"no" ou
#   "left_low"). Não existe média de texto, então a escolha natural é a
#   MODA: o valor mais frequente da coluna.
#   .mode() devolve uma Series (pode haver empate), e [0] pega o primeiro.
#
# Cada coluna é preenchida com a sua própria moda, em linhas separadas.
dados['node-caps'] = dados['node-caps'].fillna(dados['node-caps'].mode()[0])
dados['breast-quad'] = dados['breast-quad'].fillna(dados['breast-quad'].mode()[0])

print(f"Moda usada em node-caps:   {dados['node-caps'].mode()[0]}")
print(f"Moda usada em breast-quad: {dados['breast-quad'].mode()[0]}")
print(f'Total de ausentes depois do preenchimento: {dados.isna().sum().sum()}')
print()


###############################################################################
# 3. SEPARAR ATRIBUTOS E CLASSE
###############################################################################
# ATRIBUTOS (X) são as informações de entrada: idade, tamanho do tumor etc.
# CLASSE (y) é o que queremos prever: a coluna "Class".
# drop(columns=[...]) devolve uma cópia da tabela SEM a coluna indicada.
dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']

print('=' * 70)
print('3. ATRIBUTOS E CLASSE')
print('=' * 70)
print('Contagem de cada classe:')
print(dados_classes.value_counts())
print()
print('Proporção de cada classe:')
print(dados_classes.value_counts(normalize=True).round(3))
print()
# A base é DESBALANCEADA: cerca de 70% das pacientes não tiveram recorrência
# e 30% tiveram. Isso tem duas consequências importantes:
#   - um modelo "preguiçoso" que sempre responde "no-recurrence-events" já
#     acerta ~70% (essa é a acurácia de referência, calculada na etapa 5);
#   - o modelo vê poucos exemplos de recorrência e tende a ignorá-los.
#     O balanceamento (SMOTE, etapa 6.1) é uma tentativa de corrigir isso.


###############################################################################
# 4. TRATAR ATRIBUTOS QUANTITATIVOS E CATEGÓRICOS
###############################################################################
# O algoritmo da árvore de decisão do scikit-learn só trabalha com NÚMEROS.
# Por isso todo texto precisa ser convertido. A descrição da UCI diz que
# alguns atributos são LINEARES (têm ordem natural) e outros são NOMINAIS
# (são só nomes, sem ordem). Cada tipo recebe um tratamento diferente.
#
# -----------------------------------------------------------------------------
# 4.1 QUANTITATIVOS (têm ordem): age, tumor-size, inv-nodes e deg-malig
# -----------------------------------------------------------------------------
# age, tumor-size e inv-nodes chegam como FAIXAS DE TEXTO, por exemplo:
#   age = '50-59'   tumor-size = '30-34'   inv-nodes = '0-2'
# Mesmo sendo texto, essas faixas têm ORDEM: '20-29' < '30-39' < '40-49'...
#
# Convertemos cada faixa no seu PONTO MÉDIO: (início + fim) / 2.
#   Ex.: '50-59' -> (50 + 59) / 2 = 54.5
#        '0-2'   -> (0 + 2)   / 2 = 1.0
#
# POR QUE O PONTO MÉDIO E NÃO get_dummies?
#   O ponto médio PRESERVA A ORDEM e a distância entre as faixas: a árvore
#   pode aprender regras como "tumor-size > 30", que valem para todas as
#   faixas acima de 30 de uma só vez.
#   Com get_dummies, cada faixa viraria uma coluna 0/1 independente
#   ('age_40-49', 'age_50-59'...). O modelo perderia a informação de que
#   50-59 vem depois de 40-49 e precisaria aprender cada faixa separadamente,
#   com menos exemplos em cada uma.
#
# COMO A CONVERSÃO FUNCIONA (para cada coluna da lista):
#   coluna.str.split('-')    -> '50-59' vira a lista ['50', '59']
#   limites.str[0]           -> pega o 1º item da lista: '50' (início)
#   limites.str[1]           -> pega o 2º item da lista: '59' (fim)
#   .astype(int)             -> transforma o texto '50' no número 50
#   (inicio + fim) / 2       -> calcula o ponto médio: 54.5
colunas_faixas = ['age', 'tumor-size', 'inv-nodes']

print('=' * 70)
print('4. QUANTITATIVOS x CATEGÓRICOS')
print('=' * 70)
print('Conversão das faixas em ponto médio:')

for coluna_nome in colunas_faixas:
    coluna = dados_atributos[coluna_nome]
    limites = coluna.str.split('-')
    inicio = limites.str[0].astype(int)
    fim = limites.str[1].astype(int)
    dados_atributos[coluna_nome] = (inicio + fim) / 2

    # Só para conferência: mostra cada faixa distinta e o número gerado,
    # em ordem crescente (deve ficar na mesma ordem das faixas).
    conversao = pd.DataFrame({'faixa': coluna, 'ponto_medio': dados_atributos[coluna_nome]})
    conversao = conversao.drop_duplicates().sort_values('ponto_medio')
    print(f'  {coluna_nome:<10}: ' + ' | '.join(conversao['faixa'] + ' -> ' + conversao['ponto_medio'].astype(str)))
print()

# deg-malig (grau de malignidade: 1, 2 ou 3) JÁ é numérico e ordenado
# (grau 3 é mais agressivo que o 2, que é mais agressivo que o 1).
# O pandas já leu essa coluna como número, então ela não precisa de conversão.
#
# Estas são as 4 colunas numéricas. São SÓ ELAS que passam pelo MinMaxScaler
# na etapa 6.2 (as dummies abaixo já ficam entre 0 e 1).
colunas_num = ['age', 'tumor-size', 'inv-nodes', 'deg-malig']

# -----------------------------------------------------------------------------
# 4.2 ANÁLISE MONOVARIADA DOS QUANTITATIVOS
# -----------------------------------------------------------------------------
# Análise monovariada = descrever UMA coluna de cada vez, como visto em aula.
# Ela NÃO altera nenhum dado: serve para conhecer os atributos numéricos e
# justificar as escolhas de tratamento. É feita na base completa (286
# pacientes), porque é só uma descrição e não ajusta nada no modelo.
#
# Para cada coluna numérica calculamos:
#   - média e mediana -> o "centro" dos dados. Quando a média fica bem acima
#     da mediana, há valores altos puxando a média (distribuição assimétrica);
#   - desvio padrão -> o quanto os valores se espalham em torno da média;
#   - coeficiente de variação (CV) = desvio padrão / média x 100 -> o
#     espalhamento em PORCENTAGEM da média. Permite comparar colunas de
#     escalas diferentes (idade em anos x número de linfonodos);
#   - teste de Shapiro-Wilk (stats.shapiro) -> verifica se a coluna segue
#     uma distribuição NORMAL (a "curva de sino"). A hipótese inicial do
#     teste é que os dados SÃO normais:
#       p_valor > 0.05  -> não há evidência contra: "dados normais";
#       p_valor <= 0.05 -> rejeitamos a normalidade: "dados não normais".
#     O p_valor é impresso em notação científica: 4.23e-12 significa
#     0,00000000000423, ou seja, praticamente zero.
#
# Atenção: age, tumor-size e inv-nodes são PONTOS MÉDIOS de faixas, então
# estas estatísticas são aproximações dos valores reais de cada paciente.
print('Análise monovariada dos atributos quantitativos (base completa):')
for coluna_nome in colunas_num:
    dados_coluna = dados_atributos[coluna_nome]
    estatistica, p_valor = stats.shapiro(dados_coluna)
    if p_valor > 0.05:
        normalidade = 'dados normais'
    else:
        normalidade = 'dados não normais'
    print(f'  {coluna_nome:<10}: média {dados_coluna.mean():6.2f} | '
          f'mediana {dados_coluna.median():6.2f} | '
          f'desvio padrão {dados_coluna.std():6.2f} | '
          f'CV {dados_coluna.std() / dados_coluna.mean() * 100:6.1f}% | '
          f'Shapiro p = {p_valor:.2e} -> {normalidade}')
print()

# RESULTADO E O QUE ELE JUSTIFICA
#   Nenhum dos 4 atributos é normal (p_valor muito abaixo de 0.05), por dois
#   motivos:
#     - eles têm poucos valores distintos (6 faixas de idade, 3 graus de
#       malignidade...), e a curva normal é contínua;
#     - inv-nodes é muito ASSIMÉTRICO: 213 das 286 pacientes estão na faixa
#       0-2. Por isso a média (2.57) fica bem acima da mediana (1.0) e o CV
#       passa de 100%.
#   Consequências para o tratamento:
#     - o MinMaxScaler (etapa 6.2) é adequado, porque usa só o mínimo e o
#       máximo de cada coluna e não depende de os dados serem normais;
#     - a árvore de decisão também não exige dados normais: ela só compara
#       cada valor com pontos de corte;
#     - se houvesse valores ausentes numéricos, a MEDIANA seria melhor que a
#       média para preenchê-los, porque os dados não são normais. Nesta base,
#       os ausentes estão só em colunas categóricas (preenchidas com a moda).

# -----------------------------------------------------------------------------
# 4.3 CATEGÓRICOS NOMINAIS (sem ordem): get_dummies
# -----------------------------------------------------------------------------
# menopause   -> 'premeno', 'ge40', 'lt40'
# node-caps   -> 'yes', 'no'
# breast      -> 'left', 'right'
# breast-quad -> 'left_up', 'left_low', 'right_up', 'right_low', 'central'
# irradiat    -> 'yes', 'no'
#
# Aqui NÃO existe ordem: "left" não é maior nem menor que "right". Se
# trocássemos por números (left=0, right=1; central=0, left_up=1...), o
# modelo poderia interpretar uma ordem que não existe.
#
# O get_dummies (one-hot encoding) cria UMA COLUNA 0/1 PARA CADA VALOR:
#   breast='left'  -> breast_left=1, breast_right=0
#   breast='right' -> breast_left=0, breast_right=1
#
# Em aula, este passo é chamado de NORMALIZAÇÃO DOS DADOS CATEGÓRICOS: assim
# como o MinMax leva os números para a faixa de 0 a 1, o get_dummies
# representa cada categoria com colunas de 0 e 1.
#
# dummy_na=False -> não cria coluna para NaN (já preenchemos os ausentes).
# dtype=int      -> as dummies saem como 0/1 inteiros, e não True/False.
colunas_cat = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']
dados_dummies = pd.get_dummies(dados_atributos[colunas_cat], dummy_na=False, dtype=int)

# -----------------------------------------------------------------------------
# 4.4 JUNÇÃO
# -----------------------------------------------------------------------------
# join junta as tabelas lado a lado, alinhando pelas linhas (índice).
# Primeiro vêm as 4 numéricas e depois as dummies. Essa ordem de colunas
# fica gravada no modelo e é reproduzida na inferência.
#   4 numéricas + 14 dummies (3 + 2 + 2 + 5 + 2) = 18 colunas.
dados_atributos = dados_atributos[colunas_num].join(dados_dummies)

print(f'Total de colunas após o tratamento: {dados_atributos.shape[1]}')
print(list(dados_atributos.columns))
print()
print(dados_atributos.head())
print()


###############################################################################
# 5. SEPARAR TREINO E TESTE
###############################################################################
# O modelo aprende com o TREINO e é avaliado no TESTE, que ele nunca viu.
# Medir a acurácia nos próprios dados de treino ENGANA: a árvore pode
# "decorar" as pacientes e acertar quase tudo nelas, sem acertar casos novos.
# Por isso o teste fica guardado e SÓ É USADO NO FINAL (etapa 6.5).
#
# test_size=0.3     -> 30% das pacientes vão para o teste e 70% para o treino.
# random_state=42   -> fixa o sorteio, para que o resultado seja reproduzível.
# stratify=classes  -> mantém a proporção 70/30 das classes TANTO no treino
#                      QUANTO no teste. Sem isso, o sorteio poderia jogar
#                      poucas recorrências no teste, por puro acaso.
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, random_state=42, stratify=dados_classes)

print('=' * 70)
print('5. TREINO E TESTE')
print('=' * 70)
print(f'Treino: {atributos_treino.shape[0]} pacientes')
print(f'Teste:  {atributos_teste.shape[0]} pacientes')
print()

# ACURÁCIA DE REFERÊNCIA (linha de base)
#   É a acurácia de um "chute" que SEMPRE responde a classe mais comum
#   ("no-recurrence-events"), sem olhar nenhum atributo da paciente.
#   (classes_teste == 'no-recurrence-events') gera True/False para cada
#   paciente do teste, e .mean() dá a fração de True, que é a fração de
#   acertos desse chute.
#   O modelo só é útil se SUPERAR esse valor. Uma acurácia de 70% parece
#   boa, mas aqui ela é obtida sem nenhuma inteligência.
acuracia_referencia = (classes_teste == 'no-recurrence-events').mean()
print(f'Acurácia de referência (sempre "no-recurrence-events"): {acuracia_referencia:.3f}')
print()


###############################################################################
# 6. TREINAR DOIS CENÁRIOS: COM E SEM BALANCEAMENTO
###############################################################################
# O SMOTE (balanceamento) faz parte do roteiro da aula. Ele cria pacientes
# SINTÉTICAS da classe rara (recurrence-events) até as classes ficarem do
# mesmo tamanho no treino.
#
# PORÉM: o balanceamento pode BAIXAR a acurácia geral. Com as classes
# igualadas no treino, o modelo passa a PRIORIZAR a classe rara e responde
# "recurrence-events" com mais frequência. Só que, no teste (e na vida real),
# 70% das pacientes NÃO têm recorrência, e cada recorrência prevista por
# engano é um erro a mais na acurácia.
#
# Como o objetivo deste trabalho é a MELHOR ACURÁCIA, treinamos os dois
# cenários, com e sem SMOTE, e comparamos os dois NO MESMO TESTE. O cenário
# vencedor é o que será salvo na etapa 7.

# -----------------------------------------------------------------------------
# GRADE DE HIPERPARÂMETROS DA ÁRVORE DE DECISÃO
# -----------------------------------------------------------------------------
# Hiperparâmetros são as "configurações" do algoritmo, escolhidas ANTES do
# treino. A busca (etapa 6.3) testa combinações desta grade e escolhe a melhor.
#
# criterion -> como a árvore mede a "mistura" de classes em cada nó para
#              decidir a melhor pergunta (divisão):
#                'gini'     -> índice de Gini;
#                'entropy'  -> entropia (ganho de informação);
#                'log_loss' -> mesma conta da entropia, com outro nome no
#                              scikit-learn.
criterion = ['gini', 'entropy', 'log_loss']

# splitter -> como o ponto de corte de cada divisão é escolhido:
#               'best'   -> testa os cortes possíveis e fica com o melhor;
#               'random' -> sorteia cortes e fica com o melhor dos sorteados.
#                           Gera árvores mais variadas e às vezes reduz o
#                           overfitting.
splitter = ['best', 'random']

# min_samples_split -> número MÍNIMO de pacientes que um nó precisa ter para
#                      poder ser dividido de novo. Quanto maior, mais cedo a
#                      árvore para de crescer, e mais simples ela fica.
#   np.linspace(start=2, stop=100, num=15) gera 15 valores igualmente
#   espaçados entre 2 e 100; int() arredonda para baixo:
#   [2, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100]
#   - Começa em 2 porque o scikit-learn NÃO aceita 1: um nó com uma única
#     paciente não tem o que dividir.
#   - Vai até 100 porque, numa base pequena como esta (200 pacientes no
#     treino), árvores mais SIMPLES costumam GENERALIZAR MELHOR: uma árvore
#     muito profunda decora o treino e erra pacientes novas.
min_samples_split = [int(x) for x in np.linspace(start=2, stop=100, num=15)]

# Dicionário com a grade: cada chave é o nome exato do hiperparâmetro no
# DecisionTreeClassifier, e o valor é a lista de opções a testar.
# Total de combinações: 3 x 2 x 15 = 90.
decision_tree_grid = {'criterion': criterion,
                      'splitter': splitter,
                      'min_samples_split': min_samples_split}

# Aqui ficam guardados, para cada cenário, a acurácia de teste, o modelo e
# o normalizador. Na etapa 7, ele é usado para escolher o vencedor.
resultados = {}

for cenario in ['com_balanceamento', 'sem_balanceamento']:

    print('=' * 70)
    print(f'6. CENÁRIO: {cenario.upper()}')
    print('=' * 70)

    ###########################################################################
    # 6.1 BALANCEAR (só no cenário com balanceamento)
    ###########################################################################
    # O SMOTE (Synthetic Minority Over-sampling Technique) cria pacientes
    # sintéticas da classe minoritária. Para cada paciente com recorrência,
    # ele procura vizinhas parecidas da mesma classe e cria uma nova paciente
    # num ponto entre as duas (uma "mistura" das duas).
    #
    # O SMOTE é aplicado SÓ NO TREINO. O teste precisa continuar com a
    # proporção REAL (70/30), porque ele simula as pacientes que o modelo vai
    # encontrar de verdade. Pacientes sintéticas no teste distorceriam a
    # avaliação.
    #
    # Detalhe: como a paciente sintética é uma mistura de duas reais, as
    # dummies dela podem ficar com valores QUEBRADOS entre 0 e 1 (ex.:
    # breast_left = 0.37). Isso não representa uma paciente "real", mas a
    # árvore tolera: ela só compara valores com um ponto de corte
    # (ex.: breast_left <= 0.5).
    #
    # Observação: o SMOTE escolhe as vizinhas pela distância entre pacientes.
    # Como ele roda ANTES da normalização (ordem do roteiro), as colunas com
    # números maiores (age, tumor-size) pesam mais nessa escolha.
    if cenario == 'com_balanceamento':
        print('6.1 Balanceamento com SMOTE (somente no treino)')
        print(f'    Antes:  {classes_treino.value_counts().to_dict()}')
        atributos_treino_cenario, classes_treino_cenario = SMOTE(random_state=42).fit_resample(
            atributos_treino, classes_treino)
        print(f'    Depois: {classes_treino_cenario.value_counts().to_dict()}')
    else:
        # Sem balanceamento: usamos CÓPIAS do treino original, para que nada
        # feito neste cenário altere os dados originais por engano.
        print('6.1 Sem balanceamento: treino original mantido')
        atributos_treino_cenario = atributos_treino.copy()
        classes_treino_cenario = classes_treino.copy()
        print(f'    Classes: {classes_treino_cenario.value_counts().to_dict()}')
    print()

    ###########################################################################
    # 6.2 NORMALIZAR (MinMaxScaler)
    ###########################################################################
    # O MinMaxScaler leva cada coluna numérica para a faixa de 0 a 1:
    #   valor_normalizado = (valor - mínimo) / (máximo - mínimo)
    # Ex.: no treino, age vai de 34.5 a 74.5 (a única paciente da faixa 20-29
    # caiu no teste); a idade 54.5 vira (54.5 - 34.5) / (74.5 - 34.5) = 0.5.
    # Um valor do teste fora da faixa do treino sai abaixo de 0 ou acima de 1
    # (a paciente de 20-29 fica com age = -0.25). Isso é esperado e a árvore
    # lida bem com isso, porque só compara o valor com pontos de corte.
    #
    # Regras importantes:
    #   - SÓ OS ATRIBUTOS são normalizados, NUNCA a classe (ela é o rótulo
    #     "recurrence-events"/"no-recurrence-events", não um número).
    #   - Só as 4 colunas_num passam pelo normalizador. As dummies JÁ ESTÃO
    #     entre 0 e 1, então normalizá-las não mudaria nada.
    #   - O fit (que aprende mínimo e máximo) é feito SÓ NO TREINO do cenário.
    #     Se o teste participasse do fit, o modelo "espiaria" informação do
    #     teste, e a avaliação deixaria de ser justa.
    #   - No teste (e depois na inferência) usamos SÓ transform, com o
    #     mínimo e o máximo aprendidos no treino. NUNCA fazer fit_transform de
    #     novo: isso recalcularia mínimo e máximo com outros dados, e o mesmo
    #     valor passaria a ter outra escala.
    #   - Treino, avaliação e inferência usam SEMPRE dados normalizados. No
    #     iris_classificador da aula, a árvore foi treinada SEM normalizar e
    #     avaliada COM dados normalizados, o que é um ERRO: os pontos de corte
    #     aprendidos (ex.: tumor-size <= 30) seriam comparados com valores
    #     entre 0 e 1, e a previsão sairia errada.
    #
    # Curiosidade: a árvore de decisão não depende da escala dos números
    # (ela só compara "maior ou menor"). A normalização segue o roteiro da
    # aula, e o essencial é que o MESMO tratamento seja aplicado no treino,
    # no teste e na inferência.
    #
    # Como em aula: o fit devolve o MODELO NORMALIZADOR (o normalizador já
    # treinado, com o mínimo e o máximo de cada coluna). É ele que normaliza
    # o treino e o teste, e é ele que será salvo para a inferência.
    #
    # Trabalhamos em CÓPIAS para não alterar os dados originais, que ainda
    # serão usados pelo outro cenário.
    normalizador = MinMaxScaler()
    modelo_normalizador = normalizador.fit(atributos_treino_cenario[colunas_num])

    atributos_treino_norm = atributos_treino_cenario.copy()
    atributos_treino_norm[colunas_num] = modelo_normalizador.transform(atributos_treino_cenario[colunas_num])

    atributos_teste_norm = atributos_teste.copy()
    atributos_teste_norm[colunas_num] = modelo_normalizador.transform(atributos_teste[colunas_num])

    print('6.2 Normalização MinMax (fit só no treino; transform no treino e no teste)')
    print(f'    Mínimos aprendidos: {dict(zip(colunas_num, modelo_normalizador.data_min_.tolist()))}')
    print(f'    Máximos aprendidos: {dict(zip(colunas_num, modelo_normalizador.data_max_.tolist()))}')
    print()

    ###########################################################################
    # 6.3 HIPERPARAMETRIZAÇÃO (RandomizedSearchCV)
    ###########################################################################
    # O RandomizedSearchCV SORTEIA combinações da grade e avalia cada uma com
    # VALIDAÇÃO CRUZADA, usando SÓ O TREINO (o teste continua guardado).
    #
    # estimator           -> o algoritmo a ajustar: a árvore de decisão.
    #                        random_state=42 deixa a árvore reproduzível.
    # param_distributions -> a grade definida antes do laço.
    # n_iter=50           -> testa 50 das 90 combinações possíveis.
    # cv=5                -> validação cruzada em 5 partes: o treino é
    #                        dividido em 5 pedaços; o modelo treina em 4 e
    #                        é avaliado no 5º, revezando 5 vezes. O resultado
    #                        é a média das 5 acurácias.
    # scoring='accuracy'  -> o critério para escolher a melhor combinação é
    #                        a ACURÁCIA, que é o objetivo do trabalho.
    # random_state=42     -> fixa o sorteio das combinações.
    # n_jobs=-1           -> usa todos os núcleos do processador (mais rápido).
    #
    # ATENÇÃO, no cenário COM SMOTE a nota da validação cruzada sai INFLADA:
    # as pacientes sintéticas são misturas de pacientes reais, então uma
    # sintética pode cair na parte de validação enquanto as "parentes" dela
    # estão na parte de treino. O modelo acerta com facilidade algo muito
    # parecido com o que já viu. Por isso a comparação final entre os
    # cenários é feita no TESTE, que tem só pacientes reais.
    busca = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                               param_distributions=decision_tree_grid,
                               n_iter=50,
                               cv=5,
                               scoring='accuracy',
                               random_state=42,
                               n_jobs=-1)
    busca.fit(atributos_treino_norm, classes_treino_cenario)

    print('6.3 Hiperparametrização (RandomizedSearchCV, 50 combinações, cv=5)')
    print(f'    Melhores hiperparâmetros: {busca.best_params_}')
    print(f'    Acurácia média na validação cruzada: {busca.best_score_:.3f}')
    print()

    ###########################################################################
    # 6.4 TREINO DEFINITIVO
    ###########################################################################
    # Com os melhores hiperparâmetros, treinamos a árvore definitiva usando
    # TODO o treino do cenário (já normalizado).
    # **busca.best_params_ "desempacota" o dicionário: é o mesmo que escrever
    # criterion=..., splitter=..., min_samples_split=... à mão.
    modelo = DecisionTreeClassifier(random_state=42, **busca.best_params_)
    modelo.fit(atributos_treino_norm, classes_treino_cenario)

    print('6.4 Árvore definitiva treinada com os melhores hiperparâmetros')
    print()

    ###########################################################################
    # 6.5 AVALIAÇÃO
    ###########################################################################
    # Acurácia = acertos / total de pacientes.
    # - Acurácia no TREINO: mostra o quanto o modelo se ajustou aos dados que
    #   ele já viu. Não serve para medir a qualidade do modelo.
    # - Acurácia no TESTE: pacientes que o modelo NUNCA viu. É A QUE VALE.
    # Se a acurácia de treino for MUITO maior que a de teste, o modelo
    # decorou o treino em vez de aprender o padrão: isso é OVERFITTING.
    classes_preditas_treino = modelo.predict(atributos_treino_norm)
    classes_preditas_teste = modelo.predict(atributos_teste_norm)

    acuracia_treino = accuracy_score(classes_treino_cenario, classes_preditas_treino)
    acuracia_teste = accuracy_score(classes_teste, classes_preditas_teste)

    print('6.5 Avaliação')
    print(f'    Acurácia no treino: {acuracia_treino:.3f}')
    print(f'    Acurácia no teste:  {acuracia_teste:.3f}   <- a que vale')
    print(f'    Acurácia de referência: {acuracia_referencia:.3f}')
    print()

    # A tabela cruzada (crosstab) mostra ONDE o modelo acerta e erra:
    #   linhas = classe REAL; colunas = classe PREVISTA pelo modelo.
    #   A diagonal (real = previsto) são os acertos.
    # Ela revela o que a acurácia sozinha esconde: se o modelo acerta
    # alguma recorrência ou se apenas repete a classe mais comum.
    print('    Tabela cruzada no teste (real x previsto):')
    print(pd.crosstab(classes_teste, classes_preditas_teste, rownames=['real'], colnames=['previsto']))

    # Leitura direta da tabela: quantas recorrências reais o modelo achou.
    # O "chute" da acurácia de referência acha ZERO recorrências.
    recorrencias_reais = (classes_teste == 'recurrence-events').sum()
    recorrencias_acertadas = ((classes_teste == 'recurrence-events') &
                              (classes_preditas_teste == 'recurrence-events')).sum()
    print(f'    Recorrências reais identificadas: {recorrencias_acertadas} de {recorrencias_reais}')
    print()

    # Guardamos o modelo E o normalizador DO MESMO cenário: o modelo só
    # funciona com dados escalados por aquele normalizador específico.
    resultados[cenario] = {'acuracia_teste': acuracia_teste,
                           'modelo': modelo,
                           'normalizador': modelo_normalizador}


###############################################################################
# 7. ESCOLHER O MELHOR CENÁRIO E SALVAR
###############################################################################
print('=' * 70)
print('7. COMPARAÇÃO DOS CENÁRIOS (acurácia no teste)')
print('=' * 70)
print(f'  {"referência (chute)":<20}: {acuracia_referencia:.3f}')
for cenario in resultados:
    print(f'  {cenario:<20}: {resultados[cenario]["acuracia_teste"]:.3f}')
print()

# max() percorre os cenários e devolve aquele cuja acurácia de teste é a
# maior. O key=lambda diz a max() O QUE comparar: para cada cenário c, ela
# olha resultados[c]['acuracia_teste'].
# Em caso de empate, max() fica com o primeiro cenário do dicionário.
melhor_cenario = max(resultados, key=lambda c: resultados[c]['acuracia_teste'])

print(f'Cenário escolhido: {melhor_cenario} '
      f'(acurácia no teste = {resultados[melhor_cenario]["acuracia_teste"]:.3f})')
print()

# Salvamos o modelo e o normalizador DO MESMO cenário com o dump do pickle.
# O pickle grava o objeto Python inteiro num arquivo binário ('wb' = escrita
# binária); a inferência depois o recupera com load.
# O "with open(...)" garante que o arquivo é fechado ao final da gravação.
#
# As colunas de entrada NÃO precisam ser salvas à parte: o scikit-learn
# grava os nomes das colunas usadas no fit no atributo feature_names_in_.
#   - modelo.feature_names_in_       -> as 18 colunas, na ordem do treino;
#   - normalizador.feature_names_in_ -> as 4 colunas numéricas.
# A inferência usa esses atributos para montar a nova paciente igual ao treino.
with open('breast_cancer_tree.model', 'wb') as arquivo:
    dump(resultados[melhor_cenario]['modelo'], arquivo)

with open('normalizador_breast_cancer.model', 'wb') as arquivo:
    dump(resultados[melhor_cenario]['normalizador'], arquivo)

print('=' * 70)
print('ARQUIVOS SALVOS')
print('=' * 70)
print('  breast_cancer_tree.model')
print('  normalizador_breast_cancer.model')
print()
print(f'Colunas do modelo (feature_names_in_): {list(resultados[melhor_cenario]["modelo"].feature_names_in_)}')
print(f'Colunas do normalizador (feature_names_in_): {list(resultados[melhor_cenario]["normalizador"].feature_names_in_)}')
