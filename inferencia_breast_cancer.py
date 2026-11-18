###############################################################################
# MÓDULO DE INFERÊNCIA - BASE BREAST CANCER (UCI, id 14)
# Disciplina: Data Science - Avaliação Semestral
#
# OBJETIVO
#   Receber os dados de uma NOVA paciente e indicar o valor do atributo
#   "Class" usando a árvore de decisão treinada por treino_breast_cancer.py.
#
# O QUE SIGNIFICA "Class"?
#   ATENÇÃO: Class NÃO diz se o tumor é maligno ou benigno. Todas as
#   pacientes da base tiveram câncer de mama. Class indica se o câncer
#   VOLTOU (recorrência) depois do tratamento:
#     - no-recurrence-events -> indicativo de que o câncer NÃO volta;
#     - recurrence-events    -> indicativo de RECORRÊNCIA do câncer.
#
# PRÉ-REQUISITO
#   Rodar ANTES o treino (python treino_breast_cancer.py). Ele gera os dois
#   arquivos que este módulo carrega:
#     - breast_cancer_tree.model         (árvore de decisão)
#     - normalizador_breast_cancer.model (MinMaxScaler do treino)
#
# LGPD (Lei Geral de Proteção de Dados, Lei nº 13.709/2018)
#   - Dados de saúde são DADOS PESSOAIS SENSÍVEIS (art. 5º, II).
#   - A base da UCI usada no treino é ANONIMIZADA: não identifica nenhuma
#     paciente. Dados anonimizados não são considerados dados pessoais, salvo
#     se a anonimização puder ser revertida (art. 12).
#   - Este trabalho é um ESTUDO ACADÊMICO. A base legal correspondente é a
#     realização de estudos por órgão de pesquisa, garantida, sempre que
#     possível, a anonimização (art. 7º, IV; para dados sensíveis, a mesma
#     hipótese está no art. 11, II, "c").
#   - Usar o modelo com pacientes REAIS mudaria a finalidade e exigiria:
#       * outra base legal, como a tutela da saúde, exclusivamente em
#         procedimento feito por profissionais ou serviços de saúde
#         (art. 7º, VIII; para dados sensíveis, art. 11, II, "f");
#       * um ENCARREGADO (DPO) indicado pelo controlador (art. 41);
#       * respeito aos princípios do art. 6º, entre eles: finalidade,
#         necessidade (só os 9 atributos clínicos, sem nome, CPF ou outro
#         identificador), segurança e não discriminação.
###############################################################################

import pandas as pd
from pickle import load

# Configuração só de exibição: mostra todas as colunas sem cortar.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)


###############################################################################
# 1. CARREGAR O MODELO E O NORMALIZADOR
###############################################################################
# load (do pickle) recupera os objetos gravados pelo treino com dump.
# 'rb' = leitura binária. Os dois arquivos são do MESMO cenário escolhido no
# treino: o modelo só funciona com dados escalados por esse normalizador.
#
# Se os arquivos não existirem, o treino ainda não foi rodado; em vez do erro
# técnico do Python, mostramos uma orientação clara.
try:
    with open('breast_cancer_tree.model', 'rb') as arquivo:
        breast_cancer_tree = load(arquivo)
    with open('normalizador_breast_cancer.model', 'rb') as arquivo:
        modelo_normalizador = load(arquivo)
except FileNotFoundError:
    print('ERRO: arquivos .model não encontrados.')
    print('Rode primeiro: python treino_breast_cancer.py')
    raise SystemExit(1)

# O scikit-learn guarda no próprio objeto os nomes das colunas usadas no fit
# (feature_names_in_). Assim, a inferência não precisa repetir essas listas
# "na mão" e fica sempre coerente com o que foi treinado:
#   colunas_modelo -> as 18 colunas, NA ORDEM em que a árvore foi treinada;
#   colunas_num    -> as 4 colunas numéricas que o normalizador conhece.
colunas_modelo = list(breast_cancer_tree.feature_names_in_)
colunas_num = list(modelo_normalizador.feature_names_in_)

print('=' * 70)
print('1. MODELO CARREGADO')
print('=' * 70)
print(f'Colunas do modelo ({len(colunas_modelo)}): {colunas_modelo}')
print(f'Colunas normalizadas: {colunas_num}')
print()


###############################################################################
# 2. NOVA PACIENTE (no formato ORIGINAL da base)
###############################################################################
# A paciente é informada EXATAMENTE como aparece no arquivo breast-cancer.csv
# (textos e faixas). Todo o tratamento é feito pelo código na etapa 3.
#
# Valores aceitos em cada atributo (os que existem na base de treino):
#   age         -> faixa de idade: '20-29', '30-39', '40-49', '50-59',
#                  '60-69', '70-79'
#   menopause   -> 'premeno' (antes da menopausa), 'ge40' (menopausa aos 40
#                  anos ou mais), 'lt40' (menopausa antes dos 40)
#   tumor-size  -> faixa do tamanho do tumor em mm: '0-4', '5-9', '10-14',
#                  '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
#                  '45-49', '50-54'
#   inv-nodes   -> faixa de linfonodos axilares com câncer: '0-2', '3-5',
#                  '6-8', '9-11', '12-14', '15-17', '24-26'
#   node-caps   -> o tumor rompeu a cápsula do linfonodo? 'yes' ou 'no'
#   deg-malig   -> grau de malignidade: 1, 2 ou 3 (NÚMERO, sem aspas)
#   breast      -> mama afetada: 'left' ou 'right'
#   breast-quad -> quadrante da mama: 'left_up', 'left_low', 'right_up',
#                  'right_low', 'central'
#   irradiat    -> fez radioterapia? 'yes' ou 'no'
#
# CUIDADO com breast-quad: no site da UCI os valores aparecem com HÍFEN
# ('left-low'), mas no arquivo usado no treino estão com UNDERSCORE
# ('left_low'). Use SEMPRE o underscore. Um valor com hífen não existe no
# treino e é barrado pela conferência da etapa 3.
#
# pd.DataFrame([{...}]): uma lista com UM dicionário gera uma tabela com UMA
# linha, ou seja, uma paciente (mesmo formato usado em aula). Cada chave é o
# nome exato da coluna na base.
nova_paciente = pd.DataFrame([{
    'age': '40-49',             # faixa de idade
    'menopause': 'premeno',     # situação de menopausa
    'tumor-size': '30-34',      # faixa do tamanho do tumor (mm)
    'inv-nodes': '6-8',         # faixa de linfonodos axilares com câncer
    'node-caps': 'yes',         # o tumor rompeu a cápsula do linfonodo?
    'deg-malig': 3,             # grau de malignidade (1, 2 ou 3)
    'breast': 'left',           # mama afetada
    'breast-quad': 'left_low',  # quadrante da mama (com underscore)
    'irradiat': 'yes',          # fez radioterapia?
}])

print('=' * 70)
print('2. NOVA PACIENTE (formato original)')
print('=' * 70)
print(nova_paciente)
print()


###############################################################################
# 3. TRANSFORMAR EXATAMENTE COMO NO TREINO
###############################################################################
# A árvore aprendeu com dados num formato específico: faixas convertidas em
# ponto médio, dummies 0/1 e 18 colunas numa ordem fixa. A nova paciente
# precisa passar EXATAMENTE pelas mesmas transformações; qualquer diferença
# faria o modelo comparar a paciente com regras que não se aplicam a ela.

# -----------------------------------------------------------------------------
# 3.1 Faixas -> ponto médio (o MESMO laço do treino)
# -----------------------------------------------------------------------------
# '40-49' -> split('-') -> ['40', '49'] -> (40 + 49) / 2 = 44.5
colunas_faixas = ['age', 'tumor-size', 'inv-nodes']
for coluna_nome in colunas_faixas:
    coluna = nova_paciente[coluna_nome]
    limites = coluna.str.split('-')
    inicio = limites.str[0].astype(int)
    fim = limites.str[1].astype(int)
    nova_paciente[coluna_nome] = (inicio + fim) / 2

# -----------------------------------------------------------------------------
# 3.2 get_dummies nos 5 atributos nominais (igual ao treino)
# -----------------------------------------------------------------------------
# Uma paciente tem UM valor por atributo, então o get_dummies gera só UMA
# dummy por atributo (ex.: só 'breast_left', e não 'breast_right').
# As dummies que faltam são completadas com 0 no reindex do passo 3.4.
colunas_cat = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']
nova_paciente_dummies = pd.get_dummies(nova_paciente[colunas_cat], dummy_na=False, dtype=int)

# -----------------------------------------------------------------------------
# 3.3 CONFERÊNCIA: todo valor informado existe na base de treino?
# -----------------------------------------------------------------------------
# Se for digitado um valor que o modelo nunca viu (ex.: 'left-low' com hífen,
# ou 'Yes' com maiúscula), o get_dummies cria uma coluna que NÃO EXISTE no
# modelo (ex.: 'breast-quad_left-low').
# SEM esta conferência, o reindex do passo 3.4 DESCARTARIA essa coluna EM
# SILÊNCIO e preencheria todas as dummies do atributo com 0: a paciente
# ficaria "sem quadrante", e o modelo daria uma resposta sem avisar que a
# entrada estava errada. Por isso paramos o programa com uma mensagem clara.
colunas_desconhecidas = [c for c in nova_paciente_dummies.columns if c not in colunas_modelo]

if colunas_desconhecidas:
    print('=' * 70)
    print(f'ERRO: valores que não existem na base de treino: {colunas_desconhecidas}')
    print('Orientação: confira a grafia dos valores na etapa 2 (letras minúsculas;')
    print("em breast-quad use underscore, ex.: 'left_low', e não 'left-low').")
    print('=' * 70)
    raise SystemExit(1)

# -----------------------------------------------------------------------------
# 3.4 Junção e alinhamento com as colunas do modelo
# -----------------------------------------------------------------------------
# join -> junta as 4 numéricas com as dummies, como no treino.
# reindex(columns=colunas_modelo, fill_value=0):
#   - coloca as colunas NA MESMA ORDEM do treino (o modelo exige isso);
#   - cria as dummies que a paciente não gerou, com valor 0
#     (ex.: breast_right = 0 para uma paciente com breast = 'left').
nova_paciente_tratada = nova_paciente[colunas_num].join(nova_paciente_dummies)
nova_paciente_tratada = nova_paciente_tratada.reindex(columns=colunas_modelo, fill_value=0)

print('=' * 70)
print('3. PACIENTE TRANSFORMADA (ponto médio + dummies, 18 colunas)')
print('=' * 70)
print(nova_paciente_tratada)
print()


###############################################################################
# 4. NORMALIZAR (só transform, NUNCA fit)
###############################################################################
# Usamos o normalizador SALVO, com o mínimo e o máximo aprendidos no treino.
# Só transform: um novo fit com uma única paciente daria mínimo = máximo e
# tornaria a escala sem sentido; e a árvore só entende a escala do treino.
# Só as colunas_num são normalizadas; as dummies já estão em 0 ou 1.
nova_paciente_tratada[colunas_num] = modelo_normalizador.transform(nova_paciente_tratada[colunas_num])

print('=' * 70)
print('4. PACIENTE NORMALIZADA')
print('=' * 70)
print(nova_paciente_tratada.round(3))
print()


###############################################################################
# 5. INFERIR A CLASSE
###############################################################################
# predict devolve uma lista com uma resposta por paciente; como aqui há uma
# só, pegamos a primeira posição [0].
classe_inferida = breast_cancer_tree.predict(nova_paciente_tratada)[0]

print('=' * 70)
print('5. RESULTADO DA INFERÊNCIA')
print('=' * 70)
print(f'Class inferida: {classe_inferida}')
if classe_inferida == 'recurrence-events':
    print('Indicativo: RECORRÊNCIA do câncer')
else:
    print('Indicativo: NÃO recorrência do câncer')
print()
print('Resultado de um modelo estatístico para fins de estudo. Não substitui avaliação médica.')
