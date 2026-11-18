# Avaliação Semestral de Data Science: classificador da base Breast Cancer

Avaliação Semestral da matéria de Data Science de 2026.

Classificador por **árvore de decisão** para a base
[Breast Cancer da UCI (id 14)](https://archive.ics.uci.edu/dataset/14/breast+cancer),
dividido em dois módulos:

1. **Treinamento** (`treino_breast_cancer.py`): executa os procedimentos vistos em sala e salva o modelo de **melhor acurácia**.
2. **Inferência** (`inferencia_breast_cancer.py`): recebe uma nova paciente e indica o valor de `Class`.

---

## 1. Objetivo e significado de `Class`

A base reúne 286 pacientes do Instituto de Oncologia do Centro Médico Universitário de Ljubljana (doada à UCI em 1988).

> **Atenção:** `Class` **não** indica se o tumor é maligno ou benigno. **Todas** as pacientes tiveram câncer de mama.
> `Class` indica se o câncer **voltou (recorrência)** depois do tratamento.

| Class | Significado | Pacientes | Proporção |
|---|---|---:|---:|
| `no-recurrence-events` | o câncer **não** voltou | 201 | 70,3% |
| `recurrence-events` | o câncer **voltou** | 85 | 29,7% |
| **Total** | | **286** | 100% |

A base é **desbalanceada** (70/30). Por isso, um "chute" que responde sempre `no-recurrence-events` já acerta cerca de 70%. Esse valor é a **acurácia de referência** que o modelo precisa superar.

O objetivo é treinar um modelo que, a partir dos 9 atributos clínicos, indique se há **indicativo de recorrência**.

---

## 2. Arquivos

| Arquivo | Descrição |
|---|---|
| `breast-cancer.csv` | Base de dados (286 linhas, 10 colunas). Valores ausentes aparecem como `?`. |
| `treino_breast_cancer.py` | Módulo de treinamento (etapas 1 a 7). |
| `inferencia_breast_cancer.py` | Módulo de inferência para uma nova paciente. |
| `requirements.txt` | Bibliotecas com as versões exatas do ambiente, geradas com `pip freeze` (como no roteiro das aulas). |
| `breast_cancer_tree.model` | Árvore de decisão treinada (gerada pelo treino). |
| `normalizador_breast_cancer.model` | `MinMaxScaler` ajustado no treino (gerado pelo treino). |

---

## 3. Como rodar

No PowerShell, dentro da pasta do projeto:

```powershell
# 1. Criar o ambiente virtual (só na primeira vez)
python -m venv .venv

# 2. Ativar o ambiente
.venv\Scripts\Activate

# 3. Instalar as bibliotecas
pip install -r requirements.txt

# 4. Treinar (gera os dois arquivos .model)
python treino_breast_cancer.py

# 5. Inferir a classe de uma nova paciente
python inferencia_breast_cancer.py
```

Para avaliar outra paciente, edite os valores do `DataFrame` na **etapa 2** de `inferencia_breast_cancer.py`.

O `requirements.txt` foi gerado com `pip freeze > requirements.txt` depois da instalação das bibliotecas, como no roteiro das aulas. Ele fixa as versões exatas usadas nos resultados deste README: Python 3.15.2, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, imbalanced-learn 0.14.2 e scipy 1.18.1 (teste de normalidade).

- As versões fixadas exigem um Python recente. Se a instalação falhar num Python mais antigo, instale sem fixar as versões: `pip install pandas numpy scikit-learn imbalanced-learn scipy`.
- Os arquivos `.model` são gravados com `pickle` e dependem da versão do scikit-learn. Se a versão mudar, rode o treino de novo antes da inferência.

---

## 4. Os 9 atributos

A descrição da UCI informa que alguns atributos são **lineares** (têm ordem) e outros são **nominais** (só nomes, sem ordem). O tratamento de cada um segue essa distinção.

| Atributo | Significado | Valores na base | Tipo | Tratamento |
|---|---|---|---|---|
| `age` | Faixa de idade da paciente | `20-29` … `70-79` | Quantitativo (faixa ordenada) | Ponto médio → MinMax |
| `menopause` | Situação de menopausa: antes (`premeno`), aos 40 anos ou mais (`ge40`), antes dos 40 (`lt40`) | `premeno`, `ge40`, `lt40` | Nominal | `get_dummies` |
| `tumor-size` | Maior diâmetro do tumor retirado, em mm | `0-4` … `50-54` | Quantitativo (faixa ordenada) | Ponto médio → MinMax |
| `inv-nodes` | Nº de linfonodos axilares com metástase | `0-2` … `24-26` | Quantitativo (faixa ordenada) | Ponto médio → MinMax |
| `node-caps` | O tumor rompeu a cápsula do linfonodo? | `yes`, `no` (8 ausentes) | Nominal | Moda (`no`) → `get_dummies` |
| `deg-malig` | Grau histológico de malignidade (3 = células mais anormais) | `1`, `2`, `3` | Quantitativo (ordinal, já numérico) | MinMax |
| `breast` | Mama afetada | `left`, `right` | Nominal | `get_dummies` |
| `breast-quad` | Quadrante da mama (com o mamilo como centro) | `left_up`, `left_low`, `right_up`, `right_low`, `central` (1 ausente) | Nominal | Moda (`left_low`) → `get_dummies` |
| `irradiat` | Fez radioterapia? | `yes`, `no` | Nominal | `get_dummies` |

> No site da UCI, `breast-quad` aparece com **hífen** (`left-low`). No arquivo `breast-cancer.csv` está com **underscore** (`left_low`). O código usa o formato do arquivo.

---

## 5. Módulo de treinamento, passo a passo (`treino_breast_cancer.py`)

### Etapa 1: Carregar
`pd.read_csv('breast-cancer.csv', sep=',', na_values='?')`. O `na_values='?'` transforma todo `?` em `NaN`, o marcador de ausência do pandas. Sem ele, `?` seria lido como uma categoria válida (`node-caps` teria os valores `yes`, `no` e `?`).

### Etapa 2: Valores ausentes
Há 9 ausentes: 8 em `node-caps` e 1 em `breast-quad`.

- **Por que preencher e não apagar?** Apagar perderia 9 pacientes de uma base de apenas 286, junto com as outras 8 informações de cada uma.
- **Por que a moda?** As duas colunas são categóricas (texto), e não existe média de texto. O valor mais frequente (`.mode()[0]`) é a escolha natural: `no` em `node-caps` e `left_low` em `breast-quad`.

### Etapa 3: Separar atributos e classe
`dados_atributos` recebe as 9 colunas de entrada e `dados_classes` recebe `Class`. O `value_counts()` confirma o desbalanceamento: 201 × 85.

### Etapa 4: Quantitativos × categóricos
Esta é a decisão mais importante do pré-processamento.

**Quantitativos.** `age`, `tumor-size` e `inv-nodes` chegam como faixas de texto (`'50-59'`). Cada faixa vira o seu **ponto médio**:

```
'50-59' → split('-') → ['50', '59'] → (50 + 59) / 2 = 54.5
```

| Atributo | Conversão |
|---|---|
| `age` | 20-29 → 24.5, 30-39 → 34.5, 40-49 → 44.5, 50-59 → 54.5, 60-69 → 64.5, 70-79 → 74.5 |
| `tumor-size` | 0-4 → 2, 5-9 → 7, 10-14 → 12, … , 50-54 → 52 |
| `inv-nodes` | 0-2 → 1, 3-5 → 4, 6-8 → 7, 9-11 → 10, 12-14 → 13, 15-17 → 16, 24-26 → 25 |

O ponto médio **preserva a ordem** das faixas, e a árvore pode aprender regras como "tumor-size > 30" de uma só vez. Com `get_dummies`, cada faixa viraria uma coluna 0/1 independente, e o modelo perderia a informação de que 50-59 vem depois de 40-49.

`deg-malig` (1, 2, 3) já é numérico e ordenado. As quatro colunas numéricas são `['age', 'tumor-size', 'inv-nodes', 'deg-malig']`.

**Análise monovariada.** Como visto em aula, cada atributo quantitativo é descrito separadamente, na base completa, com média, mediana, desvio padrão, coeficiente de variação (CV = desvio padrão / média × 100) e o teste de normalidade de Shapiro-Wilk (`stats.shapiro`, do scipy). A hipótese inicial do teste é que os dados são normais; com p-valor ≤ 0.05, ela é rejeitada. A análise só descreve os dados e não altera nada.

| Atributo | Média | Mediana | Desvio padrão | CV | Shapiro (p-valor) | Conclusão |
|---|---:|---:|---:|---:|---:|---|
| `age` | 51.14 | 54.50 | 10.12 | 19.8% | 4.23e-12 | não normal |
| `tumor-size` | 26.41 | 27.00 | 10.53 | 39.9% | 1.52e-05 | não normal |
| `inv-nodes` | 2.57 | 1.00 | 3.45 | 134.1% | 4.65e-27 | não normal |
| `deg-malig` | 2.05 | 2.00 | 0.74 | 36.0% | 4.57e-18 | não normal |

Nenhum dos quatro atributos é normal, por dois motivos:

- eles têm poucos valores distintos (6 faixas de idade, 3 graus de malignidade), e a curva normal é contínua;
- `inv-nodes` é muito assimétrico: 213 das 286 pacientes estão na faixa 0-2, por isso a média (2.57) fica bem acima da mediana (1.00) e o CV passa de 100%.

Isso justifica as escolhas de tratamento:

- o `MinMaxScaler` usa só o mínimo e o máximo de cada coluna e não depende de os dados serem normais;
- a árvore de decisão também não exige normalidade, porque só compara valores com pontos de corte;
- se houvesse ausentes numéricos, a mediana seria preferível à média. Nesta base, os ausentes estão só em colunas categóricas e foram preenchidos com a moda.

**Categóricos nominais.** `menopause`, `node-caps`, `breast`, `breast-quad` e `irradiat` não têm ordem (`left` não é "maior" que `right`). Eles passam por `pd.get_dummies(..., dummy_na=False, dtype=int)`, que cria uma coluna 0/1 para cada valor. Em aula, esse passo é a **normalização dos dados categóricos**.

**Junção.** `join` dos 4 numéricos com as 14 dummies (3 + 2 + 2 + 5 + 2) = **18 colunas**.

### Etapa 5: Treino e teste
`train_test_split(..., test_size=0.3, random_state=42, stratify=dados_classes)` gera **200 pacientes de treino** e **86 de teste**.

- O teste fica guardado e **só é usado no final**: medir a acurácia nos próprios dados de treino engana, porque a árvore pode decorar as pacientes.
- `stratify` mantém a proporção 70/30 nos dois conjuntos.
- **Acurácia de referência** = `(classes_teste == 'no-recurrence-events').mean()` = **0.698** (60 de 86). É a acurácia de um "chute" que sempre responde a classe mais comum.

### Etapa 6: Dois cenários (com e sem balanceamento)
O SMOTE faz parte do roteiro da aula, mas pode **baixar a acurácia geral**: com as classes igualadas no treino, o modelo passa a priorizar a classe rara. Como o objetivo é a melhor acurácia, os dois cenários são treinados e comparados **no mesmo teste**.

**Grade de hiperparâmetros** (90 combinações):

| Hiperparâmetro | Valores | Para que serve |
|---|---|---|
| `criterion` | `gini`, `entropy`, `log_loss` | Medida de "mistura" das classes usada para escolher cada divisão |
| `splitter` | `best`, `random` | Escolhe o melhor corte ou o melhor entre cortes sorteados |
| `min_samples_split` | 2, 9, 16, … , 100 (15 valores) | Mínimo de pacientes num nó para dividi-lo. Começa em 2 porque o scikit-learn não aceita 1; vai até 100 porque árvores mais simples generalizam melhor em bases pequenas |

Para cada cenário:

- **6.1 Balancear.** Só no cenário com balanceamento: `SMOTE(random_state=42)` cria pacientes sintéticas de recorrência **somente no treino** (141/59 → 141/141). O teste mantém a proporção real. As dummies sintéticas podem ficar com valores entre 0 e 1, o que a árvore tolera. No outro cenário, usamos cópias do treino.
- **6.2 Normalizar.** `modelo_normalizador = normalizador.fit(...)` (`MinMaxScaler`) **só nas 4 colunas numéricas do treino** do cenário, e `transform` no treino e no teste. Só os atributos são normalizados, nunca a classe; as dummies já estão entre 0 e 1. Nunca se faz `fit_transform` de novo. Treino, avaliação e inferência usam **sempre** dados normalizados. No `iris_classificador` da aula, a árvore era treinada sem normalizar e avaliada normalizada, o que é um erro.
- **6.3 Hiperparametrização.** `RandomizedSearchCV` com 50 combinações, validação cruzada de 5 partes (`cv=5`) e `scoring='accuracy'`. No cenário com SMOTE, a nota da validação cruzada sai **inflada**: pacientes sintéticas parecidas caem em partes diferentes, e o modelo é validado em pacientes quase iguais às que viu no treino. Por isso a comparação final é feita no teste.
- **6.4 Treino definitivo.** `DecisionTreeClassifier(random_state=42, **best_params_)` treinado com todo o treino do cenário.
- **6.5 Avaliação.** Acurácia no treino e no teste (a do teste é a que vale) e a tabela cruzada real × previsto. Treino muito acima do teste indica **overfitting**.

### Etapa 7: Escolher e salvar
`max(resultados, key=lambda c: resultados[c]['acuracia_teste'])` escolhe o cenário de maior acurácia no teste. O modelo e o normalizador **do mesmo cenário** são salvos com `dump`. As 18 colunas de entrada (e as 4 normalizadas) ficam gravadas em `feature_names_in_`, e a inferência usa esse atributo para montar a nova paciente.

---

## 6. Resultado obtido

Saída real de `python treino_breast_cancer.py`:

| Cenário | Melhores hiperparâmetros | Validação cruzada | Acurácia treino | **Acurácia teste** |
|---|---|---:|---:|---:|
| Referência (chute) | — | — | — | **0.698** |
| Com SMOTE | `splitter='random'`, `min_samples_split=9`, `criterion='entropy'` | 0.753 | 0.897 | **0.593** |
| Sem SMOTE | `splitter='best'`, `min_samples_split=37`, `criterion='gini'` | 0.760 | 0.800 | **0.698** |

**Cenário escolhido: `sem_balanceamento`** (acurácia no teste = 0.698).

Tabelas cruzadas no teste (86 pacientes: 60 sem recorrência, 26 com recorrência):

| Com SMOTE | previsto: no-rec | previsto: rec |
|---|---:|---:|
| **real: no-rec** | 45 | 15 |
| **real: rec** | 20 | 6 |

| Sem SMOTE (escolhido) | previsto: no-rec | previsto: rec |
|---|---:|---:|
| **real: no-rec** | 55 | 5 |
| **real: rec** | 21 | 5 |

| Chute (referência) | previsto: no-rec | previsto: rec |
|---|---:|---:|
| **real: no-rec** | 60 | 0 |
| **real: rec** | 26 | 0 |

---

## 7. Como interpretar o resultado

**Por que o SMOTE piorou a acurácia?**
Com as classes igualadas no treino, a árvore passou a responder "recorrência" com muito mais frequência: 21 vezes no teste, contra 10 sem SMOTE. Ela achou só uma recorrência a mais (6 contra 5), mas criou 15 falsos alarmes (contra 5). Como 70% das pacientes reais não têm recorrência, cada falso alarme custa acurácia. Além disso, o modelo com SMOTE ficou mais complexo (`min_samples_split=9`) e se ajustou ao treino sintético: 0.897 no treino contra 0.593 no teste, um sinal claro de overfitting. A validação cruzada (0.753) não detectou o problema porque estava inflada pelas pacientes sintéticas.

**Por que o modelo escolhido empata com a referência, e o que ele faz de diferente?**
Os dois acertam 60 de 86 pacientes (0.698), mas acertam **pacientes diferentes**:

- o chute acerta as 60 pacientes sem recorrência e **nenhuma** recorrência;
- o modelo acerta 55 sem recorrência **e 5 das 26 recorrências**.

O modelo troca 5 falsos alarmes por 5 recorrências identificadas. Na acurácia o saldo é zero, mas o modelo **distingue** as pacientes, coisa que o chute não faz. Na validação cruzada do treino, ele também ficou acima da classe majoritária (0.760 contra 141/200 = 0.705).

Inspecionando a árvore salva, a regra que mais pesa é: **grau de malignidade 3 (`deg-malig`) com linfonodos comprometidos acima da faixa 0-2 (`inv-nodes`) → recorrência**. No treino, esse grupo tinha 22 recorrências e 5 não recorrências. Essa combinação faz sentido clínico: tumor mais agressivo e já espalhado para os linfonodos.

**Por que isso é limitação da base, e não erro do código?**

- **A base é pequena.** São 286 pacientes, e o teste tem só 86: cada paciente vale 1,2 ponto percentual de acurácia. A diferença entre 0.698 e 0.76 equivale a cerca de 5 pacientes.
- **Os atributos vêm em faixas largas.** A idade vem em décadas, o tumor em faixas de 5 mm, e `inv-nodes` = `0-2` junta 0, 1 e 2 linfonodos (o grupo de 213 das 286 pacientes). Pacientes diferentes ficam com valores idênticos.
- **Faltam informações.** A base não traz dados que hoje se sabe que influenciam a recorrência, como receptores hormonais, HER2 e o tratamento sistêmico recebido. Com esses 9 atributos, há pacientes com perfil idêntico e desfechos diferentes.
- **A literatura confirma o teto baixo.** Os estudos citados na própria documentação da UCI (arquivo `breast-cancer.names`) relatam acurácias entre **65% e 78%** nesta base.

O pré-processamento, a separação treino/teste, a normalização e a avaliação foram feitos corretamente. O resultado modesto reflete a informação disponível nos dados.

---

## 8. Módulo de inferência, passo a passo (`inferencia_breast_cancer.py`)

**Pré-requisito:** rodar o treino antes, para gerar os dois arquivos `.model`.

1. **Carregar** `breast_cancer_tree.model` e `normalizador_breast_cancer.model` com `load`. As colunas esperadas vêm de `feature_names_in_`: `colunas_modelo` (18 colunas, na ordem do treino) e `colunas_num` (4 numéricas). Se os arquivos não existirem, o programa avisa que é preciso rodar o treino.
2. **Nova paciente** num `DataFrame`, **no formato original da base** (texto e faixas). Exemplo:

   | age | menopause | tumor-size | inv-nodes | node-caps | deg-malig | breast | breast-quad | irradiat |
   |---|---|---|---|---|---|---|---|---|
   | `40-49` | `premeno` | `30-34` | `6-8` | `yes` | `3` | `left` | `left_low` | `yes` |

3. **Transformar exatamente como no treino:**
   - faixas → ponto médio, com o mesmo laço do treino;
   - `get_dummies` nos 5 atributos nominais (uma paciente gera só uma dummy por atributo);
   - **conferência:** se algum valor gerar uma dummy que não existe no modelo (ex.: `left-low` com hífen), o programa para com `ERRO: valores que não existem na base de treino: [...]`. Sem essa conferência, o `reindex` descartaria a coluna em silêncio e o modelo responderia com uma entrada errada;
   - `join` e `reindex(columns=colunas_modelo, fill_value=0)`, que coloca as 18 colunas na ordem do treino e preenche com 0 as dummies que a paciente não gerou.
4. **Normalizar** as 4 colunas numéricas com `modelo_normalizador.transform` (**nunca** `fit`).
5. **Prever** com `predict` e mostrar a classe e o indicativo.

**Testes realizados:**

| Paciente | Resultado |
|---|---|
| Exemplo acima | `recurrence-events`: **Indicativo: RECORRÊNCIA do câncer** |
| Mesma paciente com `inv-nodes='0-2'`, `node-caps='no'`, `deg-malig=1` | `no-recurrence-events`: **Indicativo: NÃO recorrência do câncer** |
| Mesma paciente com `breast-quad='left-low'` (hífen) | `ERRO: valores que não existem na base de treino: ['breast-quad_left-low']` |

> Resultado de um modelo estatístico para fins de estudo. **Não substitui avaliação médica.**

---

## 9. LGPD

A Lei Geral de Proteção de Dados (Lei nº 13.709/2018) se aplica de forma diferente ao estudo acadêmico e a um eventual uso com pacientes reais.

| Ponto | Artigo | Aplicação neste trabalho |
|---|---|---|
| Dado de saúde é **dado pessoal sensível** | art. 5º, II | Os 9 atributos e a `Class` são informações de saúde. |
| **Dado anonimizado** não é dado pessoal, salvo se a anonimização puder ser revertida | art. 5º, III e art. 12 | A base da UCI não tem nome, documento ou outro identificador das pacientes. |
| Base legal do **estudo** | art. 7º, IV (para dados sensíveis, art. 11, II, "c") | Realização de estudos por órgão de pesquisa, garantida, sempre que possível, a anonimização. É a situação deste trabalho. |
| Base legal para uso **com pacientes reais** | art. 7º, VIII (para dados sensíveis, art. 11, II, "f") | Tutela da saúde, exclusivamente em procedimento feito por profissionais de saúde, serviços de saúde ou autoridade sanitária. |
| **Encarregado (DPO)** | art. 41 | Num uso real, o controlador precisa indicar um encarregado, com identidade e contato divulgados publicamente. |
| **Princípios** | art. 6º | Finalidade (estudo), necessidade (só os 9 atributos clínicos, sem identificadores), segurança (acesso restrito aos dados) e não discriminação (o resultado não pode ser usado para discriminar pacientes). |

O modelo é um exercício acadêmico e não deve ser usado para decisões clínicas.
