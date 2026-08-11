# Relatório Técnico — Classificação e Agrupamento com o Iris Dataset

**Disciplina:** Aprendizado de Máquina  
**Tema:** a) Iris Dataset  
**Algoritmos:** Árvore de Decisão (Parte 1) e K-Means (Parte 2)

---

## 1. Introdução

Este relatório apresenta a implementação de um modelo de classificação supervisionado (Árvore de Decisão) e de um algoritmo de agrupamento não supervisionado (K-Means), aplicados ao clássico *Iris Dataset*. O objetivo é consolidar a prática de pré-processamento, ajuste de hiperparâmetros, avaliação por métricas e comparação entre grupos formados e classes reais.

---

## 2. Descrição do problema e do dataset

### 2.1 Problema

O problema é de **classificação multiclasse**: identificar a espécie de uma flor Iris a partir de medições morfométricas de sépalas e pétalas. As três espécies-alvo são:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

### 2.2 Dataset

| Item | Descrição |
|------|-----------|
| Fonte | UCI Machine Learning Repository / scikit-learn |
| Amostras | 150 (50 por espécie) |
| Atributos | 4 (todos numéricos, em cm) |
| Alvo | Espécie (3 classes) |

**Atributos de entrada:**

1. Comprimento da sépala (*sepal length*)
2. Largura da sépala (*sepal width*)
3. Comprimento da pétala (*petal length*)
4. Largura da pétala (*petal width*)

O Iris é balanceado (50 amostras por classe), sem valores ausentes e com atributos já numéricos, o que o torna adequado como problema introdutório de classificação e de agrupamento.

---

## 3. Etapas de pré-processamento

Foram realizadas as seguintes etapas (comuns à Parte 1 e, na essência, à Parte 2):

1. **Carregamento dos dados** via `sklearn.datasets.load_iris`.
2. **Verificação de valores ausentes:** nenhum valor ausente encontrado.
3. **Codificação de variáveis categóricas:** desnecessária, pois todos os atributos de entrada já são numéricos. O rótulo (espécie) já vem codificado como 0, 1 e 2.
4. **Normalização / padronização:** não aplicada. Árvores de decisão não dependem de escala; a Parte 2 manteve o mesmo pré-processamento, conforme o enunciado.
5. **Divisão treino/teste (apenas Parte 1):** 70% treino e 30% teste, com `random_state=42` e estratificação por classe (`stratify=y`), preservando a proporção das três espécies em cada conjunto (105 treino / 45 teste).

Na **Parte 2 (K-Means)** utilizou-se o conjunto completo de 150 amostras, pois o aprendizado é não supervisionado e a variável-alvo **não entra** no treinamento — apenas na comparação final.

---

## 4. Parte 1 — Árvore de Decisão

### 4.1 Modelo e configurações

Foi utilizado o `DecisionTreeClassifier` do scikit-learn.

Foram comparadas **duas configurações**:

| Configuração | Descrição |
|--------------|-----------|
| **Baseline** | Árvore sem restrição de profundidade (`max_depth=None`), critério padrão *gini*, `random_state=42`. |
| **Ajustada** | Melhores hiperparâmetros obtidos por *Grid Search* com validação cruzada (5 folds). |

**Hiperparâmetros testados no GridSearchCV:**

- `max_depth`: 2, 3, 4, 5, None  
- `min_samples_split`: 2, 5, 10  
- `criterion`: gini, entropy  
- Métrica de otimização: acurácia  

**Melhores parâmetros encontrados:**

```text
criterion = 'gini'
max_depth = 3
min_samples_split = 2
```

Com isso, a árvore ajustada ficou com **profundidade 3** e **5 folhas**, mais compacta que a baseline (profundidade 5 e 8 folhas).

### 4.2 Justificativa das decisões

- **Árvore de decisão:** modelo interpretável, adequado a dados tabulares e a problemas multiclasse; exige pouco pré-processamento.
- **Sem normalização:** o critério de divisão da árvore não se baseia em distância euclidiana.
- **Divisão estratificada 70/30:** garante representatividade das classes e um conjunto de teste suficiente para métricas.
- **Limitar `max_depth`:** reduz o risco de *overfitting*, comum em árvores irrestritas.
- **Grid Search + CV:** busca sistemática e justa dos hiperparâmetros sem “vazar” o conjunto de teste na escolha final.

### 4.3 Metodologia de avaliação

No conjunto de teste foram calculadas:

- Acurácia  
- Precisão, Recall e F1-score (média *macro*, adequada a multiclasse)  
- Matriz de confusão  
- Relatório de classificação por classe  

### 4.4 Resultados

#### Comparação global (conjunto de teste)

| Métrica   | Baseline | Modelo ajustado |
|-----------|----------|-----------------|
| Acurácia  | 0,9333   | **0,9778**      |
| Precisão  | 0,9444   | **0,9792**      |
| Recall    | 0,9333   | **0,9778**      |
| F1-score  | 0,9327   | **0,9778**      |

#### Matriz de confusão — modelo ajustado

|            | Pred. setosa | Pred. versicolor | Pred. virginica |
|------------|--------------|------------------|-----------------|
| **setosa**     | 15 | 0  | 0  |
| **versicolor** | 0  | 14 | 1  |
| **virginica**  | 0  | 0  | 15 |

Interpretação: o modelo classificou corretamente todas as *setosa* e *virginica* do teste; errou apenas **1** amostra de *versicolor* (prevista como *virginica*).

#### Acurácia na validação cruzada (treino)

Melhor acurácia média em 5 folds: **0,9524**.

Os gráficos da matriz de confusão e da árvore treinada estão no arquivo `resultados_parte1.png`.

### 4.5 Discussão crítica e melhorias

**O que melhorou e por quê**

A baseline already acertava bem (~93%), mas crescia até profundidade 5, o que pode incorporar ruído. Com `max_depth=3`, o modelo ficou mais genérico, com menos folhas, e subiu a acurácia no teste para ~98%. A melhoria objetiva em todas as métricas (acurácia, precisão, recall e F1) indica que o ajuste de hiperparâmetros foi favorável neste dataset.

**Por que o Iris facilita bons resultados**

- Classes balanceadas  
- Poucos atributos relevantes (especialmente medidas de pétala)  
- *Setosa* linearmente bem separável das demais  

**Limitações**

- Dataset pequeno (150 amostras); métricas no teste (45 amostras) têm variabilidade.  
- Em problemas reais, com muitos atributos e classes sobrepostas, o ganho de uma árvore simples pode ser menor.  
- Outras melhorias possíveis (não exploradas aqui por simplicidade): *ensemble* (Random Forest), poda por custo-complexidade, ou inclusão de validação com mais divisões aleatórias.

---

## 5. Parte 2 — Agrupamento com K-Means

### 5.1 Configuração

| Item | Escolha |
|------|---------|
| Algoritmo | K-Means (`sklearn.cluster.KMeans`) |
| Número de clusters (k) | **3** (mesmo número de espécies reais) |
| Entrada | Apenas os 4 atributos numéricos (sem o rótulo) |
| Pré-processamento | Mesmo da Parte 1 (sem normalização) |
| `random_state` | 42 |
| `n_init` | 10 |

### 5.2 Resultados do agrupamento

**Tamanhos dos clusters:** 62, 50 e 38 amostras.

#### Tabela de contingência (classe real × cluster)

| Classe real | Cluster 0 | Cluster 1 | Cluster 2 |
|-------------|-----------|-----------|-----------|
| setosa      | 0         | **50**    | 0         |
| versicolor  | **48**    | 0         | 2         |
| virginica   | 14        | 0         | **36**    |

#### Espécie predominante por cluster

| Cluster | Classe dominante | Pureza aproximada |
|---------|------------------|-------------------|
| 0 | versicolor | 77% (48/62) |
| 1 | setosa     | 100% (50/50) |
| 2 | virginica  | 95% (36/38) |

Após mapear cada cluster para a espécie majoritária, **134 de 150** amostras (≈ **89,3%**) coincidem com o rótulo real.

O gráfico dos clusters (ao lado das classes reais, usando comprimento e largura da pétala) está em `resultados_parte2.png`.

### 5.3 Os clusters se aproximaram das classes reais? Em que medida?

**Sim, em boa medida (~89%), mas de forma desigual por espécie.**

- A *setosa* formou um cluster **totalmente puro** (as 50 amostras no Cluster 1).  
- *Versicolor* e *virginica* ficaram razoavelmente separadas, porém com **mistura**: 14 virginicas entraram no cluster majoritariamente versicolor e 2 versicolors no de virginica.

### 5.4 Por que isso aconteceu?

O K-Means agrupa por **proximidade no espaço dos atributos**, sem conhecer as espécies. No Iris:

1. A *setosa* tem medidas de pétala bem distintas → fácil de isolar.
2. *Versicolor* e *virginica* se sobrepõem morfologicamente (principalmente em pétala) → fronteira menos clara → alguns pontos caem no grupo “vizinho”.
3. Como o algoritmo minimiza distância aos centroides (e não erro de classificação), ele não “corrige” confusões entre classes visualmente próximas.

Em resumo: o agrupamento **reflete a estrutura geométrica real dos dados**, que só parcialmente coincide com as três espécies rotuladas.

---

## 6. Conclusões finais

1. A **Árvore de Decisão** resolveu bem o problema de classificação do Iris. Com ajuste de hiperparâmetros (`max_depth=3`, *gini*), o modelo alcançou cerca de **98% de acurácia** no teste, superando a baseline sem restrições (~93%).
2. As métricas de precisão, recall e F1 acompanharam a melhoria, e a matriz de confusão mostrou apenas um erro residual entre *versicolor* e *virginica*.
3. O **K-Means** recuperou grupos próximos às classes, especialmente a *setosa*, com alinhamento global de cerca de **89%**. As confusões restantes decorrem da sobreposição natural entre as outras duas espécies, e não de falha do pré-processamento.
4. Supervisado × não supervisionado: a árvore usa o rótulo e aprende regras de decisão; o K-Means só usa a geometria. Por isso a classificação supervisionada superou o alinhamento puro dos clusters — comportamento esperado.

### 6.1 Trabalhos futuros

- Aplicar **padronização** no K-Means e comparar com a versão sem escala.  
- Estimar *k* pelo método do cotovelo ou silhueta (sem assumir k=3).  
- Comparar a árvore com **Random Forest** ou **MLP** no mesmo conjunto.  
- Avaliar em datasets maiores e mais ruidosos, onde o pré-processamento e o *overfitting* pesam mais.
- Gerar visualizações com PCA para análise 2D alternativa.

---

## 7. Materiais entregues (implementação)

| Arquivo | Conteúdo |
|---------|----------|
| `parte1_arvore_decisao.py` | Classificação com Árvore de Decisão, Grid Search e métricas |
| `parte2_kmeans.py` | K-Means, tabela de contingência e comparação com classes |
| `resultados_parte1.png` | Matriz de confusão e visualização da árvore |
| `resultados_parte2.png` | Gráfico dos clusters e das classes reais |

**Como reproduzir:**

```bash
python parte1_arvore_decisao.py
python parte2_kmeans.py
```

Dependências principais: `scikit-learn`, `pandas`, `matplotlib`, `numpy`.

---

## Referências

1. Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics.  
2. UCI Machine Learning Repository — Iris: https://archive.ics.uci.edu/dataset/53/iris  
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.  
4. Material e enunciado da disciplina (Parte 1 — Árvore/MLP; Parte 2 — Agrupamento).
