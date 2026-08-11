"""
Parte 2 — Agrupamento (K-Means) no Iris Dataset
Usa o mesmo pré-processamento da Parte 1 (sem rótulos no treino).
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Carregar o dataset (mesmo da Parte 1)
# ---------------------------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

print("=" * 60)
print("1. DATASET IRIS (mesmos dados da Parte 1)")
print("=" * 60)
print(f"Amostras: {len(X)} | Atributos: {list(feature_names)}")
print(f"Classes reais: {list(class_names)}")
print("IMPORTANTE: o K-Means NÃO usa a variável-alvo (y) para treinar.")


# ---------------------------------------------------------------------------
# 2. Pré-processamento (igual à Parte 1)
# ---------------------------------------------------------------------------
# - Sem valores ausentes
# - Atributos já numéricos
# - Sem normalização (como na Parte 1, para manter o mesmo pré-processamento)
# - Aqui usamos TODAS as amostras (não supervisionado; não há "treino/teste")

print("\n" + "=" * 60)
print("2. PRÉ-PROCESSAMENTO")
print("=" * 60)
print("Mesmo da Parte 1: dados numéricos, sem ausentes, sem normalização.")
print("No clustering usamos o dataset completo (não há rótulos no treino).")


# ---------------------------------------------------------------------------
# 3. K-Means com k = 3 (número de espécies reais)
# ---------------------------------------------------------------------------
# Justificativa do k=3: esperamos 3 grupos, um por espécie.

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

print("\n" + "=" * 60)
print(f"3. K-MEANS (k = {k})")
print("=" * 60)
print(f"Inércia (soma das distâncias aos centroides): {kmeans.inertia_:.2f}")
print(f"Tamanho de cada cluster: {np.bincount(clusters)}")


# ---------------------------------------------------------------------------
# 4. Comparar clusters com as classes reais
# ---------------------------------------------------------------------------
# Matriz de contingência: linhas = espécie real, colunas = cluster

tabela = pd.crosstab(
    pd.Series([class_names[i] for i in y], name="Classe real"),
    pd.Series(clusters, name="Cluster"),
)

print("\n" + "=" * 60)
print("4. TABELA DE CONTINGÊNCIA (classe real × cluster)")
print("=" * 60)
print(tabela)

# Atribuir a cada cluster a espécie mais frequente (para interpretar o alinhamento)
print("\nEspécie predominante em cada cluster:")
for c in range(k):
    idx = clusters == c
    contagem = pd.Series(y[idx]).value_counts()
    especie_dominante = class_names[contagem.idxmax()]
    pureza = contagem.max() / contagem.sum()
    print(f"  Cluster {c}: {especie_dominante} ({pureza:.0%} das amostras do cluster)")


# ---------------------------------------------------------------------------
# 5. Acurácia aproximada do agrupamento
# ---------------------------------------------------------------------------
# Mapeia cada cluster → classe majoritária e mede quantas batem com o rótulo real

mapa_cluster_para_classe = {}
for c in range(k):
    idx = clusters == c
    mapa_cluster_para_classe[c] = int(pd.Series(y[idx]).value_counts().idxmax())

y_pred_via_cluster = np.array([mapa_cluster_para_classe[c] for c in clusters])
acertos = (y_pred_via_cluster == y).sum()
print(f"\nAcertos ao 'traduzir' cluster -> especie: {acertos}/{len(y)} ({acertos / len(y):.1%})")
print(f"Matriz de confusao (apos mapear clusters):\n{confusion_matrix(y, y_pred_via_cluster)}")


# ---------------------------------------------------------------------------
# 6. Gráfico simples dos clusters
# ---------------------------------------------------------------------------
# Usamos pétala comprimento × pétala largura (atributos mais discriminativos do Iris)

petal_len = X[:, 2]  # petal length (cm)
petal_wid = X[:, 3]  # petal width (cm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Clusters do K-Means
scatter0 = axes[0].scatter(petal_len, petal_wid, c=clusters, cmap="viridis", edgecolors="k", s=50)
axes[0].scatter(
    kmeans.cluster_centers_[:, 2],
    kmeans.cluster_centers_[:, 3],
    c="red",
    marker="X",
    s=200,
    label="Centroides",
    edgecolors="black",
)
axes[0].set_title("Clusters formados pelo K-Means (k=3)")
axes[0].set_xlabel(feature_names[2])
axes[0].set_ylabel(feature_names[3])
axes[0].legend()
plt.colorbar(scatter0, ax=axes[0], label="Cluster")

# (b) Classes reais (para comparar visualmente)
scatter1 = axes[1].scatter(petal_len, petal_wid, c=y, cmap="viridis", edgecolors="k", s=50)
axes[1].set_title("Classes reais (rótulos)")
axes[1].set_xlabel(feature_names[2])
axes[1].set_ylabel(feature_names[3])
plt.colorbar(scatter1, ax=axes[1], label="Espécie")

plt.tight_layout()
plt.savefig("resultados_parte2.png", dpi=120, bbox_inches="tight")
print("\nGráfico salvo em: resultados_parte2.png")
plt.show()


# ---------------------------------------------------------------------------
# 7. Conclusão breve (para o relatório)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("7. CONCLUSÃO (Parte 2)")
print("=" * 60)
print(
    """
Os clusters se aproximaram bem das classes reais no Iris:
- Setosa fica quase sempre isolada em um cluster próprio (separa fácil).
- Versicolor e Virginica se sobrepõem um pouco no espaço de atributos,
  então o K-Means pode misturar algumas dessas flores.

Por quê? O K-Means agrupa por proximidade (distância), sem usar o rótulo.
Quando espécies são morfologicamente parecidas, os grupos se misturam.
Já a Setosa é bem distinta, então o cluster correspondente fica "puro".
"""
)

print("Parte 2 concluída.")
