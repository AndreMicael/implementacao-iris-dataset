"""
Parte 1 — Classificação do Iris com Árvore de Decisão
Tema: Iris Dataset (Setosa, Versicolor, Virginica)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Carregar e explorar o dataset
# ---------------------------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["especie"] = [class_names[i] for i in y]

print("=" * 60)
print("1. DATASET IRIS")
print("=" * 60)
print(f"Total de amostras: {len(df)}")
print(f"Atributos: {list(feature_names)}")
print(f"Classes: {list(class_names)}")
print(f"\nValores ausentes por coluna:\n{df.isnull().sum()}")
print(f"\nDistribuição das classes:\n{df['especie'].value_counts()}")
print(f"\nPrimeiras linhas:\n{df.head()}")


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------
# - Iris não tem valores ausentes
# - Todos os atributos já são numéricos (não precisa de codificação)
# - Árvore de decisão não exige normalização/padronização
# - Fazemos a divisão treino/teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("2. PRÉ-PROCESSAMENTO")
print("=" * 60)
print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
print("Sem valores ausentes. Atributos numéricos. Sem necessidade de normalização.")


# ---------------------------------------------------------------------------
# 3. Modelo 1 — configuração simples (baseline)
# ---------------------------------------------------------------------------
modelo_simples = DecisionTreeClassifier(random_state=42)
modelo_simples.fit(X_train, y_train)
y_pred_simples = modelo_simples.predict(X_test)

print("\n" + "=" * 60)
print("3. MODELO 1 — Árvore sem restrições (baseline)")
print("=" * 60)
print(f"Profundidade da árvore: {modelo_simples.get_depth()}")
print(f"Número de folhas: {modelo_simples.get_n_leaves()}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_simples):.4f}")
print(f"Precisão (macro): {precision_score(y_test, y_pred_simples, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred_simples, average='macro'):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_simples, average='macro'):.4f}")
print(f"\nMatriz de confusão:\n{confusion_matrix(y_test, y_pred_simples)}")
print(f"\nRelatório completo:\n{classification_report(y_test, y_pred_simples, target_names=class_names)}")


# ---------------------------------------------------------------------------
# 4. Ajuste de hiperparâmetros (Grid Search)
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. AJUSTE DE HIPERPARÂMETROS (GridSearchCV)")
print("=" * 60)

param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Melhor acurácia (validação cruzada): {grid.best_score_:.4f}")


# ---------------------------------------------------------------------------
# 5. Modelo 2 — com hiperparâmetros ajustados
# ---------------------------------------------------------------------------
modelo_ajustado = grid.best_estimator_
y_pred_ajustado = modelo_ajustado.predict(X_test)

print("\n" + "=" * 60)
print("5. MODELO 2 — Árvore com hiperparâmetros ajustados")
print("=" * 60)
print(f"Profundidade da árvore: {modelo_ajustado.get_depth()}")
print(f"Número de folhas: {modelo_ajustado.get_n_leaves()}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_ajustado):.4f}")
print(f"Precisão (macro): {precision_score(y_test, y_pred_ajustado, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred_ajustado, average='macro'):.4f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred_ajustado, average='macro'):.4f}")
print(f"\nMatriz de confusão:\n{confusion_matrix(y_test, y_pred_ajustado)}")
print(f"\nRelatório completo:\n{classification_report(y_test, y_pred_ajustado, target_names=class_names)}")


# ---------------------------------------------------------------------------
# 6. Comparação das duas configurações
# ---------------------------------------------------------------------------
print("=" * 60)
print("6. COMPARAÇÃO: baseline vs. ajustado")
print("=" * 60)

comparacao = pd.DataFrame(
    {
        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-score"],
        "Baseline": [
            accuracy_score(y_test, y_pred_simples),
            precision_score(y_test, y_pred_simples, average="macro"),
            recall_score(y_test, y_pred_simples, average="macro"),
            f1_score(y_test, y_pred_simples, average="macro"),
        ],
        "Ajustado": [
            accuracy_score(y_test, y_pred_ajustado),
            precision_score(y_test, y_pred_ajustado, average="macro"),
            recall_score(y_test, y_pred_ajustado, average="macro"),
            f1_score(y_test, y_pred_ajustado, average="macro"),
        ],
    }
)
print(comparacao.to_string(index=False))


# ---------------------------------------------------------------------------
# 7. Gráficos (matriz de confusão + árvore)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusão do modelo ajustado
cm = confusion_matrix(y_test, y_pred_ajustado)
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_title("Matriz de Confusão (modelo ajustado)")
axes[0].set_xlabel("Classe prevista")
axes[0].set_ylabel("Classe real")
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(class_names, rotation=45, ha="right")
axes[0].set_yticklabels(class_names)
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
fig.colorbar(im, ax=axes[0], fraction=0.046)

# Visualização da árvore
plot_tree(
    modelo_ajustado,
    feature_names=feature_names,
    class_names=list(class_names),
    filled=True,
    rounded=True,
    fontsize=7,
    ax=axes[1],
)
axes[1].set_title("Árvore de Decisão (modelo ajustado)")

plt.tight_layout()
plt.savefig("resultados_parte1.png", dpi=120, bbox_inches="tight")
print("\nGráfico salvo em: resultados_parte1.png")
plt.show()

print("\nParte 1 concluída.")
