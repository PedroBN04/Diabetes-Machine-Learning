"""Árvore de Decisão com a mesma limpeza da MLP, mas só 3 colunas.

Uso (dentro da pasta entrega/arvore):  python3 treino.py
Saída: as métricas e as regras no terminal e o arquivo ../../submissoes/submission_tree.csv
"""
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from limpeza import limpar

SEED = 42

# a árvore não usa as outras colunas do limpeza.py
COLUNAS_ARVORE = ["criterio_ada", "hba1c", "hist_familiar"]

# ---------------------------------------------------------------
# 1. Leitura e limpeza
# ---------------------------------------------------------------
treino = pd.read_csv("../../dados/train.csv")

# 20 pacientes aparecem duas vezes com a linha idêntica. Se uma cópia fosse
# para o treino e a outra para a validação, a nota ficaria inflada.
treino = treino.drop_duplicates(subset=treino.columns.drop("id"))
print("linhas no treino depois de tirar duplicadas:", len(treino))

X = limpar(treino)[COLUNAS_ARVORE]
y = treino["class"]

# ---------------------------------------------------------------
# 2. Modelo e grade de hiperparâmetros
# Árvore não precisa de escala, só de preencher os vazios. A poda é feita de
# dois jeitos: antes (max_depth, min_samples_leaf, max_leaf_nodes limitam o
# crescimento) e depois (ccp_alpha corta galhos que quase não ajudam).
# A grade antes tinha 864 combinações e o script levava mais de 8 minutos,
# porque a busca roda de novo em cada uma das 30 rodadas. Tirei max_depth 6 e
# 8 e ccp_alpha 0.03, e ficaram 384.
# ---------------------------------------------------------------
pipe = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("modelo", DecisionTreeClassifier(random_state=SEED)),
])
grade = {
    "modelo__criterion": ["gini", "entropy"],
    "modelo__max_depth": [2, 3, 4, 5],
    "modelo__min_samples_leaf": [5, 20, 40],
    "modelo__max_leaf_nodes": [4, 6, 10, None],
    "modelo__ccp_alpha": [0, 0.01],
    "modelo__class_weight": [None, "balanced"],
}
cv_interna = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
busca = GridSearchCV(pipe, grade, scoring="f1_macro", cv=cv_interna, n_jobs=-1)


def metricas(y_real, pred):
    return {
        "Precisão": precision_score(y_real, pred),
        "Recall (Diabetes)": recall_score(y_real, pred, pos_label=1),
        "Recall (Controle)": recall_score(y_real, pred, pos_label=0),
        "F1-macro": f1_score(y_real, pred, average="macro"),
    }


# ---------------------------------------------------------------
# 3. Holdout: a busca só vê os 80%, e a nota sai dos 20% restantes
# ---------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
busca.fit(X_tr, y_tr)
print("\n===== Holdout (20% do train.csv) =====")
print("melhores parâmetros nos 80%:", busca.best_params_)
for nome, valor in metricas(y_val, busca.predict(X_val)).items():
    print(f"{nome}: {valor:.3f}")

# ---------------------------------------------------------------
# 4. 30 rodadas (5 partes x 6 repetições), com a busca refeita dentro de cada
# rodada. Assim a parte de validação não participa da escolha dos parâmetros.
# ---------------------------------------------------------------
cv_30 = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=7)
rodadas = []
for idx_tr, idx_val in cv_30.split(X, y):
    busca.fit(X.iloc[idx_tr], y.iloc[idx_tr])
    rodadas.append(metricas(y.iloc[idx_val], busca.predict(X.iloc[idx_val])))
rodadas = pd.DataFrame(rodadas)

print("\n===== 30 rodadas (busca dentro de cada rodada) =====")
for nome in rodadas.columns:
    print(f"{nome}: {rodadas[nome].mean():.3f} (desvio {rodadas[nome].std():.3f})")

# ---------------------------------------------------------------
# 5. Árvore final
# A busca com todo o train.csv escolhe uma árvore de 10 folhas, só que vários
# cortes dela dão a mesma classe dos dois lados. Depois de ver isso, troquei
# por max_depth=3 e max_leaf_nodes=4. Essa simplificação foi decisão minha,
# feita depois da busca: a árvore de 4 folhas não é o que a grade escolheu, e
# com esta semente ela até tem nota de CV menor. Fiquei com ela porque dá as
# mesmas respostas nas linhas do train.csv (o assert confere) e é mais fácil
# de explicar.
# ---------------------------------------------------------------
busca.fit(X, y)
print("\nparâmetros da busca (train.csv inteiro):", busca.best_params_)
print("folhas da árvore da busca:", busca.best_estimator_.named_steps["modelo"].get_n_leaves())

simples = clone(busca.best_estimator_).set_params(modelo__max_depth=3, modelo__max_leaf_nodes=4).fit(X, y)
assert (simples.predict(X) == busca.best_estimator_.predict(X)).all()
print("folhas da árvore simplificada:", simples.named_steps["modelo"].get_n_leaves())

print("\nRegras da árvore simplificada:")
print(export_text(simples.named_steps["modelo"], feature_names=COLUNAS_ARVORE))

# ---------------------------------------------------------------
# 6. Submissão: usa a árvore simplificada para prever o test.csv
# ---------------------------------------------------------------
teste = pd.read_csv("../../dados/test.csv")
envio = pd.read_csv("../../dados/submission_template.csv")
pred_teste = pd.Series(simples.predict(limpar(teste)[COLUNAS_ARVORE]), index=teste["id"])
envio["class"] = envio["id"].map(pred_teste)
envio.to_csv("../../submissoes/submission_tree.csv", index=False)
print("submission_tree.csv salvo com", len(envio), "linhas")
