"""KNN com a mesma limpeza e as mesmas 10 colunas da MLP.

Uso (dentro da pasta entrega/knn):  python3 treino.py
Saída: as métricas no terminal e o arquivo ../../submissoes/submission_knn.csv
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from limpeza import COLUNAS, limpar

SEED = 42

# ---------------------------------------------------------------
# 1. Leitura e limpeza
# ---------------------------------------------------------------
treino = pd.read_csv("../../dados/train.csv")

# 20 pacientes aparecem duas vezes com a linha idêntica. Se uma cópia fosse
# para o treino e a outra para a validação, a nota ficaria inflada.
treino = treino.drop_duplicates(subset=treino.columns.drop("id"))
print("linhas no treino depois de tirar duplicadas:", len(treino))

X = limpar(treino)[COLUNAS]
y = treino["class"]

# ---------------------------------------------------------------
# 2. Modelo e grade de hiperparâmetros
# O KNN mede distância, então a escala muda o resultado. O RobustScaler usa
# mediana e quartis, e sofre menos com os valores extremos da base.
# ---------------------------------------------------------------
pipe = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),
    ("escala", RobustScaler()),
    ("modelo", KNeighborsClassifier()),
])
grade = {
    "modelo__n_neighbors": list(range(3, 52, 2)),
    "modelo__weights": ["uniform", "distance"],
    "modelo__metric": ["euclidean", "manhattan", "chebyshev"],
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
# 5. Submissão: refaz a busca com todo o train.csv e prevê o test.csv
# ---------------------------------------------------------------
busca.fit(X, y)
print("\nparâmetros finais (train.csv inteiro):", busca.best_params_)

teste = pd.read_csv("../../dados/test.csv")
envio = pd.read_csv("../../dados/submission_template.csv")
pred_teste = pd.Series(busca.predict(limpar(teste)[COLUNAS]), index=teste["id"])
envio["class"] = envio["id"].map(pred_teste)
envio.to_csv("../../submissoes/submission_knn.csv", index=False)
print("submission_knn.csv salvo com", len(envio), "linhas")
