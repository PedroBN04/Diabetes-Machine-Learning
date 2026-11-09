"""Treino e avaliação dos modelos (KNN, Árvore de Decisão e MLP).

Uso (dentro da pasta entrega/mlp):  python3 treino.py
Saída: as métricas de cada modelo no terminal e o arquivo ../../submissoes/submission_mlp.csv
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from limpeza import COLUNAS, limpar

SEED = 42

# ---------------------------------------------------------------
# 1. Leitura e limpeza
# ---------------------------------------------------------------
treino = pd.read_csv("../../dados/train.csv")
teste = pd.read_csv("../../dados/test.csv")

# 20 pacientes aparecem duas vezes com a linha idêntica. Se uma cópia fosse
# para o treino e a outra para a validação, a nota ficaria inflada.
treino = treino.drop_duplicates(subset=treino.columns.drop("id"))
print("linhas no treino depois de tirar duplicadas:", len(treino))

X = limpar(treino)[COLUNAS]
y = treino["class"]

# ---------------------------------------------------------------
# 2. Separação treino / validação (só dentro do train.csv)
# ---------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ---------------------------------------------------------------
# 3. Modelos e grades de hiperparâmetros
# Todos passam pelo mesmo pré-processamento: preencher vazios com a
# mediana e depois colocar as colunas na mesma escala. O KNN e a MLP
# dependem da escala; para a árvore não muda nada.
#
# A escala também entra na busca. Comparei StandardScaler, MinMaxScaler,
# RobustScaler (usa mediana e quartis, sofre menos com valores extremos),
# QuantileTransformer (deixa cada coluna com formato de normal) e PCA.
# MinMax ficou atrás e o PCA não ajudou em nenhum modelo, então ficaram
# só os outros três na grade.
# ---------------------------------------------------------------
def pipeline(modelo):
    return Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("modelo", modelo),
    ])


escalas = [
    StandardScaler(),
    RobustScaler(),
    QuantileTransformer(n_quantiles=100, output_distribution="normal"),
]

modelos = {
    "KNN": (
        pipeline(KNeighborsClassifier()),
        {
            "escala": escalas,
            "modelo__n_neighbors": [5, 9, 13, 17, 21, 25, 29, 35, 41],
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        },
    ),
    "Árvore de Decisão": (
        pipeline(DecisionTreeClassifier(random_state=SEED)),
        {
            "modelo__max_depth": [3, 4, 5, 6, 8],
            "modelo__min_samples_leaf": [1, 5, 10, 20],
            "modelo__criterion": ["gini", "entropy"],
            # ccp_alpha poda a árvore depois de pronta, corta galhos que
            # quase não melhoram o resultado
            "modelo__ccp_alpha": [0, 0.005, 0.01, 0.02],
            "modelo__class_weight": [None, "balanced"],
        },
    ),
    "MLP": (
        pipeline(MLPClassifier(max_iter=3000, random_state=SEED)),
        {
            "escala": escalas,
            "modelo__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
            "modelo__activation": ["relu", "tanh"],
            # alpha alto = mais regularização. Com poucos dados (520 linhas)
            # os melhores valores ficaram entre 0.3 e 3
            "modelo__alpha": [0.1, 0.3, 0.6, 1.0, 3.0],
            # 0.001 sempre perdeu para 0.01 nos testes, deixei só esse
            "modelo__learning_rate_init": [0.01],
        },
    ),
}

# validação cruzada estratificada com 5 partes, otimizando o F1-macro
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
resultados = []

for nome, (pipe, grade) in modelos.items():
    busca = GridSearchCV(pipe, grade, scoring="f1_macro", cv=cv, n_jobs=-1)
    busca.fit(X_tr, y_tr)
    pred = busca.predict(X_val)

    res = {
        "modelo": nome,
        "f1_macro_cv": busca.best_score_,
        "precisao": precision_score(y_val, pred),
        "recall_diabetes": recall_score(y_val, pred, pos_label=1),
        "recall_controle": recall_score(y_val, pred, pos_label=0),
        "f1_macro_val": f1_score(y_val, pred, average="macro"),
    }
    resultados.append(res)

    print(f"\n===== {nome} =====")
    print("melhores parâmetros:", busca.best_params_)
    print(classification_report(y_val, pred, target_names=["Controle", "Diabetes"], digits=3))
    modelos[nome] = busca.best_estimator_

tabela = pd.DataFrame(resultados).set_index("modelo").round(3)
print("\nResumo (validação = 20% separados do train.csv):")
print(tabela.to_string())

# ---------------------------------------------------------------
# 4. Escolha do modelo final
#
# Escolhi a MLP porque teve a melhor nota no Kaggle (0.91). Na validação
# cruzada ela empata com o KNN, os dois perto de 0.90, então a vantagem é
# pequena e pode ser ruído. A árvore é a mais fácil de explicar e ficou um
# pouco atrás. O KNN e a árvore deste script são só comparação: os modelos
# finais deles estão nas pastas knn/ e arvore/.
# ---------------------------------------------------------------
melhor = "MLP"
final = modelos[melhor]

# A nota de uma única divisão de 20% varia muito trocando só a semente. Para
# ter uma ideia mais realista do que esperar no Kaggle, repito a validação
# cruzada 6 vezes (30 rodadas no total). Os parâmetros saíram dos 80% acima e
# essas rodadas passam pelas mesmas linhas, então a nota fica um pouco otimista.
cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=SEED)
notas = cross_val_score(final, X, y, cv=cv_rep, scoring="f1_macro", n_jobs=-1)
print(f"\n{melhor}: F1-macro em 30 rodadas = {notas.mean():.3f} (desvio {notas.std():.3f})")

# ---------------------------------------------------------------
# 5. Submissão: retreina com todo o train.csv e prevê o test.csv
# ---------------------------------------------------------------
final.fit(X, y)

envio = pd.read_csv("../../dados/submission_template.csv")
pred_teste = pd.Series(final.predict(limpar(teste)[COLUNAS]), index=teste["id"])
envio["class"] = envio["id"].map(pred_teste)
envio.to_csv("../../submissoes/submission_mlp.csv", index=False)
print("submission_mlp.csv salvo com", len(envio), "linhas")
