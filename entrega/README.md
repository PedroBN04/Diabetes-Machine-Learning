# Detecção de diabetes – competição Kaggle da disciplina

Trabalho de mineração de dados: classificar pacientes em **0 (controle)** ou
**1 (diabetes)** a partir de dados clínicos, com F1-macro como métrica. Os
modelos pedidos são KNN, Árvore de Decisão e Rede Neural (MLP).

## Organização das pastas

```
Datascience/
├── entrega/
│   ├── README.md
│   ├── requirements.txt
│   ├── mlp/      limpeza.py e treino.py
│   ├── knn/      limpeza.py e treino.py
│   └── arvore/   limpeza.py e treino.py
├── dados/        train.csv, test.csv e submission_template.csv
└── submissoes/   os três CSVs enviados ao Kaggle
```

Os dados não vão junto com a entrega porque a base é de uso exclusivo da
disciplina, então ela não deve ser enviada para repositório público nem para
nenhum serviço externo. Para rodar, a pasta `dados/` precisa estar ao lado da
`entrega/`, e cada script precisa ser executado de dentro da pasta dele.

## Arquivos

| arquivo | o que tem |
|---|---|
| `requirements.txt` | versões das bibliotecas usadas |
| `mlp/limpeza.py` | funções que limpam e padronizam as colunas |
| `mlp/treino.py` | busca da MLP, métricas e submissão |
| `knn/limpeza.py` | cópia igual à da pasta `mlp/` |
| `knn/treino.py` | busca do KNN, métricas e submissão |
| `arvore/limpeza.py` | cópia igual à da pasta `mlp/` |
| `arvore/treino.py` | busca da árvore, simplificação, regras, métricas e submissão |
| `../submissoes/submission_mlp.csv` | submissão da MLP (0.91 no Kaggle) |
| `../submissoes/submission_knn.csv` | submissão do KNN novo (0.885 no Kaggle) |
| `../submissoes/submission_tree.csv` | submissão da árvore nova (0.857 no Kaggle) |

No código do `mlp/treino.py` só mudaram os caminhos dos arquivos, e alguns
comentários foram corrigidos. Por isso ele ainda imprime um KNN e uma árvore
antigos, que não são os modelos das pastas `knn/` e `arvore/`.

## Como rodar no Fedora

Os scripts foram rodados com Python 3.12, e as versões do `requirements.txt`
estão fixas. Algumas delas não têm pacote pronto para Python 3.14, que é o que
as versões recentes do Fedora trazem, então é mais seguro instalar o 3.12 ao
lado do Python do sistema. Ele não substitui o que já está instalado.

1. Instalar o Python 3.12 pelo dnf:

```bash
sudo dnf install python3.12
```

2. Entrar na pasta da entrega e criar o ambiente virtual:

```bash
cd Datascience/entrega
```

```bash
python3.12 -m venv .venv
```

3. Ativar o ambiente (precisa fazer isso toda vez que abrir um terminal novo):

```bash
source .venv/bin/activate
```

4. Instalar as bibliotecas:

```bash
pip install -r requirements.txt
```

5. Rodar a MLP. Leva cerca de dois minutos:

```bash
cd mlp
python3 treino.py
```

6. Rodar o KNN. Leva de dois a cinco minutos:

```bash
cd ../knn
python3 treino.py
```

7. Rodar a árvore. É a mais demorada, de quatro a dez minutos:

```bash
cd ../arvore
python3 treino.py
```

Esses tempos são de um computador com 8 núcleos lógicos. Em uma máquina mais fraca vai
demorar mais, principalmente a árvore. Cada script mostra as métricas no
terminal e grava a sua submissão em `submissoes/`, por cima da que já estiver
lá. Os CSVs da pasta foram gerados no Windows. Em outro sistema a MLP pode mudar
alguma previsão, então guarde uma cópia deles antes de rodar se quiser mantê-los.

Para sair do ambiente virtual depois, é só digitar `deactivate`.

**Se der erro de compilação no `pip install`**, o ambiente provavelmente foi
criado com outra versão do Python. Confira com `python3 --version` dentro do
ambiente. Se não for 3.12, apague a pasta `.venv` e repita os passos 2 a 4.

## O que foi feito

### Limpeza

Os três modelos agora usam a mesma limpeza. Apesar do enunciado falar em
espectroscopia FTIR, as colunas são dados clínicos comuns (idade, peso,
glicose, HbA1c etc.) e estão bem bagunçadas. O que precisei arrumar:

- **Linhas repetidas:** 20 pacientes aparecem duas vezes com a linha
  idêntica. Tirei as cópias antes de separar a validação, senão a mesma
  pessoa podia cair no treino e na validação ao mesmo tempo.
- **Sexo, fumante, atividade física, histórico familiar:** escritos de vários
  jeitos (`F`, `female`, ` MALE`, `Ligth`, `Sedentry`, `Sim`, `Não`...).
  Padronizei tudo em números.
- **Altura:** uma parte em metros (`1,62`) e outra em centímetros. Passei
  tudo para cm.
- **Glicose:** a maioria em mg/dL, mas alguns valores abaixo de 30 estão em
  mmol/L. Multipliquei esses por 18.
- **HEMOGLOBIN:** pelos valores é a hemoglobina glicada (HbA1c). Os valores
  acima de 20 estão em mmol/mol, convertidos para % com
  `% = mmol/mol / 10.929 + 2.15`.
- **Idade:** tem -1, 0 e 999. Nesses casos usei ano do exame menos ano de
  nascimento.
- **Pressão:** separei `120/80`, `120 / 80` e `120x80` em sistólica e
  diastólica. `?` vira vazio.
- **Triglicerídeos:** alguns valores são muito altos, usei log.

Os valores que continuam vazios são preenchidos com a mediana. Essa mediana
é calculada só com os dados de treino de cada rodada, dentro do pipeline.

### Sobre "FTIR"

O enunciado fala em espectroscopia FTIR, mas o arquivo não tem espectro
nenhum: são 19 colunas clínicas. Por isso técnicas de espectro, como SNV ou
suavização de curva, não se aplicam aqui. O que dava para testar era a escala
e o PCA:

- `StandardScaler`, `RobustScaler` e `QuantileTransformer` deram resultados
  parecidos. Os três entram na busca da MLP. O KNN usa só o `RobustScaler`.
- `MinMaxScaler` ficou um pouco atrás na primeira rodada de testes.
- PCA (com 3, 5, 7 componentes ou 95% da variância) não ajudou nenhum modelo.
  Com só 10 colunas não tem muito o que reduzir.

### Features

Testei tirar e colocar uma coluna de cada vez. Sexo, altura, peso, sódio,
fumo, atividade, tipo sanguíneo, aparelho e local de coleta não ajudaram ou
pioraram, principalmente no KNN. Fiquei com 10:

- HbA1c, glicose, idade, IMC, pressão sistólica, log dos triglicerídeos
- histórico familiar
- duas colunas que marcam se faltou HbA1c ou glicose
- uma coluna com o critério de diagnóstico da ADA
  (HbA1c ≥ 6.5% ou glicose ≥ 126 mg/dL)

A coluna de "faltou HbA1c" parece estranha, então olhei os números. Na base
toda, faltar HbA1c é igual nas duas classes: 23,2% no controle e 23,7% no
diabetes. A diferença só aparece entre quem tem `criterio_ada == 0`, ou seja,
quem não bateu o critério: ali faltam 24,1% no controle e 57,7% no diabetes.

### O que muda de um modelo para outro

| | MLP | KNN | Árvore |
|---|---|---|---|
| limpeza | a mesma | a mesma | a mesma |
| colunas | as 10 acima | as 10 acima | `criterio_ada`, `hba1c`, `hist_familiar` |
| vazios | mediana | mediana | mediana |
| escala | a busca escolhe entre Standard, Robust e Quantile | `RobustScaler` | nenhuma |
| grade | camadas, ativação e `alpha` | `n_neighbors` ímpar de 3 a 51, `weights` e `metric` (euclidiana, Manhattan, Chebyshev), 150 combinações | `criterion`, `max_depth` (2 a 5), `min_samples_leaf`, `max_leaf_nodes`, `ccp_alpha` (0 e 0.01) e `class_weight`, 384 combinações |
| parâmetros finais | `RobustScaler`, camadas de 16 e 8 neurônios, ReLU, `alpha=1.0`, `learning_rate_init=0.01` (essa escolha vem dos 80% do holdout) | Manhattan, 21 vizinhos, pesos iguais | a busca escolhe gini, profundidade 4, 10 folhas, mínimo de 5 por folha; a final tem 4 folhas (ver abaixo) |

### Validação e vazamento de dados

- O `test.csv` não é usado no treino nem na validação. Nos scripts do KNN e da
  árvore ele só é lido no final. No da MLP ele é lido no começo, junto com o
  treino, mas só é usado no final, para gerar a submissão.
- Fora dos scripts, o teste foi aberto na exploração inicial: calculei resumos
  das colunas (quartis e valores vazios) também nele. Nenhum número desses entra
  nos modelos, mas não dá para garantir que ver esses resumos não influenciou
  alguma decisão de limpeza, como os cortes de unidade.
- Conferi que nenhum paciente (`PATIENT_CODE`) nem nenhuma linha aparece
  no treino e no teste ao mesmo tempo.
- A limpeza (`limpeza.py`) trata cada linha sozinha, sem tirar média ou
  qualquer outro número da base. Tudo que é "aprendido" (mediana, escala,
  modelo) fica dentro do `Pipeline` do scikit-learn, então é ajustado só
  com a parte de treino de cada rodada.
- Sobram 520 linhas no `train.csv` sem as duplicadas. O holdout separa 20%
  delas (104 pacientes), estratificado pela classe. Nos 80% restantes roda o
  `GridSearchCV` com validação cruzada de 5 partes, otimizando o F1-macro.
- No KNN e na árvore, as 30 rodadas (5 partes × 6 repetições) são aninhadas:
  a busca é refeita dentro de cada rodada, então a parte de validação não
  ajuda a escolher os parâmetros.

## Resultados

| modelo | medida | precisão | recall diabetes | recall controle | F1-macro |
|---|---|---|---|---|---|
| KNN | holdout | 1.000 | 0.944 | 1.000 | 0.971 |
| KNN | 30 rodadas, aninhado | 0.970 ± 0.028 | 0.838 ± 0.060 | 0.971 ± 0.029 | **0.901 ± 0.031** |
| Árvore | holdout | 0.962 | 0.944 | 0.960 | 0.952 |
| Árvore | 30 rodadas, aninhado | 0.952 ± 0.033 | 0.816 ± 0.055 | 0.955 ± 0.031 | **0.883 ± 0.035** |
| MLP | holdout | 0.964 | 0.981 | 0.960 | 0.971 |
| MLP | 30 rodadas, **não** aninhado | não calculado | não calculado | não calculado | **0.903 ± 0.023** |

| modelo | nota no Kaggle |
|---|---|
| MLP | 0.91 |
| KNN novo | 0.885 |
| Árvore nova | 0.857 |

O 0.903 ± 0.023 da MLP usa parâmetros escolhidos nos 80% do holdout, e as 30
rodadas passam por essas mesmas linhas. Então não é aninhado e fica um pouco
otimista. Para comparar: rodando a MLP aninhada em 10 dobras, o resultado também
foi 0.903.

A grade da árvore tinha 864 combinações (também `max_depth` 6 e 8 e `ccp_alpha`
0.03), mas o script levava mais de 8 minutos, então cortei para 384. Rodei as
duas: as métricas impressas e a submissão saíram iguais.

**Nenhum modelo chegou à meta de 0.93 nas 30 rodadas.**

### O quanto confiar no holdout

Pouco. A divisão com semente 42 é favorável: ela ganhou ou empatou com todas as
outras 20 sementes que testei, nos três modelos. A média do holdout entre as
sementes fica em cerca de 0.904 no KNN, 0.879 na árvore e 0.906 na MLP com
parâmetros fixos, bem abaixo do que aparece na tabela.

Isso vale também para o recall de 0.981 da MLP no holdout. Ele não quer dizer
que a MLP acha mais diabéticos: o recall real dela fica em torno de 0.87 a
0.88.

### Escolha do modelo final: MLP

Na validação cruzada a MLP não se separa do KNN, e nenhuma diferença entre pares
de modelos foi significativa. O que separa a MLP é a nota de 0.91 no Kaggle, e
mesmo essa diferença cabe em cerca de 2 desvios de ruído. Continuo com a MLP
porque foi a melhor nota no Kaggle, sabendo que a vantagem medida é pequena.

O script retreina a MLP com o `train.csv` inteiro e prevê o `test.csv`.

Para tentar subir a nota testei:

- outros jeitos de preencher os vazios (`KNNImputer`, `IterativeImputer`)
- faixas clínicas (normal / pré-diabetes / diabetes) no lugar dos valores
- cortar os valores extremos
- PCA e quatro scalers diferentes
- votação entre os três modelos
- só para comparar, Random Forest e Gradient Boosting

Nada subiu a nota de forma estável. Olhando os erros, boa parte são pacientes
com HbA1c e glicose normais marcados como diabéticos (talvez já em tratamento)
e 18 pacientes sem nenhum dos dois exames. Nenhuma coluna da base explica esses
casos, então parece ser o limite dos próprios dados.

## KNN e árvore: comparação com as versões anteriores

As versões anteriores usavam 4 colunas em cada modelo. As novas foram trocadas
pelas colunas descritas acima.

| modelo | colunas | F1-macro aninhado | recall diabetes | recall controle | Kaggle |
|---|---|---|---|---|---|
| KNN anterior | 4 | 0.904 | 0.87 | 0.93 | 0.87 |
| KNN novo | 10 | 0.902 | 0.83 | 0.97 | 0.885 |
| Árvore anterior | 4 | 0.888 | 0.87 | 0.91 | 0.86 |
| Árvore nova | 3 | 0.881 | 0.83 | 0.94 | 0.857 |

Os números desta tabela vêm de uma comparação à parte, com as mesmas dobras nas
duas versões. A semente da busca interna era outra e a grade da árvore era a de 864
combinações, por isso aparecem pequenas diferenças em relação à tabela de
resultados.

No F1-macro os dois pares estão empatados estatisticamente (t corrigido de -0.22
no KNN e -0.52 na árvore). A troca não veio de uma nota melhor. As 4 colunas
antigas ganharam uma busca entre todas as combinações feita neste mesmo
`train.csv`, e a validação aninhada não corrige isso, porque a escolha das
colunas ficou de fora dela. Então as notas de CV das versões anteriores podem
estar infladas, e por isso prefiro um conjunto que não foi escolhido desse jeito.

A troca tem um custo: os modelos novos acham menos diabéticos (recall de cerca de
0.83 contra 0.87) e erram menos no controle (0.97 contra 0.93 no KNN, 0.94
contra 0.91 na árvore).

No Kaggle o KNN foi de 0.87 para 0.885 e a árvore ficou praticamente igual (0.86
para 0.857). Nenhuma das duas mudanças prova muita coisa: uma diferença abaixo de
0.04 a 0.05 entre a versão antiga e a nova deve ser lida como ruído, para qualquer
um dos lados. A MLP continua com a melhor nota (0.91).

## O que a árvore faz

A busca no `train.csv` inteiro escolhe uma árvore de 10 folhas (gini,
profundidade 4, mínimo de 5 amostras por folha). Só que vários cortes dela dão a
mesma classe dos dois lados. Depois da busca, troquei por `max_depth=3` e
`max_leaf_nodes=4`, uma decisão minha, que a grade não tomou: com esta semente
a árvore de 10 folhas ganha na CV interna (0.8824 contra 0.8706). Fiquei com a
de 4 folhas porque ela dá as mesmas respostas nas 520 linhas do treino (o script
confere com um `assert`) e é mais fácil de explicar. É ela que gera a submissão.

O holdout e as 30 rodadas avaliam a busca, ou seja, a árvore que a grade escolhe
em cada rodada, e não essa árvore de 4 folhas fixa.

Regras que o script imprime:

```
|--- criterio_ada <= 0.50
|   |--- hba1c <= 5.80
|   |   |--- class: 0
|   |--- hba1c >  5.80
|   |   |--- hba1c <= 6.38
|   |   |   |--- class: 0
|   |   |--- hba1c >  6.38
|   |   |   |--- class: 1
|--- criterio_ada >  0.50
|   |--- class: 1
```

Na prática ela prevê diabetes se `criterio_ada == 1` ou se a HbA1c, depois de
preenchida com a mediana, passa de 6.38. O `hist_familiar` continua nas colunas
de entrada, mas não muda nenhuma previsão.

A regra da ADA sozinha, sem modelo nenhum, dá 0.8823, praticamente a mesma nota
da árvore. A árvore acrescenta muito pouco em cima dela.

Tem também um ponto fraco claro: as 88 linhas com `criterio_ada == 0` e HbA1c
faltando sempre recebem controle, e 30 delas são de diabetes.

## O que mais tentei e não ajudou

- Para a árvore: deixar ela tratar os vazios sozinha (o scikit-learn 1.8
  aceita NaN na árvore) ou preencher com -1 para separar quem não fez o
  exame. Os dois ficaram piores que a mediana.
- Para o KNN: `KNNImputer` no lugar da mediana e `QuantileTransformer`. O
  primeiro piorou e o segundo empatou.
