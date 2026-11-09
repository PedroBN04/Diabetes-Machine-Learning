"""Funções de limpeza dos dados. Usadas pelo treino.py.

A base vem bem suja: unidades misturadas, textos escritos de vários jeitos,
datas em formatos diferentes. Aqui a gente padroniza tudo em números.
"""
import numpy as np
import pandas as pd


def ler_data(texto):
    # as datas aparecem como 2020-01-26, 26/07/2018 ou "Jan 8 2018"
    if pd.isna(texto):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%b %d %Y"):
        try:
            return pd.to_datetime(texto, format=fmt)
        except ValueError:
            pass
    return pd.NaT


def limpar(df):
    d = pd.DataFrame(index=df.index)

    # --- sexo: F, f, FEMALE, Female... vira 1 = feminino, 0 = masculino
    sexo = df["GENDER"].astype(str).str.strip().str.upper().str[0]
    d["feminino"] = sexo.map({"F": 1, "M": 0})

    # --- idade: tem valores impossíveis (-1, 999). Quando a idade não faz
    # sentido, calculo pelo ano do exame menos o ano de nascimento.
    ano_exame = df["EXAM_DATE"].map(ler_data).dt.year
    idade = pd.to_numeric(df["AGE"], errors="coerce")
    idade_calc = ano_exame - df["BIRTH_YEAR"]
    idade = idade.where((idade >= 18) & (idade <= 100), idade_calc)
    d["idade"] = idade.where((idade >= 18) & (idade <= 100))

    # --- altura: umas em metros (1,62 ou 1.62), outras em cm
    altura = pd.to_numeric(df["HEIGHT"].astype(str).str.replace(",", "."), errors="coerce")
    altura = np.where(altura < 3, altura * 100, altura)
    d["altura"] = altura
    d["peso"] = pd.to_numeric(df["WEIGHT"], errors="coerce")
    d["imc"] = d["peso"] / (d["altura"] / 100) ** 2

    # --- pressão "120/80", "120 / 80" ou "120x80" -> duas colunas
    pa = df["BLOOD_PRESSURE"].astype(str).str.extract(r"(\d+)\s*[/x]\s*(\d+)").astype(float)
    d["pa_sistolica"] = pa[0]
    d["pa_diastolica"] = pa[1]

    # --- fumante: never / former / current (com variações)
    fumo = df["SMOKER"].astype(str).str.strip().str.lower()
    d["fumante"] = fumo.map({"never": 0, "former": 1, "ex-smoker": 1, "current": 2, "yes": 2})

    # --- atividade física (tem erros de digitação tipo "Ligth", "Sedentry")
    ativ = df["PHYSICAL_ACTIVITY"].astype(str).str.strip().str.lower().str[:3]
    d["atividade"] = ativ.map({"sed": 0, "lig": 1, "mod": 2, "int": 3})

    # --- histórico familiar: Yes/Sim/Y/1 = 1, No/Não/N/0 = 0, Unknown = vazio
    hist = df["FAMILY_HISTORY"].astype(str).str.strip().str.lower()
    d["hist_familiar"] = np.select(
        [hist.isin(["yes", "y", "sim", "1"]), hist.str.startswith("n") | (hist == "0")],
        [1, 0],
        default=np.nan,
    )
    d.loc[hist == "nan", "hist_familiar"] = np.nan

    # --- glicose: a maioria em mg/dL, mas algumas em mmol/L (valores < 30).
    # 1 mmol/L de glicose = 18 mg/dL
    glic = pd.to_numeric(df["GLUCOSE"], errors="coerce")
    d["glicose"] = np.where(glic < 30, glic * 18, glic)

    # --- HEMOGLOBIN na verdade é a hemoglobina glicada (HbA1c). Em % fica
    # entre 4 e 15; os valores acima de 20 estão em mmol/mol (padrão IFCC).
    # Conversão: % = mmol/mol / 10.929 + 2.15
    hb = pd.to_numeric(df["HEMOGLOBIN"], errors="coerce")
    d["hba1c"] = np.where(hb > 20, hb / 10.929 + 2.15, hb)

    # triglicerídeos tem uns valores enormes, uso log pra diminuir o peso deles
    trig = pd.to_numeric(df["TRIGLYCERIDES"], errors="coerce")
    d["log_triglicerides"] = np.log1p(trig)
    d["sodio"] = pd.to_numeric(df["SODIUM"], errors="coerce")

    d["aparelho_b"] = (df["LAB_DEVICE"] == "FTIR-B").astype(int)

    # indica quando o exame de HbA1c ou glicose não foi feito. Olhando a base
    # toda, faltar HbA1c é igual nas duas classes (23,2% no controle e 23,7% no
    # diabetes). A diferença aparece só entre quem tem criterio_ada == 0: 24,1%
    # no controle contra 57,7% no diabetes. Por isso a coluna ainda é útil.
    d["sem_hba1c"] = d["hba1c"].isna().astype(int)
    d["sem_glicose"] = d["glicose"].isna().astype(int)

    # critério de diagnóstico da ADA (American Diabetes Association):
    # HbA1c >= 6.5% ou glicose em jejum >= 126 mg/dL. Ajudou bastante o KNN.
    d["criterio_ada"] = ((d["hba1c"] >= 6.5) | (d["glicose"] >= 126)).astype(int)
    return d


# colunas que entram nos modelos. Testei tirar e colocar uma de cada vez:
# sexo, altura, peso, sódio, fumo, aparelho etc. não ajudavam ou pioravam
# (principalmente o KNN). O histórico familiar foi a única que melhorou a MLP.
COLUNAS = [
    "hba1c", "glicose", "idade", "imc", "pa_sistolica", "log_triglicerides",
    "sem_hba1c", "sem_glicose", "criterio_ada", "hist_familiar",
]
