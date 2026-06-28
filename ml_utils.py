from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "Target"
DROP_TARGET_VALUE = "Desistente"


def load_dataset(path: str) -> pd.DataFrame:
    # Essa função tem como objetivo carregar o arquivo (StudentsPrepared.xlsx)
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Formato de arquivo não suportado. Use .xlsx ou .csv")


def build_binary_target(df: pd.DataFrame) -> pd.Series:
    # Essa função tem como objetivo criar a coluna alvo binária, onde 1 representa "Desistente" e 0 representa "Não Desistente".
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' não encontrada.")
    return (df[TARGET_COLUMN].astype(str).str.strip() == DROP_TARGET_VALUE).astype(int)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Essa função tem como objetivo adicionar colunas derivadas (features) ao DataFrame.
    data = df.copy()

    def safe_ratio(num_col: str, den_col: str, output_col: str) -> None:
        # Essa função calcula a razão entre duas colunas, tratando divisões por zero e valores ausentes.
        if num_col in data.columns and den_col in data.columns:
            denominator = data[den_col].replace(0, np.nan)
            data[output_col] = (data[num_col] / denominator).fillna(0.0)

    # Taxas de aprovação 1 Semestre
    safe_ratio(
        "UnidadesCurriculares1SemestreAprovado",
        "UnidadesCurriculares1SemestreInscrito",
        "TaxaAprovacao1Sem",
    )
    # Taxas de aprovação 2 Semestre
    safe_ratio(
        "UnidadesCurriculares2SemestreAprovado",
        "UnidadesCurriculares2SemestreInscrito",
        "TaxaAprovacao2Sem",
    )
    # Taxas de avaliacao 1 Semestre
    safe_ratio(
        "UnidadesCurriculares1SemestreAvaliacoes",
        "UnidadesCurriculares1SemestreInscrito",
        "TaxaAvaliacoes1Sem",
    )
    # Taxas de avaliacao 2 Semestre
    safe_ratio(
        "UnidadesCurriculares2SemestreAvaliacoes",
        "UnidadesCurriculares2SemestreInscrito",
        "TaxaAvaliacoes2Sem",
    )

    # Evolução entre  semestres
    # Valor positivo: aluno melhorou do 1º para o 2º semestre.
    # Valor zero: manteve o mesmo desempenho.
    # Valor negativo: piorou no 2º semestre.
    if {
        "UnidadesCurriculares1SemestreGrau",
        "UnidadesCurriculares2SemestreGrau",
    }.issubset(data.columns):
        data["EvolucaoGrauSemestres"] = (
            data["UnidadesCurriculares2SemestreGrau"]
            - data["UnidadesCurriculares1SemestreGrau"]
        )

    # Total de aprovações nos 2 semestres
    # soma as aprovações dos dois semestres para medir o desempenho acumulado do aluno.
    if {
        "UnidadesCurriculares1SemestreAprovado",
        "UnidadesCurriculares2SemestreAprovado",
    }.issubset(data.columns):
        data["TotalAprovacoes2Semestres"] = (
            data["UnidadesCurriculares1SemestreAprovado"]
            + data["UnidadesCurriculares2SemestreAprovado"]
        )

    # Remover a coluna alvo original
    if TARGET_COLUMN in data.columns:
        data = data.drop(columns=[TARGET_COLUMN])

    return data
