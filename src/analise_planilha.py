# Análise da planilha "StudentsPrepared.xlsx"
# Codigo para analisar a planilha, verificar dimensões, dados faltantes, amostra das primeiras linhas e tipos de dados.

import pandas as pd
import numpy as np

df = pd.read_excel("C:\\Users\\Angélica\\Desktop\\fiap\\StudentsPrepared.xlsx")

print("DIMENSÕES:")
print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
print("\nDADOS FALTANTES (NaN/None):")
missing = df.isnull().sum()
missing_with_pct = pd.DataFrame({
    "Coluna": missing.index,
    "Faltantes": missing.values,
    "Percentual": (missing.values / len(df) * 100).round(2)
})
missing_with_pct = missing_with_pct[missing_with_pct["Faltantes"] > 0]

if len(missing_with_pct) == 0:
    print("✓ Nenhum dado faltante detectado!")
else:
    print(missing_with_pct.to_string(index=False))

print("\n\nAMOSTRA DAS PRIMEIRAS LINHAS:")
print(df.head(3).to_string())

print("\n\nTIPOS DE DADOS:")
print(df.dtypes)