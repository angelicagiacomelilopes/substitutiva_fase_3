from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

try:
    from src.ml_utils import add_engineered_features
except ModuleNotFoundError:
    from ml_utils import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Configurações iniciais da página
st.set_page_config(page_title="Previsão de Evasão", layout="wide")
st.title("Previsão de Evasão de Estudantes")
st.write("Aplicação para inferência de risco de evasão utilizando modelo de classificação binária.")

# Função para carregar o modelo e as métricas, com cache para otimizar desempenho
@st.cache_resource
def load_artifacts():
    # Carrega o modelo e as colunas de features do arquivo joblib
    payload = joblib.load(ARTIFACTS_DIR / "model.joblib")
    metrics = {}
    try:
        with open(ARTIFACTS_DIR / "metrics.json", "r", encoding="utf-8") as fp:
            metrics = json.load(fp)
    except FileNotFoundError:
        metrics = {}
    return payload, metrics


try:
    payload, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Modelo não encontrado. Execute `python src/train_model.py` antes de iniciar o app.")
    st.stop()

# Extrai o modelo e as colunas de features do payload carregado
model = payload["model"]
feature_columns = payload["feature_columns"]

# Exibe as métricas do modelo em um expander para melhor organização da interface
with st.expander("Métricas do modelo", expanded=True):
    if metrics:
        st.json(metrics)
    else:
        st.info("Arquivo de métricas não encontrado.")

# Seção para predição em lote via upload de arquivo CSV/XLSX
st.subheader("Predição em lote (upload de CSV/XLSX)")
uploaded_file = st.file_uploader("Envie um arquivo com as colunas originais da base", type=["csv", "xlsx"])

# Processa o arquivo enviado, gera as predições e exibe os resultados, além de oferecer opção de download dos resultados em CSV
if uploaded_file is not None:
    if uploaded_file.name.endswith(".xlsx"):
        input_df = pd.read_excel(uploaded_file)
    else:
        input_df = pd.read_csv(uploaded_file)
    # Aplica as mesmas transformações de engenharia de features usadas no treinamento
    processed_df = add_engineered_features(input_df)
    # Verifica se todas as colunas necessárias para a predição estão presentes após o processamento
    missing_cols = [col for col in feature_columns if col not in processed_df.columns]
    if missing_cols:
        st.error(f"Colunas ausentes no arquivo enviado: {missing_cols}")
    else:
        # Gera as predições utilizando o modelo carregado
        X_pred = processed_df[feature_columns].copy()
        probs = model.predict_proba(X_pred)[:, 1]
        preds = (probs >= 0.5).astype(int)
        # Cria um DataFrame de resultados combinando as predições com os dados originais para exibição e download
        result_df = input_df.copy()
        result_df["ProbabilidadeEvasao"] = probs
        result_df["PredicaoEvasao"] = preds
        result_df["ClassePredita"] = result_df["PredicaoEvasao"].map({1: "Desistente", 0: "Não Desistente"})
        # Exibe as predições geradas e oferece opção de download dos resultados em CSV
        st.success("Predições geradas com sucesso.")
        st.dataframe(result_df.head(50), use_container_width=True)
        # Prepara o buffer para download do CSV
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        # Botão para download do arquivo CSV com as predições
        st.download_button(
            label="Baixar resultados em CSV",
            data=csv_buffer.getvalue(),
            file_name="predicoes_evasao.csv",
            mime="text/csv",
        )
# Seção para predição única via formulário, permitindo simulação rápida com campos principais
st.subheader("Predição única (formulário)")
st.caption("Preencha os campos principais para uma simulação rápida.")
# Formulário para entrada de dados para predição única, com campos principais para simulação rápida
with st.form("single_prediction"):
    nota = st.number_input("Nota de Admissão", min_value=0.0, max_value=200.0, value=120.0)
    devedor = st.selectbox("Devedor", [0, 1], index=0)
    mensalidades = st.selectbox("Mensalidades em dia", [0, 1], index=1)
    genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
    curso = st.text_input("Curso", value="Turismo")
    nacionalidade = st.text_input("Nacionalidade", value="Português")
    estado_civil = st.text_input("Estado Civil", value="Solteiro")
    bolsista = st.selectbox("Bolsista", [0, 1], index=0)
    international = st.selectbox("International", [0, 1], index=0)
    aprovado1 = st.number_input("Aprovado 1º semestre", min_value=0, max_value=20, value=6)
    inscrito1 = st.number_input("Inscrito 1º semestre", min_value=0, max_value=20, value=6)
    aprovado2 = st.number_input("Aprovado 2º semestre", min_value=0, max_value=20, value=6)
    inscrito2 = st.number_input("Inscrito 2º semestre", min_value=0, max_value=20, value=6)

    submitted = st.form_submit_button("Prever")
# Quando o formulário é submetido, cria um template de dados com os valores inseridos, aplica as mesmas transformações de engenharia de features usadas no treinamento, gera a predição e exibe a probabilidade de evasão e a classe predita
if submitted:
    template = {col: None for col in feature_columns}
    template.update(
        {
            "EstadoCivil": estado_civil,
            "Curso": curso,
            "QualificacaoAnterior": "Ensino Secundário",
            "QualificacaoAnteriorGrau": 120.0,
            "Nacionalidade": nacionalidade,
            "NotaAdmissao": nota,
            "NecessidadesEspeciais": 0,
            "Devedor": devedor,
            "MensalidadesEmDia": mensalidades,
            "Genero": genero,
            "Bolsista": bolsista,
            "International": international,
            "UnidadesCurriculares1SemestreCreditado": 0,
            "UnidadesCurriculares1SemestreInscrito": inscrito1,
            "UnidadesCurriculares1SemestreAvaliacoes": inscrito1,
            "UnidadesCurriculares1SemestreAprovado": aprovado1,
            "UnidadesCurriculares1SemestreGrau": 12.0,
            "UnidadesCurriculares1SemestreSemAvaliacoes": 0,
            "UnidadesCurriculares2SemestreCreditado": 0,
            "UnidadesCurriculares2SemestreInscrito": inscrito2,
            "UnidadesCurriculares2SemestreAvaliacoes": inscrito2,
            "UnidadesCurriculares2SemestreAprovado": aprovado2,
            "UnidadesCurriculares2SemestreGrau": 12.0,
            "UnidadesCurriculares2SemestreSemAvaliacoes": 0,
            "TaxaDesemprego": 10.8,
            "TaxaInflacao": 1.4,
            "PIB": 1.74,
        }
    )
    
    # Cria um DataFrame com o template de dados, aplica as transformações de engenharia de features, seleciona as colunas de features necessárias para a predição, gera a probabilidade de evasão e a classe predita, e exibe os resultados
    single_df = pd.DataFrame([template])
    single_df = add_engineered_features(single_df)
    single_df = single_df[feature_columns]

    prob = float(model.predict_proba(single_df)[0, 1])
    pred = int(prob >= 0.5)

    st.metric("Probabilidade de Evasão", f"{prob:.2%}")
    st.write("Classe predita:", "Desistente" if pred == 1 else "Não Desistente")
