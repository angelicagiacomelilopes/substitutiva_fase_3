
# FIAP — Pós-graduação em Machine Learning Engineering (MLET)
## Trabalho Substitutiva — Fase 3

- **Turma:** 5MLET — 2026
- **Período para entrega:** 15/06 a 15/07
- **Protocolo de Solicitação:** 310903
- **Aluno:** Angélica Giacomeli Lopes
- **RM:** 363921

## Desafio

Desenvolver uma pipeline de um **modelo preditivo binário** utilizando uma [base de dados](/StudentsPrepared.xlsx) sobre desempenho acadêmico dos alunos e alunas de uma faculdade para prever a **evasão de estudantes**.

Para realizar esse trabalho, atenda aos seguintes requisitos:

- [ ] Realizar técnicas de feature engineering nos dados para tratar variáveis numéricas e categóricas.
- [ ] Realizar a separação da base em treino e teste.
- [ ] Treinar um modelo binário com estratégia de validação cruzada.
- [ ] Analisar os resultados do modelo, justificando se houve overfitting e/ou underfitting.
- [ ] Realizar o deploy do modelo utilizando a aplicação Streamlit.
- [ ] Disponibilizar o modelo em um repositório no GitHub com documentação dos passos desenvolvidos e conclusões da análise.
- [ ] Gravar um vídeo de no mínimo 5 minutos explicando as técnicas realizadas e apresentar a aplicação deployada no Streamlit.


A entrega deve ser um arquivo `.txt` com:

- link do repositório no GitHub;
- link da aplicação deployada no Streamlit;
- link do vídeo explicando a estratégia de Machine Learning utilizada e a apresentação da aplicação.

## Estrutura do projeto

- `StudentsPrepared.xlsx`: base de dados original
- `ml_utils.py`: funções utilitárias (carga de dados, criação da variável-alvo e feature engineering)
- `train_model.py`: pipeline de treino, validação cruzada, avaliação e persistência do modelo
- `app.py`: aplicação Streamlit para predição em lote e predição única
- `artifacts/model.joblib`: modelo treinado e metadados de features
- `artifacts/metrics.json`: métricas consolidadas
- `artifacts/classification_report.txt`: relatório de classificação e matriz de confusão

## Técnicas aplicadas

# Etapas para o desenvolvimento do trabalho

## 1) Definição do problema e alvo binário

Primeiro, defini o objetivo do projeto: prever evasão estudantil em formato binário.

- Classe positiva (`1`): `Desistente`
- Classe negativa (`0`): `Graduado` ou `Matriculado`


### 2) Separação das variáveis por tipo

Analise das colunas em variáveis numéricas e categóricas para aplicar o pré-processamento correto em cada grupo.

### Variáveis numéricas

- QualificacaoAnteriorGrau
- NotaAdmissao
- NecessidadesEspeciais
- Devedor
- MensalidadesEmDia
- Bolsista
- International
- UnidadesCurriculares1SemestreCreditado
- UnidadesCurriculares1SemestreInscrito
- UnidadesCurriculares1SemestreAvaliacoes
- UnidadesCurriculares1SemestreAprovado
- UnidadesCurriculares1SemestreGrau
- UnidadesCurriculares1SemestreSemAvaliacoes
- UnidadesCurriculares2SemestreCreditado
- UnidadesCurriculares2SemestreInscrito
- UnidadesCurriculares2SemestreAvaliacoes
- UnidadesCurriculares2SemestreAprovado
- UnidadesCurriculares2SemestreGrau
- UnidadesCurriculares2SemestreSemAvaliacoes
- TaxaDesemprego
- TaxaInflacao
- PIB

### Variáveis categóricas

- EstadoCivil
- Curso
- QualificacaoAnterior
- Nacionalidade
- Genero
- Target (coluna original, usada para gerar o alvo binário)

### 3) Verificação e tratamento de dados faltantes

Análise de qualidade da base
Dados estão completos, sem valores ausentes (`NaN`/`None`).

Adicionada rotirnas para tratamentos de dados numéricos e categóricos, casos estejam com valores ausentes

- **Numéricos:** `SimpleImputer(strategy="median")` - mediana + padronização (`StandardScaler`)
- **Categóricos:** `SimpleImputer(strategy="most_frequent")` - moda + codificação (`OneHotEncoder`) 

### 4) Feature Engineering

Foram criadas features derivadas para enriquecer o sinal preditivo, com base no objetivo de prever evasão, representando desempenho acadêmico e engajamento do estudante.

- **Taxas de aprovação por semestre**
  - `TaxaAprovacao1Sem` = `Aprovado1 / Inscrito1`
  - `TaxaAprovacao2Sem` = `Aprovado2 / Inscrito2`
- **Taxas de avaliação por semestre**
- `TaxaAvaliacoes1Sem` = `Avaliacoes1 / Inscrito1`
- `TaxaAvaliacoes2Sem` = `Avaliacoes2 / Inscrito2`
- **Evolução acadêmica entre semestres**
  - `EvolucaoGrauSemestres` = `Grau2 - Grau1`
- **Desempenho acumulado**
  - `TotalAprovacoes2Semestres` = `Aprovado1 + Aprovado2`


### 5) Separação treino/teste

- Divisão estratificada: **80% treino / 20% teste**
- `train_test_split(test_size=0.2, stratify=y, random_state=42)`


### 6) Treinamento com validação cruzada

Utilizado Regressão Logistica com balanceamento de clases

- Modelo: `LogisticRegression(max_iter=2000, class_weight="balanced")`
- Estratégia: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Métrica principal da validação: **F1-score**


### 7) Resultados obtidos

As métricas analisadas foram:

- Accuracy =  % de acertos totais (desistentes e não-desistentes juntos)
- Precision = De quem o modelo disse que era desistente, quantos realmente eram?
- Recall = De todos os desistentes reais, quantos o modelo conseguiu identificar?
- F1-score = Média harmônica entre precision e recall — balanço entre os dois.
- ROC-AUC = Capacidade geral do modelo de separar as duas classes.

Resultados obtidos (arquivo `artifacts/metrics.json`):

|Métrica| Valor | SIgnificado | Objetivo |
|-------|-------|-------------|----------|
|`cv_train_f1_mean`| 0.7993 | F1 médio nos dados de treino dos 5 folds | mede o quanto o modelo aprendeu|
|`cv_valid_f1_mean`|0.7826  | F1 médio nos dados de validação dos 5 folds | mede o quanto o modelo generaliza |
|`cv_valid_f1_std`| 0.0201 | Desvio padrão do F1 nos dados de validação | mede a variabilidade do modelo |
| `test_accuracy`| 0.8768 | Acurácia nos dados de teste | mede a proporção de previsões corretas |
| `test_precision`| 0.7966 | Precisão nos dados de teste | mede a proporção de verdadeiros positivos entre os positivos previstos|
| `test_recall`| 0.8275 | Recall nos dados de teste | mede a proporção de verdadeiros positivos entre os positivos reais|
| `test_f1`| 0.8117 | F1 nos dados de teste | mede a média harmônica entre precisão e recall|
| `test_roc_auc`| 0.9325 | ROC AUC nos dados de teste | mede a capacidade do modelo de distinguir entre classes|
| `overfitting_gap`| 0.0167 | Diferença entre F1 de treino e validação | mede o possível overfitting|
| `fit_diagnosis`| 0.0167 | Diagnóstico do ajuste do modelo | indica overfitting, underfitting ou generalização aceitável|


|Métrica |	REsultado |	Analise|
|---------|-----------|---------|
|Accuracy	|0.8768	    |Em 877 predições, 876 acertaram |
|Precision|	0.7966	  |De quem o modelo disse "vai desistir", 79.66% realmente desistirão| 
|Recall	  |0.8275	    |De 253 desistentes reais, o modelo capturou 209 |
|F1	      |0.8117	    |Bom balanço entre precision e recall|
|AUC-ROC  |	0.9325	  |Excelente capacidade de separar desistentes de não-desistente|

Metricas:
  

Diferença entre F1 de treino e validação | mede o possível overfitting|




### 8) Análise de overfitting e underfitting

Comparei os resultados de treino, validação e teste para verificar capacidade de generalização.

- O gap entre treino e validação foi baixo (`0.0167`).
- O F1 de teste (`0.8117`) ficou consistente com validação (`0.7826`).

Conclusão: **não houve evidência forte de overfitting ou underfitting**, indicando bom ajuste do modelo, ou seja, isso indica **boa capacidade de generalização**, sem sinais fortes de overfitting.

Também não há indícios de underfitting, pois as métricas absolutas estão em patamar elevado (AUC > 0.93 e F1 > 0.81 no teste).

### 9) Deploy e entrega final

- Repositório no GitHub com documentação
- Aplicação deployada no Streamlit
- Vídeo de no mínimo 5 minutos explicando técnicas e demonstração
- Arquivo `.txt` com os 3 links solicitados



### 10) Como executar localmente

### 1. Instalar dependências

```bash
python -m pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python train_model.py
```

### 3. Rodar o Streamlit

```bash
python -m streamlit run app.py
```

Se a porta padrão estiver em uso, execute com uma porta alternativa:

```bash
python -m streamlit run app.py --server.port 8510
```

A aplicação abrirá no navegador e permitirá:
- Upload de arquivo (`.csv`/`.xlsx`) para predição em lote
- Simulação rápida via formulário para um estudante

## Deploy no Streamlit Cloud

1. Acesse https://substitutivafase3-angelica-giacomeli.streamlit.app/

##  Links do trabalho

LINK_REPOSITORIO_GITHUB: https://github.com/angelicagiacomelilopes/substitutiva_fase_3

LINK_STREAMLIT_DEPLOY: https://substitutivafase3-angelica-giacomeli.streamlit.app/

LINK_VIDEO_APRESENTACAO: falta o video

