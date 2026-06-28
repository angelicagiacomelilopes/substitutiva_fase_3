from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml_utils import add_engineered_features, build_binary_target, load_dataset


@dataclass
class Metrics:
    cv_train_f1_mean: float 
    cv_valid_f1_mean: float 
    cv_valid_f1_std: float 
    test_accuracy: float  
    test_precision: float  
    test_recall: float  
    test_f1: float  
    test_roc_auc: float  
    overfitting_gap: float 
    fit_diagnosis: str 

def infer_fit_diagnosis(train_f1: float, valid_f1: float, test_f1: float) -> str:
    # Analisar os valores nos dados de treino
    
    gap = train_f1 - valid_f1
    if gap > 0.08 and valid_f1 > 0.65:
        return "Possível overfitting: desempenho no treino está significativamente maior que validação."
    if train_f1 < 0.60 and valid_f1 < 0.60 and test_f1 < 0.60:
        return "Possível underfitting: desempenho baixo tanto em treino quanto validação/teste."
    return "Sem evidência forte de overfitting/underfitting; modelo com generalização aceitável."


def build_pipeline(categorical_cols: list[str], numerical_cols: list[str]) -> Pipeline:
    # construir pré-processamento e modelagem para um modelo de classificação.

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    # Realiza treinamento 
    data_path = "StudentsPrepared.xlsx"
    artifacts_dir = "artifacts" 
    os.makedirs(artifacts_dir, exist_ok=True) 

    df = load_dataset(data_path)
    y = build_binary_target(df) 
    X = add_engineered_features(df) 

    # colunas categóricas e numéricas 
    categorical_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object", "string", "category", "bool"]).columns.tolist()

    # treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(categorical_cols, numerical_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        return_train_score=True,
        n_jobs=-1,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    train_f1_mean = float(np.mean(cv_results["train_score"]))
    valid_f1_mean = float(np.mean(cv_results["test_score"]))
    valid_f1_std = float(np.std(cv_results["test_score"]))

    metrics = Metrics(
        cv_train_f1_mean=train_f1_mean,
        cv_valid_f1_mean=valid_f1_mean,
        cv_valid_f1_std=valid_f1_std,
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        test_precision=float(precision_score(y_test, y_pred, zero_division=0)),
        test_recall=float(recall_score(y_test, y_pred, zero_division=0)),
        test_f1=float(f1_score(y_test, y_pred, zero_division=0)),
        test_roc_auc=float(roc_auc_score(y_test, y_prob)),
        overfitting_gap=float(train_f1_mean - valid_f1_mean),
        fit_diagnosis=infer_fit_diagnosis(train_f1_mean, valid_f1_mean, float(f1_score(y_test, y_pred, zero_division=0))),
    )

    payload = {
        "model": pipeline,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
    }
    
    model_path = os.path.join(artifacts_dir, "model.joblib")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    report_path = os.path.join(artifacts_dir, "classification_report.txt")

    joblib.dump(payload, model_path)

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(asdict(metrics), fp, ensure_ascii=False, indent=2)

    with open(report_path, "w", encoding="utf-8") as fp:
        fp.write(classification_report(y_test, y_pred, digits=4))
        fp.write("\n\nMatriz de confusão:\n")
        fp.write(str(confusion_matrix(y_test, y_pred)))

    print("Treinamento concluído com sucesso.")
    print(f"Modelo salvo em: {model_path}")
    print(f"Métricas salvas em: {metrics_path}")
    print(f"Relatório de classificação salvo em: {report_path}")
    print("Resumo métricas:")
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
