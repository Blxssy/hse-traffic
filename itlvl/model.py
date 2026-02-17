from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from itlvl.config import AppConfig


def build_pipeline(cfg: AppConfig) -> Pipeline:
    """
    - числовые признаки: salary/age/gender/exp
    - категориальные: city (one-hot)
    - текст: только title (TF-IDF), потому что skills часто пустой и ломает словарь
    """
    num_cols = ["salary_rub", "age", "gender", "exp_months"]
    cat_cols = ["city"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            (
                "title_tfidf",
                TfidfVectorizer(
                    max_features=cfg.max_tfidf_features,
                    min_df=1,
                ),
                "title",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="saga",
    )

    return Pipeline([("pre", pre), ("clf", clf)])


def train_and_eval(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: AppConfig,
) -> None:
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["title"] = X_train["title"].fillna("").astype(str)
    X_test["title"] = X_test["title"].fillna("").astype(str)

    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    print("\n=== Classification report ===")
    print(classification_report(y_test, pred, digits=3))

    print("=== Confusion matrix (rows=true, cols=pred) ===")
    labels = ["junior", "middle", "senior"]
    cm = confusion_matrix(y_test, pred, labels=labels)
    print(labels)
    print(cm)

    print("\n=== Notes / PoC conclusions ===")
    print("- Разметка y эвристическая (по названию должности и опыту), поэтому возможны шум и ошибки.")
    print("- Классы несбалансированы (обычно senior больше), используем class_weight='balanced'.")
    print("- Ошибки чаще всего: middle/junior и middle/senior из-за размытых границ уровней.")
    print("- Для улучшения: более качественная разметка, нормализация правил, добавление навыков/техстека как признаков.")
