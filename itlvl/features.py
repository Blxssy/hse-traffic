from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from itlvl.config import AppConfig


def build_train_test(it_df: pd.DataFrame, cfg: AppConfig):
    X = it_df.drop(columns=["level"])
    y = it_df["level"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
