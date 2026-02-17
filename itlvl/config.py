from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    test_size: float = 0.2
    random_state: int = 42

    junior_max_months: int = 24
    senior_min_months: int = 60

    max_tfidf_features: int = 50_000
    min_df: int = 3
