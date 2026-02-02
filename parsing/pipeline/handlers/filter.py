from __future__ import annotations

import numpy as np

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler


class FilterRowsHandler(Handler):
    """
    Чистит датасет перед построением фичей:
    - фильтрует аномальную зарплату
    - клипает опыт в месяцах
    """

    def __init__(self, min_salary: float = 10_000.0, max_salary: float = 1_000_000.0, max_exp_months: int = 720):
        super().__init__()
        self._min_salary = float(min_salary)
        self._max_salary = float(max_salary)
        self._max_exp = int(max_exp_months)

    def _process(self, ctx: PipelineContext) -> None:
        if ctx.df is None:
            raise RuntimeError("df is not loaded")

        df = ctx.df

        required = ["_salary_rub", "_exp_months"]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Missing column {col}. Ensure previous handlers created it.")

        # 1) Фильтр по зарплате
        df = df[(df["_salary_rub"] >= self._min_salary) & (df["_salary_rub"] <= self._max_salary)].copy()

        # 2) Клип опыта (защита от улетающих значений)
        # Важно: _exp_months может быть float из-за NaN раньше, приводим аккуратно.
        df["_exp_months"] = np.clip(df["_exp_months"].astype("float32"), 0, self._max_exp).astype("float32")

        ctx.df = df
