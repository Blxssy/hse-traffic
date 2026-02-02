from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler
from pipeline.utils import parse_salary_rub


class ParseTargetSalaryHandler(Handler):
    """
    y = зарплата в рублях.
    Отбрасываем строки, где зарплата не парсится.
    """

    SALARY_COL = "ЗП"

    def _process(self, ctx: PipelineContext) -> None:
        if ctx.df is None:
            raise RuntimeError("df is not loaded")

        df = ctx.df
        if self.SALARY_COL not in df.columns:
            raise KeyError(f"Column not found: {self.SALARY_COL}")

        salary = df[self.SALARY_COL].map(parse_salary_rub)
        mask = salary.notna()

        df = df.loc[mask].copy()
        df["_salary_rub"] = salary.loc[mask].astype("float32")

        ctx.df = df
