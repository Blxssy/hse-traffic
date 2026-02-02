from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler


class BuildFeaturesHandler(Handler):
    """
    Строим:
    - y: _salary_rub
    - x: числовые + one-hot по городу/занятости/графику
    """

    def _process(self, ctx: PipelineContext) -> None:
        if ctx.df is None:
            raise RuntimeError("df is not loaded")

        df = ctx.df

        # y
        y = df["_salary_rub"].to_numpy(dtype=np.float32)

        # базовые числовые
        numeric = df[["_gender", "_age", "_exp_months", "_has_car"]].astype("float32")

        # категориальные
        cats = df[["_city", "_employment", "_schedule"]].fillna("")
        dummies = pd.get_dummies(cats, columns=["_city", "_employment", "_schedule"], dummy_na=False)

        x_df = pd.concat([numeric, dummies], axis=1)

        x = x_df.to_numpy(dtype=np.float32)

        ctx.x = x
        ctx.y = y
