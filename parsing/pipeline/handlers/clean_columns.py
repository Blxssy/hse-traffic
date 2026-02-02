from __future__ import annotations

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler


class CleanColumnsHandler(Handler):
    def _process(self, ctx: PipelineContext) -> None:
        if ctx.df is None:
            raise RuntimeError("df is not loaded")

        df = ctx.df

        drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # нормализуем имена колонок
        df.columns = [str(c).strip() for c in df.columns]

        ctx.df = df
