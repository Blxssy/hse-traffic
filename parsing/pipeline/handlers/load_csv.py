from __future__ import annotations

import pandas as pd

from parsing.pipeline.context import PipelineContext
from parsing.pipeline.handlers.base import Handler


class LoadCSVHandler(Handler):
    def _process(self, ctx: PipelineContext) -> None:
        df = pd.read_csv(
            ctx.csv_path,
            engine="python",
            sep=",",
            quotechar='"',
            encoding="utf-8",
            on_bad_lines="skip",
        )
        ctx.df = df
