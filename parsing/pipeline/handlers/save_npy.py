from __future__ import annotations

import numpy as np

from pipeline.context import PipelineContext
from pipeline.handlers.base import Handler


class SaveNpyHandler(Handler):
    def _process(self, ctx: PipelineContext) -> None:
        if ctx.x is None or ctx.y is None:
            raise RuntimeError("x/y not built")

        x_path = ctx.out_dir / "x_data.npy"
        y_path = ctx.out_dir / "y_data.npy"

        np.save(x_path, ctx.x)
        np.save(y_path, ctx.y)
