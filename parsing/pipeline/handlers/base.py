from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pipeline.context import PipelineContext


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Optional["Handler"] = None

    def set_next(self, nxt: "Handler") -> "Handler":
        self._next = nxt
        return nxt

    def handle(self, ctx: PipelineContext) -> None:
        self._process(ctx)
        if self._next is not None:
            self._next.handle(ctx)

    @abstractmethod
    def _process(self, ctx: PipelineContext) -> None:
        raise NotImplementedError
