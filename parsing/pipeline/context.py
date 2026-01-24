from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    csv_path: Path
    df: Optional[pd.DataFrame] = None

    # итоговые массивы
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    # куда сохраняем .npy
    out_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out_dir = self.csv_path.parent
