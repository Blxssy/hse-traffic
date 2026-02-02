from __future__ import annotations

from pathlib import Path
import numpy as np

from hhreg.model import RidgeRegressor


def save_model(path: Path, model: RidgeRegressor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w=model.w,
        b=np.array([model.b], dtype=np.float32),
        mean=model.mean,
        std=model.std,
        version=np.array([1], dtype=np.int32),
    )


def load_model(path: Path) -> RidgeRegressor:
    data = np.load(path)
    w = data["w"].astype(np.float32, copy=False)
    b = float(data["b"].reshape(-1)[0])
    mean = data["mean"].astype(np.float32, copy=False)
    std = data["std"].astype(np.float32, copy=False)
    return RidgeRegressor(w=w, b=b, mean=mean, std=std)
