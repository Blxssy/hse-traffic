from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RidgeRegressor:
    """
    Линейная регрессия с L2-регуляризацией (Ridge) и стандартизацией.
    """
    w: np.ndarray
    b: float
    mean: np.ndarray
    std: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = self._as_2d(x)
        x_norm = (x - self.mean) / self.std
        return x_norm @ self.w + self.b

    @staticmethod
    def fit(x: np.ndarray, y: np.ndarray, alpha: float = 10.0) -> "RidgeRegressor":
        x = RidgeRegressor._as_2d(x).astype(np.float64, copy=False)
        y = y.reshape(-1).astype(np.float64, copy=False)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"X rows != y size: {x.shape[0]} != {y.shape[0]}")

        # стандартизация
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0

        x_norm = (x - mean) / std

        # добавим bias через центрирование y: b = mean(y) - mean(Xw)
        y_mean = float(y.mean())
        y_center = y - y_mean

        d = x_norm.shape[1]
        xtx = x_norm.T @ x_norm
        reg = alpha * np.eye(d, dtype=np.float64)
        xty = x_norm.T @ y_center

        w = np.linalg.solve(xtx + reg, xty)

        # bias возвращаем в исходное пространство
        # pred = X_norm w + b; хотим среднее pred = y_mean -> b = y_mean
        b = y_mean

        return RidgeRegressor(
            w=w.astype(np.float32),
            b=float(b),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )

    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"X must be 2D array, got ndim={x.ndim}")
        return x
