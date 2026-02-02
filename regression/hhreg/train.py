from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hhreg.model import RidgeRegressor
from hhreg.io import save_model


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hhreg.train",
        description="Train ridge regression on x_data.npy + y_data.npy and save weights to resources/model.npz",
    )
    parser.add_argument("x_path", type=str, help="Path to x_data.npy")
    parser.add_argument("y_path", type=str, help="Path to y_data.npy")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength")
    args = parser.parse_args()

    x_path = Path(args.x_path).expanduser().resolve()
    y_path = Path(args.y_path).expanduser().resolve()
    if not x_path.exists():
        raise FileNotFoundError(f"x_data.npy not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"y_data.npy not found: {y_path}")

    x = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.float32, copy=False)

    model = RidgeRegressor.fit(x, y, alpha=float(args.alpha))

    project_dir = Path(__file__).resolve().parents[1]
    out_path = project_dir / "resources" / "model.npz"
    save_model(out_path, model)

    print(f"Saved model to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
