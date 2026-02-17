from __future__ import annotations

import argparse
from pathlib import Path

from itlvl.config import AppConfig
from itlvl.dataset import load_hh_csv, make_it_dataset
from itlvl.features import build_train_test
from itlvl.model import train_and_eval
from itlvl.plots import plot_class_balance


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="app",
        description="PoC: classify IT developer level from hh.ru resumes",
    )
    parser.add_argument("csv_path", type=str, help="Path to hh.csv")
    args = parser.parse_args()

    cfg = AppConfig()
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_hh_csv(csv_path)
    it_df = make_it_dataset(df, cfg)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # plot balance
    plot_class_balance(it_df["level"], reports_dir / "class_balance.png")

    # build data for ML
    X_train, X_test, y_train, y_test = build_train_test(it_df, cfg)

    # train
    train_and_eval(X_train, X_test, y_train, y_test, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
