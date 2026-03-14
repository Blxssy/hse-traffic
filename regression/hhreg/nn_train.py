from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hhreg.nn import FCNRegressor
from hhreg.nn_io import Scaler, save_model, save_scaler


mlflow.set_tracking_uri("http://kamnsv.com:55000")
EXPERIMENT_NAME = "LIne Regression HH"

@dataclass(frozen=True)
class TrainConfig:
    registered_model_name: str = "roman_savelev_fcn"

    test_size: float = 0.2
    random_state: int = 42

    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.05

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 80

    use_log_target: bool = True


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hhreg.nn_train",
        description="Train FCN regressor + track experiments in MLflow",
    )
    parser.add_argument("x_path", type=str, help="Path to x_data.npy")
    parser.add_argument("y_path", type=str, help="Path to y_data.npy")
    parser.add_argument("--model-name", type=str, default=None, help="Override registered model name (<fio>_fcn)")
    args = parser.parse_args()

    cfg = TrainConfig(
        registered_model_name=args.model_name or TrainConfig.registered_model_name
    )

    _seed_everything(cfg.random_state)

    x = np.load(Path(args.x_path)).astype(np.float32, copy=False)
    y = np.load(Path(args.y_path)).astype(np.float32, copy=False).reshape(-1)

    # простая защита от NaN/Inf
    row_ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[row_ok]
    y = y[row_ok]

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # scaler по train
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    scaler = Scaler(mean=mean, std=std, use_log_target=cfg.use_log_target)

    x_train_n = scaler.transform(x_train)
    x_test_n = scaler.transform(x_test)

    # target transform
    y_train_t = np.log1p(y_train) if cfg.use_log_target else y_train
    y_test_t = np.log1p(y_test) if cfg.use_log_target else y_test

    # torch datasets
    ds_train = TensorDataset(
        torch.from_numpy(x_train_n), torch.from_numpy(y_train_t.reshape(-1, 1))
    )
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)

    model = FCNRegressor(
        input_dim=x.shape[1],
        hidden_dims=list(cfg.hidden_dims),
        dropout=cfg.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    mlflow.set_experiment(EXPERIMENT_NAME)

    resources_dir = Path(__file__).resolve().parents[1] / "resources"
    model_path = resources_dir / "model.pt"
    scaler_path = resources_dir / "scaler.npz"

    with mlflow.start_run(run_name=cfg.registered_model_name) as run:
        # params
        mlflow.log_params(
            {
                "model_type": "fcn",
                "hidden_dims": list(cfg.hidden_dims),
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "use_log_target": cfg.use_log_target,
                "seed": cfg.random_state,
                "n_features": int(x.shape[1]),
                "train_rows": int(x_train.shape[0]),
                "test_rows": int(x_test.shape[0]),
            }
        )

        # training loop
        model.train()
        for epoch in range(1, cfg.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in dl_train:
                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item()) * xb.size(0)

            epoch_loss /= len(ds_train)
            mlflow.log_metric("train_mse", epoch_loss, step=epoch)

        # eval on test
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x_test_n).to(device)
            pred_t = model(xt).detach().cpu().numpy().reshape(-1)

        # inverse target transform to salary space
        y_pred = np.expm1(pred_t) if cfg.use_log_target else pred_t
        y_pred = np.clip(y_pred, 0, None)

        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("r2_score_test", float(r2))

        # save artifacts to resources (repo requirement)
        save_model(model_path, model.cpu())
        save_scaler(scaler_path, scaler)

        mlflow.log_artifact(str(model_path), artifact_path="resources")
        mlflow.log_artifact(str(scaler_path), artifact_path="resources")

        # log model to MLflow + register
        mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            artifact_path="model",
            registered_model_name=cfg.registered_model_name,
        )

        # run summary artifact
        summary = {
            "run_id": run.info.run_id,
            "experiment": EXPERIMENT_NAME,
            "registered_model_name": cfg.registered_model_name,
            "r2_score_test": float(r2),
        }
        tmp = Path("run_summary.json")
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(tmp), artifact_path="meta")
        tmp.unlink(missing_ok=True)

        print("RUN_ID:", run.info.run_id)
        print("r2_score_test:", float(r2))
        print("Saved:", model_path)
        print("Saved:", scaler_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())