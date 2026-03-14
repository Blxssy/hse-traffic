from __future__ import annotations

import mlflow


EXPERIMENT_NAME = "LIne Regression HH"

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.r2_score_test DESC"],
    max_results=10,
)

print(runs[["run_id", "metrics.r2_score_test"]].head(10).to_string(index=False))
print("\nBEST RUN_ID:", runs.iloc[0]["run_id"])
print("BEST r2_score_test:", runs.iloc[0]["metrics.r2_score_test"])