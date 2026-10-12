"""Compare two regressors, select on validation MAE, and track with MLflow."""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_data(seed=42):
    """Create disjoint 60/20/20 partitions before fitting any preprocessing."""
    dataset = load_diabetes(as_frame=True, scaled=False)
    features, target = dataset.data, dataset.target
    development_ids, test_ids = train_test_split(
        features.index.to_numpy(), test_size=0.20, random_state=seed
    )
    train_ids, validation_ids = train_test_split(
        development_ids, test_size=0.25, random_state=seed
    )
    return features, target, {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }


def build_candidates(seed=42, ridge_alpha=1.0, trees=100):
    """Only the linear model needs scaling; its pipeline fits the scaler safely."""
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)),
        "random_forest": RandomForestRegressor(
            n_estimators=trees,
            max_depth=5,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        ),
    }


def regression_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def select_winner(comparison):
    """Choose by validation MAE only; the name provides a deterministic tie-break."""
    return min(comparison, key=lambda row: (row["validation_mae"], row["model"]))[
        "model"
    ]


def train(output_dir, model_output, seed=42, ridge_alpha=1.0, trees=100):
    output_dir = Path(output_dir)
    model_output = Path(model_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # MLflow save_model requires an empty destination. Fail before touching test data.
    if model_output.exists() and any(model_output.iterdir()):
        raise ValueError(f"Use an empty model output directory: {model_output}")

    features, target, split = split_data(seed)
    train_ids, validation_ids, test_ids = (
        split["train"], split["validation"], split["test"]
    )
    candidates = build_candidates(seed, ridge_alpha, trees)
    comparison = []

    # Azure injects its tracking URI and run ID. Local runs use the caller's URI.
    with mlflow.start_run() as parent_run:
        mlflow.set_tags({"project": "azure-ml-mlflow-lab", "purpose": "education"})
        mlflow.log_params(
            {
                "dataset": "sklearn_diabetes_raw",
                "seed": seed,
                "train_rows": len(train_ids),
                "validation_rows": len(validation_ids),
                "test_rows": len(test_ids),
                "selection_metric": "validation_mae",
            }
        )
        baseline = DummyRegressor(strategy="mean").fit(
            features.loc[train_ids], target.loc[train_ids]
        )
        baseline_mae = mean_absolute_error(
            target.loc[validation_ids], baseline.predict(features.loc[validation_ids])
        )
        mlflow.log_metric("baseline_validation_mae", float(baseline_mae))

        # The two candidates share exactly the same training and validation rows.
        for name, estimator in candidates.items():
            with mlflow.start_run(run_name=name, nested=True) as candidate_run:
                mlflow.log_param("model", name)
                mlflow.log_params(estimator.get_params(deep=True))
                estimator.fit(features.loc[train_ids], target.loc[train_ids])
                predicted = estimator.predict(features.loc[validation_ids])
                metrics = regression_metrics(target.loc[validation_ids], predicted)
                mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
                example = features.loc[train_ids].head(3)
                mlflow.sklearn.log_model(
                    estimator,
                    artifact_path="model",
                    signature=infer_signature(example, estimator.predict(example)),
                    input_example=example,
                )
                comparison.append(
                    {
                        "model": name,
                        "run_id": candidate_run.info.run_id,
                        **{f"validation_{k}": v for k, v in metrics.items()},
                    }
                )

        # Select before looking at test labels. Refit the same configuration on 80%.
        winner_name = select_winner(comparison)
        development_ids = np.concatenate([train_ids, validation_ids])
        winner = clone(candidates[winner_name]).fit(
            features.loc[development_ids], target.loc[development_ids]
        )
        test_predictions = winner.predict(features.loc[test_ids])
        test_metrics = regression_metrics(target.loc[test_ids], test_predictions)
        mlflow.log_param("selected_model", winner_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # This model folder is an Azure ML named output and can be registered later.
        example = features.loc[train_ids].head(3)
        mlflow.sklearn.save_model(
            winner,
            path=str(model_output),
            signature=infer_signature(example, winner.predict(example)),
            input_example=example,
        )
        mlflow.log_artifacts(str(model_output), artifact_path="selected_model")

        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(target.loc[test_ids], test_predictions, alpha=0.7)
        bounds = [float(target.loc[test_ids].min()), float(target.loc[test_ids].max())]
        axis.plot(bounds, bounds, "--", color="gray", label="Perfect prediction")
        axis.set(xlabel="Observed target", ylabel="Predicted target", title=f"Holdout: {winner_name}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "test_predictions.png", dpi=160)
        plt.close(figure)

        summary = {
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution": "azure" if os.getenv("AZUREML_RUN_ID") else "local",
            "parent_run_id": parent_run.info.run_id,
            "dataset": "scikit-learn diabetes (raw features)",
            "seed": seed,
            "split_rows": {name: len(ids) for name, ids in split.items()},
            "baseline_validation_mae": float(baseline_mae),
            "candidates": comparison,
            "selected_model": winner_name,
            "selection_rule": "lowest validation MAE; test excluded from selection",
            "test_metrics": test_metrics,
            "runtime": {
                "python": platform.python_version(),
                **{package: version(package) for package in ["scikit-learn", "mlflow", "numpy", "pandas"]},
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output_dir / "split_indices.json").write_text(
            json.dumps({name: ids.tolist() for name, ids in split.items()}, indent=2)
        )
        pd.DataFrame(comparison).to_csv(output_dir / "comparison.csv", index=False)
        for name in ["summary.json", "split_indices.json", "comparison.csv", "test_predictions.png"]:
            mlflow.log_artifact(str(output_dir / name), artifact_path="evaluation")
        print(json.dumps(summary, indent=2))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/local")
    parser.add_argument("--model-output", default="outputs/local/model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--tracking-uri", help="Leave unset in Azure ML jobs.")
    args = parser.parse_args()
    if args.ridge_alpha <= 0 or args.trees <= 0:
        parser.error("ridge-alpha and trees must be positive")
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    if not os.getenv("AZUREML_RUN_ID"):
        mlflow.set_experiment("learn-mlflow-regression-local")
    train(args.output_dir, args.model_output, args.seed, args.ridge_alpha, args.trees)


if __name__ == "__main__":
    main()
