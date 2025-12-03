#!/usr/bin/env python3
import os
import pickle
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

"""
Updated register_model.py

- Robust checks for missing experiments / runs
- Handles missing dv.pkl gracefully (skips run)
- Uses backward-compatible RMSE computation (no `squared` kwarg)
- Logs evaluation results to a new experiment "random-forest-best-models"
- Registers the best model (runs:/<RUN_ID>/model) as "best-random-forest"
"""

TEST_EXPERIMENT_NAME = "random-forest-hyperopt"
RESULTS_EXPERIMENT_NAME = "random-forest-best-models"
REGISTERED_MODEL_NAME = "best-random-forest"
TOP_N = 5


def read_test_data(file_path: str) -> pd.DataFrame:
    df = pd.read_parquet(file_path)

    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

    df['duration'] = (
        df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    ).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)

    return df


def evaluate_and_register(data_path: str):
    client = MlflowClient()

    # locate the hyperopt experiment
    experiment = mlflow.get_experiment_by_name(TEST_EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{TEST_EXPERIMENT_NAME}' not found. Exiting.")
        return

    # search top N runs by recorded metric 'rmse' (ascending)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=TOP_N,
    )

    if not runs:
        print("No runs found in experiment. Exiting.")
        return

    # Prepare results experiment (to store evaluation metrics)
    results_experiment = mlflow.get_experiment_by_name(RESULTS_EXPERIMENT_NAME)
    if results_experiment is None:
        results_experiment_id = mlflow.create_experiment(RESULTS_EXPERIMENT_NAME)
    else:
        results_experiment_id = results_experiment.experiment_id

    test_file = os.path.join(data_path, "green_tripdata_2023-03.parquet")
    print("\nEvaluating top %d models on March-2023 test set...\n" % TOP_N)

    test_data = read_test_data(test_file)
    X_test = test_data[['PULocationID', 'DOLocationID']]
    y_test = test_data['duration']

    best_rmse = float("inf")
    best_run_id = None
    best_model_uri = None

    # Start a run in the results experiment to log per-run RMSEs
    with mlflow.start_run(experiment_id=results_experiment_id, run_name="eval-top-models"):
        for run in runs:
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"
            dv_artifact_path = f"runs:/{run_id}/preprocessor/dv.pkl"

            # Try to download dv.pkl (vectorizer)
            try:
                dv_local_path = mlflow.artifacts.download_artifacts(dv_artifact_path)
                # download_artifacts may return a file path or a directory; handle both
                if os.path.isdir(dv_local_path):
                    # find dv.pkl inside
                    candidate = os.path.join(dv_local_path, "dv.pkl")
                    if os.path.exists(candidate):
                        dv_local_path = candidate
                with open(dv_local_path, "rb") as f:
                    dv = pickle.load(f)
            except Exception as e:
                print(f"❌ Skipping run {run_id}: Cannot load dv.pkl ({e})")
                # Log that this run was skipped
                mlflow.log_metric(f"skipped_{run_id}", 1)
                continue

            # Try to load model
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                print(f"❌ Skipping run {run_id}: Cannot load model ({e})")
                mlflow.log_metric(f"skipped_model_{run_id}", 1)
                continue

            # Transform and predict
            try:
                X_test_transformed = dv.transform(X_test.to_dict(orient="records"))
                preds = model.predict(X_test_transformed)
            except Exception as e:
                print(f"❌ Skipping run {run_id}: Error during prediction ({e})")
                mlflow.log_metric(f"skipped_pred_{run_id}", 1)
                continue

            # Compute RMSE (backward compatible)
            rmse = mean_squared_error(y_test, preds) ** 0.5
            print(f"Run {run_id}: Test RMSE = {rmse:.3f}")

            # Log metric for this evaluated run in the results experiment
            mlflow.log_metric(f"rmse_{run_id}", rmse)

            # Update best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_run_id = run_id
                best_model_uri = model_uri

        # End of evaluation loop

    if best_model_uri:
        print(f"\nBest test RMSE: {best_rmse:.3f} (run {best_run_id})")
        # Register the model in the model registry
        try:
            mlflow.register_model(best_model_uri, REGISTERED_MODEL_NAME)
            print(f"Model registered as '{REGISTERED_MODEL_NAME}' (from run {best_run_id})")
        except Exception as e:
            print(f"Failed to register model ({e})")
    else:
        print("No valid model found to register.")


def main(data_path: str):
    evaluate_and_register(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate top models and register the best one.")
    parser.add_argument("--data_path", required=True, help="Path to directory with test parquet file")
    args = parser.parse_args()
    main(args.data_path)
