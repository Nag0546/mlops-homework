import os
import math
import argparse
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def load_data(data_path):
    train_df = pd.read_parquet(os.path.join(data_path, "green_tripdata_2023-01.parquet"))
    valid_df = pd.read_parquet(os.path.join(data_path, "green_tripdata_2023-02.parquet"))

    features = ["PULocationID", "DOLocationID"]
    dv = DictVectorizer()

    train_dicts = train_df[features].astype(str).to_dict(orient="records")
    valid_dicts = valid_df[features].astype(str).to_dict(orient="records")

    X_train = dv.fit_transform(train_dicts)
    y_train = train_df["duration"].values
    X_valid = dv.transform(valid_dicts)
    y_valid = valid_df["duration"].values

    return X_train, y_train, X_valid, y_valid, dv


# Hyperopt search space
search_space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
    "max_depth": hp.choice("max_depth", [10, 20, 30, None]),
    "min_samples_split": hp.choice("min_samples_split", [2, 4, 8, 10]),
}


def objective(params):
    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=None if params["max_depth"] is None else int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

        # Compute RMSE
        mse = mean_squared_error(y_valid, preds)
        rmse = math.sqrt(mse)
        mlflow.log_metric("rmse", rmse)

        # Log the model and vectorizer for later use
        mlflow.sklearn.log_model(model, artifact_path="model")

        with open("dv.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dv.pkl", artifact_path="preprocessor")

        return {"loss": rmse, "status": STATUS_OK}


def run_hpo(data_path, max_evals=20):
    global X_train, y_train, X_valid, y_valid, dv
    X_train, y_train, X_valid, y_valid, dv = load_data(data_path)

    mlflow.set_experiment("random-forest-hyperopt")
    trials = Trials()

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    print("Best parameters (hyperopt encoding):", best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to preprocessed parquet files (./output)")
    parser.add_argument("--max_evals", default=20, type=int, help="Number of HPO evaluations")
    args = parser.parse_args()

    run_hpo(args.data_path, args.max_evals)
