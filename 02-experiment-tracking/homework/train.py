import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def load_data(output_folder):
    df_train = pd.read_parquet(os.path.join(output_folder, "green_tripdata_2023-01.parquet"))
    df_valid = pd.read_parquet(os.path.join(output_folder, "green_tripdata_2023-02.parquet"))
    return df_train, df_valid


def prepare_features(df):
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    dicts = df[["PULocationID", "DOLocationID"]].to_dict(orient="records")
    return dicts, df["duration"].values


def run_train(data_path):
    df_train, df_valid = load_data(data_path)

    train_dicts, y_train = prepare_features(df_train)
    valid_dicts, y_valid = prepare_features(df_valid)

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_valid = dv.transform(valid_dicts)

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        rf = RandomForestRegressor(
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

        print(f"RMSE: {rmse:.3f}")
        mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./output")
    args = parser.parse_args()

    run_train(args.data_path)
