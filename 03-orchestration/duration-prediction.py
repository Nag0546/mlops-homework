#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

# Create models folder
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

# ------------------------------
# Data reading and preparation
# ------------------------------
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    # Compute trip duration in minutes
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60

    # Filter durations between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Number of records loaded and filtered: {len(df)}")
    return df

# ------------------------------
# Feature transformation
# ------------------------------
def create_X(df, dv=None, features=None):
    if features is None:
        features = ['PULocationID', 'DOLocationID', 'trip_distance']
    dicts = df[features].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

# ------------------------------
# XGBoost model training
# ------------------------------
def train_xgboost(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',  # updated deprecated 'reg:linear'
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        mlflow.log_metric("rmse", rmse)

        # Save preprocessor
        with open("models/preprocessor_xgb.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor_xgb.b", artifact_path="preprocessor")

        # Log XGBoost model
        mlflow.xgboost.log_model(booster, artifact_path="xgboost_model")

        return run.info.run_id

# ------------------------------
# Linear regression training
# ------------------------------
def train_linear_regression(df_train, df_val):
    features = ['PULocationID', 'DOLocationID']
    X_train, dv = create_X(df_train, features=features)
    X_val, _ = create_X(df_val, dv=dv, features=features)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"Linear Regression Intercept: {lr.intercept_:.2f}")

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5

    with mlflow.start_run() as run:
        mlflow.log_metric("rmse", rmse)

        # Save vectorizer and model
        with open("models/preprocessor_lr.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor_lr.b", artifact_path="preprocessor")

        with open("models/linear_regression_model.b", "wb") as f_out:
            pickle.dump(lr, f_out)
        mlflow.log_artifact("models/linear_regression_model.b", artifact_path="model")

        return run.info.run_id

# ------------------------------
# Run pipeline
# ------------------------------
def run(year, month):
    df_train = read_dataframe(year, month)
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(next_year, next_month)

    # XGBoost
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id_xgb = train_xgboost(X_train, y_train, X_val, y_val, dv)
    print(f"XGBoost run_id: {run_id_xgb}")

    # Linear Regression
    run_id_lr = train_linear_regression(df_train, df_val)
    print(f"Linear Regression run_id: {run_id_lr}")

    return run_id_xgb, run_id_lr

# ------------------------------
# Main entry
# ------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train models to predict NYC taxi trip duration.')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    run(year=args.year, month=args.month)
