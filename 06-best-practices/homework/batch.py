#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd

# ---------------------------------------
# Helper: get input/output paths from env
# ---------------------------------------
def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

# ---------------------------------------
# Data preparation
# ---------------------------------------
def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

# ---------------------------------------
# Read data (supports LocalStack S3)
# ---------------------------------------
def read_data(filename, categorical):
    s3_endpoint = os.getenv('S3_ENDPOINT_URL')
    storage_options = {'client_kwargs': {'endpoint_url': s3_endpoint}} if s3_endpoint else None
    df = pd.read_parquet(filename, storage_options=storage_options)
    df = prepare_data(df, categorical)
    return df

# ---------------------------------------
# Save data (to local or S3)
# ---------------------------------------
def save_data(df, output_file):
    s3_endpoint = os.getenv('S3_ENDPOINT_URL')
    storage_options = {'client_kwargs': {'endpoint_url': s3_endpoint}} if s3_endpoint else None

    # Ensure local folder exists if writing locally
    if not output_file.startswith("s3://"):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df.to_parquet(output_file, engine='pyarrow', index=False, storage_options=storage_options if output_file.startswith("s3://") else None)

# ---------------------------------------
# Main function
# ---------------------------------------
def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']

    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    save_data(df_result, output_file)

# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
