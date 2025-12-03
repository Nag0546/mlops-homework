import os
import argparse
import pandas as pd


def read_data(filename):
    df = pd.read_parquet(filename)

    # Green taxi uses these datetime columns
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

    # Duration in minutes
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter out trips <1 minute or >60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns
    df['PULocationID'] = df['PULocationID'].astype(str)
    df['DOLocationID'] = df['DOLocationID'].astype(str)

    return df


def main(raw_data_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)

    # Use green taxi files
    files = [
        'green_tripdata_2023-01.parquet',
        'green_tripdata_2023-02.parquet',
        'green_tripdata_2023-03.parquet'
    ]

    for file in files:
        input_file = os.path.join(raw_data_path, file)
        print(f"Reading: {input_file}")

        df = read_data(input_file)

        output_file = os.path.join(dest_path, file)
        df.to_parquet(output_file, index=False)

        print(f"Processed: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", required=True)
    parser.add_argument("--dest_path", required=True)
    args = parser.parse_args()

    main(args.raw_data_path, args.dest_path)
