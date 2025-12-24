import pickle
import pandas as pd

# Load model and DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = (
        df[categorical]
        .fillna(-1)
        .astype('int')
        .astype('str')
    )
    
    return df

# April 2023 Yellow Taxi data
year = 2023
month = 4
data_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

df = read_data(data_url)

# Prepare features and predict
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Print mean predicted duration
print("Mean predicted duration:", y_pred.mean())
