import pandas as pd
from datetime import datetime as dt
import os

data = [
    (1, 1, dt(2023, 1, 1, 1, 1), dt(2023, 1, 1, 1, 10)),
    (1, None, dt(2023, 1, 1, 2, 2, 0), dt(2023, 1, 1, 2, 2, 59)),
    (3, 4, dt(2023, 1, 1, 2, 2, 0), dt(2023, 1, 2, 2, 1)),
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

options = {'client_kwargs': {'endpoint_url': os.getenv('S3_ENDPOINT_URL')}}
input_file = os.getenv('INPUT_FILE_PATTERN').format(year=2023, month=1)

df_input.to_parquet(input_file, engine='pyarrow', index=False, storage_options=options)
