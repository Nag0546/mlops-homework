# create_sample_parquet.py
import pandas as pd
from datetime import datetime

data = {
    "PULocationID": [1, 2],
    "DOLocationID": [3, 4],
    "tpep_pickup_datetime": [datetime(2023,3,1,10,0), datetime(2023,3,1,11,0)],
    "tpep_dropoff_datetime": [datetime(2023,3,1,10,30), datetime(2023,3,1,11,30)]
}

df = pd.DataFrame(data)
df.to_parquet("sample.parquet", engine="pyarrow", index=False)
print("sample.parquet created!")
