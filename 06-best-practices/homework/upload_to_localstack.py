import boto3

# Connect to Localstack S3
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1"
)

# Create bucket
bucket_name = "nyc-duration"
s3.create_bucket(Bucket=bucket_name)

# Upload file
local_file = "sample.parquet"  # make sure this file exists
s3.upload_file(local_file, bucket_name, "in/2023-03.parquet")

# List objects to verify
response = s3.list_objects_v2(Bucket=bucket_name, Prefix="in/")
for obj in response.get("Contents", []):
    print(obj["Key"])
