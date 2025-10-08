import boto3
import pandas as pd
import time
import os

def fetch_data_from_athena(database, table, output_bucket, region="us-east-1", limit=10):
    """Fetch data from Athena and return a pandas DataFrame."""
    athena = boto3.client("athena", region_name=region)

    # SQL query
    query = f"SELECT * FROM {table} LIMIT {limit}"

    # Start Athena query
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_bucket},
    )

    query_execution_id = response["QueryExecutionId"]

    # Wait for completion
    while True:
        status = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(2)

    if state != "SUCCEEDED":
        raise Exception(f"Athena query failed with state: {state}")

    # Construct result path
    result_path = f"{output_bucket}{query_execution_id}.csv"
    print("Query completed. Download the result from:", result_path)

    # Load CSV to pandas
    df = pd.read_csv(result_path)
    return df


def preprocess_and_upload(bucket, prefix, region="us-east-1"):
    """Preprocess data: fetch from Athena, clean, and upload to S3."""
    database = "dine_demo_mlops"  # Replace with your Glue DB
    table = "dine_data2"
    output_bucket = "s3://aishwarya-mlops-demo/dine_customer_churn/temp/"

    # Fetch Athena data
    df = fetch_data_from_athena(database, table, output_bucket, region)

    # Example preprocessing (customize as needed)
    df = df.dropna()  # remove nulls
    df.to_csv("preprocessed.csv", index=False)

    # Upload to S3
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file("preprocessed.csv", bucket, f"{prefix}/preprocessed.csv")

    print(f"Preprocessed data uploaded to s3://{bucket}/{prefix}/preprocessed.csv")
    return f"s3://{bucket}/{prefix}/preprocessed.csv"
