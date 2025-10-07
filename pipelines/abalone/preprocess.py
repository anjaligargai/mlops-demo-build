"""
Data Preprocessing for Dine Brands Customer Churn Dataset
- Load dataset from S3
- Add feature/target columns
- Train-test split
- Upload processed train/test data back to S3
"""

import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split


def preprocess_and_upload(raw_dataset_s3: str, output_prefix: str):
    bucket = raw_dataset_s3.split("/")[2]
    key = "/".join(raw_dataset_s3.split("/")[3:])

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    feature_names = [
        "date","store_id","store_name","city","state","store_type",
        "item_id","item_name","category","price","quantity_sold",
        "revenue","food_cost","profit","day_of_week","month",
        "quarter","is_weekend","is_holiday","temperature","is_promotion",
        "stock_out","prep_time","calories","is_vegetarian",
    ]
    target_col = "customer churn"
    df.columns = feature_names + [target_col]

    # Split dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("train_val.csv", index=False)
    test_df[feature_names].to_csv("x_test.csv", index=False, header=False)
    test_df[[target_col]].to_csv("y_test.csv", index=False, header=False)

    # Upload processed data back to S3
    prepared_prefix = f"{output_prefix}/prepared"
    train_val_s3_key = f"{prepared_prefix}/train_val.csv"
    x_test_s3_key = f"{prepared_prefix}/x_test/x_test.csv"
    y_test_s3_key = f"{prepared_prefix}/y_test/y_test.csv"

    s3.upload_file("train_val.csv", bucket, train_val_s3_key)
    s3.upload_file("x_test.csv", bucket, x_test_s3_key)
    s3.upload_file("y_test.csv", bucket, y_test_s3_key)

    return {
        "s3_train_val": f"s3://{bucket}/{train_val_s3_key}",
        "s3_x_test_prefix": f"s3://{bucket}/{prepared_prefix}/x_test/",
        "s3_y_test": f"s3://{bucket}/{y_test_s3_key}"
    }
