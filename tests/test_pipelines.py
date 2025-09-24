import boto3
import json
import sys
import time
import os

sm_client = boto3.client("sagemaker")
runtime_client = boto3.client("sagemaker-runtime")

# Environment variables from CodePipeline/CloudFormation
MODEL_PACKAGE_GROUP_NAME = os.environ.get("MODEL_PACKAGE_GROUP_NAME", "anjali-mlops-demo")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "anjali-mlops-demo-staging")


def test_model_registration():
    """Check if latest model is registered in Model Package Group"""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1
    )
    assert len(response["ModelPackageSummaryList"]) > 0, "No models found in package group!"
    latest_model = response["ModelPackageSummaryList"][0]
    print(f"✅ Model registered: {latest_model['ModelPackageArn']}")
    return latest_model["ModelPackageArn"]


def test_deployment():
    """Check if endpoint exists and is InService"""
    response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = response["EndpointStatus"]
    assert status == "InService", f"Endpoint {ENDPOINT_NAME} not ready. Current status: {status}"
    print(f"✅ Endpoint {ENDPOINT_NAME} is deployed and InService.")
    return True


def test_inference():
    """Invoke the endpoint with a realistic store transaction payload"""
    payload = {
        "date": "2020-01-01",
        "store_id": "STORE_001",
        "store_name": "Dine Location 1",
        "city": "East Tammymouth",
        "state": "NV",
        "store_type": "Mall",
        "item_id": "ITEM_064",
        "item_name": "Sandwich",
        "category": "Appetizers",
        "price": 24.92,
        "quantity_sold": 58,
        "revenue": 1445.36,
        "food_cost": 294.64,
        "profit": 1150.72,
        "day_of_week": "Wednesday",
        "month": 1,
        "quarter": "Q1",
        "is_weekend": False,
        "is_holiday": True,
        "temperature": 80.3,
        "is_promotion": False,
        "stock_out": False,
        "prep_time": 16,
        "calories": 919,
        "is_vegetarian": False,
        "customer churn": 1
    }

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = response["Body"].read().decode("utf-8")
    print(f"✅ Inference response: {result}")
    return result



if __name__ == "__main__":
    try:
        print("Running Model Registration Test...")
        test_model_registration()

        print("\nRunning Deployment Test...")
        test_deployment()

        print("\nRunning Inference Test...")
        test_inference()

        print("\n🎉 All tests passed successfully!")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
