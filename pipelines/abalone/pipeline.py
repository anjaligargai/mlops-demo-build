"""
SageMaker AutoML Pipeline for Dine Brands dataset
Process -> AutoML -> Create Model -> Batch Transform -> Evaluate -> Register Model
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.2"])

import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

from sagemaker import (
    AutoML,
    AutoMLInput,
    get_execution_role,
    MetricsSource,
    ModelMetrics,
)
from sagemaker.workflow.functions import Join
from sagemaker.processing import ProcessingOutput, ProcessingInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TransformStep


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    return boto_session.client("sagemaker")


def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


# --------------------------------------------------------------------------
# Pipeline definition
# --------------------------------------------------------------------------

def get_pipeline(
    region,
    role,
    default_bucket,
    pipeline_name="DineAutoMLTrainingPipeline",
    model_package_group_name="AutoMLModelPackageGroup",
    base_job_prefix="",
    sagemaker_project_name=None,
    sagemaker_project_id=None,
    output_prefix="dine-auto-ml-training",
):

    pipeline_session = get_pipeline_session(region, default_bucket)

    if role is None:
        role = get_execution_role()

    # Parameters
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    max_automl_runtime = ParameterInteger(name="MaxAutoMLRuntime", default_value=3600)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_registration_metric_threshold = ParameterFloat(
        name="ModelRegistrationMetricThreshold", default_value=0.8
    )
    s3_bucket = ParameterString(
        name="S3Bucket", default_value=pipeline_session.default_bucket()
    )
    target_attribute_name = ParameterString(name="TargetAttributeName", default_value="customer_churn")

    # ----------------------------------------------------------------------
    # Prepare dataset (read from S3 → split → upload back to S3)
    # ----------------------------------------------------------------------
    raw_dataset_path = "s3://aishwarya-mlops-demo/dine_customer_churn/dine_data/dataset1_30k.csv"

    # Parse bucket/key
    bucket = raw_dataset_path.split("/")[2]
    key = "/".join(raw_dataset_path.split("/")[3:])

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    feature_names = [
        "date", "store_id", "store_name", "city", "state", "store_type",
        "item_id", "item_name", "category", "price", "quantity_sold",
        "revenue", "food_cost", "profit", "day_of_week", "month",
        "quarter", "is_weekend", "is_holiday", "temperature", "is_promotion",
        "stock_out", "prep_time", "calories", "is_vegetarian",
    ]
    column_names = feature_names + [target_attribute_name.default_value]
    df.columns = column_names

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save locally
    train_df.to_csv("train_val.csv", index=False)
    test_df[column_names[:-1]].to_csv("x_test.csv", header=True, index=False)
    test_df[[target_attribute_name.default_value]].to_csv("y_test.csv", header=True, index=False)

    # Upload back to S3
    train_val_s3_key = f"{output_prefix}/prepared/train_val.csv"
    s3.upload_file("train_val.csv", bucket, train_val_s3_key)
    s3_train_val = f"s3://{bucket}/{train_val_s3_key}"

    # Upload test sets into dedicated folders
    x_test_key = f"{output_prefix}/prepared/x_test/x_test.csv"
    y_test_key = f"{output_prefix}/prepared/y_test/y_test.csv"

    s3.upload_file("x_test.csv", bucket, x_test_key)
    s3.upload_file("y_test.csv", bucket, y_test_key)

    s3_x_test_prefix = f"s3://{bucket}/{output_prefix}/prepared/x_test/"
    s3_y_test_path = f"s3://{bucket}/{y_test_key}"

    # ----------------------------------------------------------------------
    # AutoML training step
    # ----------------------------------------------------------------------
    automl = AutoML(
        role=role,
        target_attribute_name=target_attribute_name.default_value,
        sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=max_automl_runtime,
        mode="ENSEMBLING",
    )
    train_args = automl.fit(
        inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_attribute_name.default_value)]
    )

    step_auto_ml_training = AutoMLStep(
        name="AutoMLTrainingStep",
        step_args=train_args,
    )

    # Best model
    best_auto_ml_model = step_auto_ml_training.get_best_auto_ml_model(
        role, sagemaker_session=pipeline_session
    )
    step_create_model = ModelStep(
        name="ModelCreationStep", step_args=best_auto_ml_model.create(instance_type=instance_type)
    )

    # ----------------------------------------------------------------------
    # Batch transform (using x_test for inference)
    # ----------------------------------------------------------------------
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=Join(on="/", values=["s3:/", s3_bucket, output_prefix, "transform"]),
        sagemaker_session=pipeline_session,
    )
    step_batch_transform = TransformStep(
        name="BatchTransformStep",
        step_args=transformer.transform(data=s3_x_test_prefix, content_type="text/csv"),
    )

    # ----------------------------------------------------------------------
    # Model evaluation
    # ----------------------------------------------------------------------
    evaluation_report = PropertyFile(
        name="evaluation", output_name="evaluation_metrics", path="evaluation_metrics.json"
    )

    sklearn_processor = SKLearnProcessor(
        role=role,
        framework_version="1.0-1",
        instance_count=instance_count,
        instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
    )
    step_args_sklearn_processor = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                destination="/opt/ml/processing/input/predictions",
            ),
            ProcessingInput(
                source=s3_y_test_path,
                destination="/opt/ml/processing/input/true_labels",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source="/opt/ml/processing/evaluation",
                destination=Join(on="/", values=["s3:/", s3_bucket, output_prefix, "evaluation"]),
            ),
        ],
        code="pipelines/abalone/evalution.py",
    )
    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=step_args_sklearn_processor,
        property_files=[evaluation_report],
    )

    # ----------------------------------------------------------------------
    # Register model
    # ----------------------------------------------------------------------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_auto_ml_training.properties.BestCandidateProperties.ModelInsightsJsonReportPath,
            content_type="application/json",
        ),
        explainability=MetricsSource(
            s3_uri=step_auto_ml_training.properties.BestCandidateProperties.ExplainabilityJsonReportPath,
            content_type="application/json",
        ),
    )

    step_register_model = ModelStep(
        name="ModelRegistrationStep",
        step_args=best_auto_ml_model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=[instance_type],
            transform_instances=[instance_type],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
        ),
    )

    # ----------------------------------------------------------------------
    # Pipeline assembly
    # ----------------------------------------------------------------------
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            instance_count,
            instance_type,
            max_automl_runtime,
            model_approval_status,
            model_registration_metric_threshold,
            s3_bucket,
            target_attribute_name,
        ],
        steps=[
            step_auto_ml_training,
            step_create_model,
            step_batch_transform,
            step_evaluation,
            step_register_model,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline
