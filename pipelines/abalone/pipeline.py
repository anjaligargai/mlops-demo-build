"""
Example workflow pipeline script for AutoML training.

Process -> AutoML -> Create Model -> Batch Transform -> Evaluate -> Register Model
"""

import os
import json
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

import sagemaker
from sagemaker import (
    AutoML,
    AutoMLInput,
    get_execution_role,
    MetricsSource,
    ModelMetrics,
)
from sagemaker.predictor import Predictor
from sagemaker.processing import ProcessingOutput, ProcessingInput, ScriptProcessor
from sagemaker.s3 import s3_path_join, S3Downloader, S3Uploader
from sagemaker.serializers import CSVSerializer
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join, JsonGet
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
    """Gets the sagemaker client."""
    boto_session = boto3.Session(region_name=region)
    return boto_session.client("sagemaker")


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region."""
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
    role=None,
    default_bucket=None,
    pipeline_name="AutoMLTrainingPipeline",
    model_package_group_name="AutoMLModelPackageGroup",
    output_prefix="auto-ml-training",
):
    """Builds a SageMaker AutoML pipeline."""

    pipeline_session = get_pipeline_session(region, default_bucket)

    if role is None:
        role = get_execution_role()

    # Parameters
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    max_automl_runtime = ParameterInteger(name="MaxAutoMLRuntime", default_value=3600)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_registration_metric_threshold = ParameterFloat(
        name="ModelRegistrationMetricThreshold", default_value=0.2
    )
    s3_bucket = ParameterString(
        name="S3Bucket", default_value=pipeline_session.default_bucket()
    )
    target_attribute_name = ParameterString(name="TargetAttributeName", default_value="class")

    # ----------------------------------------------------------------------
    # Prepare data
    # ----------------------------------------------------------------------

    feature_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country",
    ]
    column_names = feature_names + [target_attribute_name.default_value]

    # Training dataset
    dataset_file_name = "adult.data"
    S3Downloader.download(
        f"s3://sagemaker-example-files-prod-{boto3.session.Session().region_name}/datasets/tabular/uci_adult/{dataset_file_name}",
        ".",
        sagemaker_session=pipeline_session,
    )
    df = pd.read_csv(dataset_file_name, header=None, names=column_names)
    df.to_csv("train_val.csv", index=False)

    # Test dataset
    dataset_file_name = "adult.test"
    S3Downloader.download(
        f"s3://sagemaker-example-files-prod-{boto3.session.Session().region_name}/datasets/tabular/uci_adult/{dataset_file_name}",
        ".",
        sagemaker_session=pipeline_session,
    )
    df = pd.read_csv(dataset_file_name, header=None, names=column_names, skiprows=1)
    df[target_attribute_name.default_value] = df[target_attribute_name.default_value].map(
        {" <=50K.": " <=50K", " >50K.": " >50K"}
    )
    df.to_csv("x_test.csv", header=False, index=False, columns=feature_names)
    df.to_csv("y_test.csv", header=False, index=False, columns=[target_attribute_name.default_value])

    # Upload to S3
    s3_prefix = s3_path_join("s3://", s3_bucket.default_value, "data")
    S3Uploader.upload("train_val.csv", s3_prefix, sagemaker_session=pipeline_session)
    S3Uploader.upload("x_test.csv", s3_prefix, sagemaker_session=pipeline_session)
    S3Uploader.upload("y_test.csv", s3_prefix, sagemaker_session=pipeline_session)

    s3_train_val = s3_path_join(s3_prefix, "train_val.csv")
    s3_x_test = s3_path_join(s3_prefix, "x_test.csv")
    s3_y_test = s3_path_join(s3_prefix, "y_test.csv")

    # ----------------------------------------------------------------------
    # AutoML training step
    # ----------------------------------------------------------------------

    automl = AutoML(
        role=role,
        target_attribute_name=target_attribute_name,
        sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=max_automl_runtime,
        mode="ENSEMBLING",
    )
    train_args = automl.fit(
        inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_attribute_name)]
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
    # Batch transform
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
        step_args=transformer.transform(data=s3_x_test, content_type="text/csv"),
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
            ProcessingInput(source=s3_y_test, destination="/opt/ml/processing/input/true_labels"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation_metrics",
                source="/opt/ml/processing/evaluation",
                destination=Join(on="/", values=["s3:/", s3_bucket, output_prefix, "evaluation"]),
            ),
        ],
        code="evaluation.py",
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
            model_package_group_name,
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
