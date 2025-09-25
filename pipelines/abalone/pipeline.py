"""
SageMaker AutoML Pipeline for Dine Brands dataset
Process -> AutoML -> Create Model -> Batch Transform -> Evaluate -> Condition
Condition: if F1 >= threshold → Register
Else → Retry AutoML with different config
Else → New data retrain
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "--quiet"])

import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split

from sagemaker import AutoML, AutoMLInput, get_execution_role
from sagemaker import MetricsSource, ModelMetrics
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
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import JsonGet

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session, sagemaker_client=sagemaker_client, default_bucket=default_bucket
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
    output_prefix="dine-auto-ml-training",
    base_job_prefix="dine_demo",
    sagemaker_project_name="dine_demo"
):

    pipeline_session = get_pipeline_session(region, default_bucket)
    if role is None:
        role = get_execution_role()

    # -------------------------
    # Parameters
    # -------------------------
    instance_count = ParameterInteger(name="InstanceCount", default_value=1)
    instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
    max_automl_runtime = ParameterInteger(name="MaxAutoMLRuntime", default_value=3600)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="Approved")
    model_registration_metric_threshold = ParameterFloat(name="ModelRegistrationMetricThreshold", default_value=0.8)
    s3_bucket_param = ParameterString(name="S3Bucket", default_value=pipeline_session.default_bucket())
    target_attribute_name = ParameterString(name="TargetAttributeName", default_value="customer churn")

    # -------------------------
    # Dataset 1 (base dataset)
    # -------------------------
    raw_dataset_s3 = "s3://aishwarya-mlops-demo/dine_customer_churn/dine_data/dataset1_30k.csv"
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
    target_col = target_attribute_name.default_value
    df.columns = feature_names + [target_col]

    # split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("train_val.csv", index=False)
    test_df[feature_names].to_csv("x_test.csv", index=False, header=False)
    test_df[[target_col]].to_csv("y_test.csv", index=False, header=False)

    # upload
    prepared_prefix = f"{output_prefix}/prepared"
    train_val_s3_key = f"{prepared_prefix}/train_val.csv"
    x_test_s3_key = f"{prepared_prefix}/x_test/x_test.csv"
    y_test_s3_key = f"{prepared_prefix}/y_test/y_test.csv"

    s3.upload_file("train_val.csv", bucket, train_val_s3_key)
    s3.upload_file("x_test.csv", bucket, x_test_s3_key)
    s3.upload_file("y_test.csv", bucket, y_test_s3_key)

    s3_train_val = f"s3://{bucket}/{train_val_s3_key}"
    s3_x_test_prefix = f"s3://{bucket}/{prepared_prefix}/x_test/"
    s3_y_test = f"s3://{bucket}/{y_test_s3_key}"

    # -------------------------
    # Step 1: AutoML training
    # -------------------------
    automl = AutoML(
        role=role,
        target_attribute_name=target_col,
        sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=max_automl_runtime,
        mode="ENSEMBLING",
    )
    step_auto_ml_training = AutoMLStep(
        name="AutoMLTrainingStep",
        step_args=automl.fit(inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_col)])
    )

    # create model
    best_model = step_auto_ml_training.get_best_auto_ml_model(role, sagemaker_session=pipeline_session)
    step_create_model = ModelStep(name="ModelCreationStep", step_args=best_model.create(instance_type=instance_type))

    # batch transform
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "transform"]),
        sagemaker_session=pipeline_session,
    )
    step_batch_transform = TransformStep(
        name="BatchTransformStep", step_args=transformer.transform(data=s3_x_test_prefix, content_type="text/csv")
    )

    # evaluation
    evaluation_report = PropertyFile(name="evaluation", output_name="evaluation_metrics", path="evaluation_metrics.json")
    sklearn_processor = SKLearnProcessor(
        role=role, framework_version="1.0-1",
        instance_count=instance_count, instance_type=instance_type.default_value,
        sagemaker_session=pipeline_session,
    )
    step_evaluation = ProcessingStep(
        name="ModelEvaluationStep",
        step_args=sklearn_processor.run(
            inputs=[
                ProcessingInput(source=step_batch_transform.properties.TransformOutput.S3OutputPath,
                                destination="/opt/ml/processing/input/predictions"),
                ProcessingInput(source=s3_y_test, destination="/opt/ml/processing/input/true_labels"),
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation_metrics", source="/opt/ml/processing/evaluation",
                                 destination=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "evaluation"]))
            ],
            code="pipelines/abalone/evalution.py",
        ),
        property_files=[evaluation_report],
    )

    # -------------------------
    # Condition 1 → Retry if F1 < threshold
    # -------------------------
    cond_f1_first = ConditionGreaterThanOrEqualTo(
        left = JsonGet(
        step=step_evaluation,  # directly pass the ProcessingStep object
        property_file=evaluation_report,
        json_path="classification_metrics.weighted_f1.value"
    )
    )

    # Retry AutoML with different params (Option 1)
    automl_retry = AutoML(
        role=role,
        target_attribute_name=target_col,
        sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=7200,
        mode="HYPERPARAMETER_TUNING",
    )
    step_automl_retry = AutoMLStep(
        name="AutoMLRetryStep",
        step_args=automl_retry.fit(inputs=[AutoMLInput(inputs=s3_train_val, target_attribute_name=target_col)])
    )

    step_cond_first = ConditionStep(
        name="CheckF1ScoreFirst",
        conditions=[cond_f1_first],
        if_steps=[],        # register model later
        else_steps=[step_automl_retry],
    )

    # -------------------------
    # Retry evaluation (Option 1 flow)
    # -------------------------
    retry_model = step_automl_retry.get_best_auto_ml_model(role, sagemaker_session=pipeline_session)
    step_create_model_retry = ModelStep(name="ModelCreationStepRetry", step_args=retry_model.create(instance_type=instance_type))
    transformer_retry = Transformer(
        model_name=step_create_model_retry.properties.ModelName,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "transform_retry"]),
        sagemaker_session=pipeline_session,
    )
    step_batch_transform_retry = TransformStep(
        name="BatchTransformStepRetry",
        step_args=transformer_retry.transform(data=s3_x_test_prefix, content_type="text/csv"),
    )
    step_eval_retry = ProcessingStep(
        name="ModelEvaluationStepRetry",
        step_args=sklearn_processor.run(
            inputs=[
                ProcessingInput(source=step_batch_transform_retry.properties.TransformOutput.S3OutputPath,
                                destination="/opt/ml/processing/input/predictions"),
                ProcessingInput(source=s3_y_test, destination="/opt/ml/processing/input/true_labels"),
            ],
            outputs=[ProcessingOutput(output_name="evaluation_metrics", source="/opt/ml/processing/evaluation",
                                      destination=Join(on="/", values=["s3:/", s3_bucket_param, output_prefix, "evaluation_retry"]))],
            code="pipelines/abalone/evalution.py",
        ),
        property_files=[evaluation_report],
    )

    cond_f1_retry = ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name="ModelEvaluationStepRetry", property_file=evaluation_report,
                     json_path="classification_metrics.weighted_f1.value"),
        right=model_registration_metric_threshold,
    )

    # Option 2: new dataset
    new_data_s3 = "s3://aishwarya-mlops-demo/dine_customer_churn/dine_data2/dataset2_30k.csv"
    automl_new_data = AutoML(
        role=role, target_attribute_name=target_col, sagemaker_session=pipeline_session,
        total_job_runtime_in_seconds=7200, mode="ENSEMBLING"
    )
    step_automl_new_data = AutoMLStep(
        name="AutoMLNewDataStep",
        step_args=automl_new_data.fit(inputs=[AutoMLInput(inputs=new_data_s3, target_attribute_name=target_col)])
    )

    step_cond_retry = ConditionStep(
        name="CheckF1ScoreRetry",
        conditions=[cond_f1_retry],
        if_steps=[],        # register model later
        else_steps=[step_automl_new_data],
    )

    # -------------------------
    # Register Model (common)
    # -------------------------
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
        step_args=best_model.register(
            content_types=["text/csv"], response_types=["text/csv"],
            inference_instances=[instance_type], transform_instances=[instance_type],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status, model_metrics=model_metrics,
        ),
    )

    # -------------------------
    # Assemble pipeline
    # -------------------------
    steps = [
        step_auto_ml_training, step_create_model, step_batch_transform, step_evaluation,
        step_cond_first, step_create_model_retry, step_batch_transform_retry, step_eval_retry, step_cond_retry,
        step_register_model
    ]

    return Pipeline(
        name=pipeline_name,
        parameters=[instance_count, instance_type, max_automl_runtime,
                    model_approval_status, model_registration_metric_threshold,
                    s3_bucket_param, target_attribute_name],
        steps=steps,
        sagemaker_session=pipeline_session,
    )
