import json
import os
import pathlib
import pandas as pd
from sklearn.metrics import f1_score
import mlflow

if __name__ == "__main__":
    # -------------------------
    # Load predictions & labels
    # -------------------------
    y_pred_path = "/opt/ml/processing/input/predictions/x_test.csv.out"
    y_pred = pd.read_csv(y_pred_path, header=None)

    y_true_path = "/opt/ml/processing/input/true_labels/y_test.csv"
    y_true = pd.read_csv(y_true_path, header=None)

    # -------------------------
    # Compute metrics
    # -------------------------
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report_dict = {
        "classification_metrics": {
            "weighted_f1": {
                "value": weighted_f1,
                "standard_deviation": "NaN",
            },
        },
    }

    # -------------------------
    # Save metrics for SageMaker pipeline
    # -------------------------
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation_metrics.json")

    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    # -------------------------
    # Log metrics to MLflow
    # -------------------------
    # Tracking URI can be S3, local FS, or MLflow server
    mlflow.set_tracking_uri("s3://aishwarya-mlops-demo/mlflow-tracking")

    # Use environment variable to differentiate runs (optional)
    run_name = os.getenv("MLFLOW_RUN_NAME", "AutoML_Evaluation")
    dataset_version = os.getenv("DATASET_VERSION", "dataset1")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("evaluation_script", "evaluation.py")
        mlflow.log_metric("weighted_f1", weighted_f1)

        # Save the full evaluation JSON as an artifact
        mlflow.log_artifact(evaluation_path, artifact_path="evaluation")
