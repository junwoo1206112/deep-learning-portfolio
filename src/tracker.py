import mlflow
from pathlib import Path


def setup_tracking(config: dict):
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])


def start_run(run_name: str | None = None):
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics: dict, step: int | None = None):
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value), step=step)


def log_model_artifact(model_path: str):
    path = Path(model_path)
    if path.is_file():
        mlflow.log_artifact(str(path))
    elif path.is_dir():
        mlflow.log_artifacts(str(path))


def end_run():
    mlflow.end_run()
