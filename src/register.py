"""Register the model only if the gates passed; mark it as the challenger."""
import json, sys
import mlflow
from mlflow import MlflowClient

NAME = "fraud-detector"
metrics = json.load(open("metrics.json"))
if not metrics["gates_passed"]:
    sys.exit("Gates failed: model NOT registered.")
info = json.load(open("models/run_info.json"))
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mv = mlflow.register_model(info["model_uri"], NAME)
client = MlflowClient()
client.set_model_version_tag(NAME, mv.version, "validation_status", "passed")
client.set_registered_model_alias(NAME, "challenger", mv.version)
print(f"Registered {NAME} v{mv.version} as @challenger")
