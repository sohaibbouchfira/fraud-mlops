"""Promote or roll back: move the @champion alias to a given version.
Usage: python src/promote.py 3"""
import sys
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MlflowClient().set_registered_model_alias("fraud-detector", "champion", sys.argv[1])
model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")
print(f"@champion -> v{sys.argv[1]} (loaded OK)")
