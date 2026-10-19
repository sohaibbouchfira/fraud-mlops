"""Train the model, save it, and log the run to MLflow with lineage tags."""
import hashlib, json, os, pickle, subprocess
import mlflow, mlflow.sklearn, pandas as pd, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features import build_features, TARGET

params = yaml.safe_load(open("params.yaml"))
df = pd.read_csv("data/transactions.csv")
X, y = build_features(df), df[TARGET]
X_train, _, y_train, _ = train_test_split(X, y, test_size=params["test_size"],
                                          stratify=y, random_state=params["seed"])
model = RandomForestClassifier(**params["model"], class_weight="balanced",
                               random_state=params["seed"], n_jobs=-1)
model.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)
pickle.dump(model, open("models/model.pkl", "wb"))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


mlflow.set_tracking_uri("sqlite:///mlflow.db")      # a database backend is required by the registry
mlflow.set_experiment("fraud-detection")
with mlflow.start_run() as run:
    mlflow.log_params({**params["model"], "seed": params["seed"]})
    mlflow.set_tags({"git_commit": git_commit(),
                     "data_md5": hashlib.md5(open("data/transactions.csv", "rb").read()).hexdigest()})
    # Recent MLflow saves sklearn models with skops (safe format) and asks us to trust tree types explicitly
    info = mlflow.sklearn.log_model(model, name="model",
                                    skops_trusted_types=["sklearn.tree._tree.Tree"])
json.dump({"run_id": run.info.run_id, "model_uri": info.model_uri}, open("models/run_info.json", "w"))
print("run", run.info.run_id)
