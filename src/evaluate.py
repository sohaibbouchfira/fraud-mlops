"""Evaluate on the held-out split, apply the gates, write metrics.json."""
import json, pickle
import mlflow, pandas as pd, yaml
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from features import build_features, TARGET

params = yaml.safe_load(open("params.yaml"))
df = pd.read_csv("data/transactions.csv")
X, y = build_features(df), df[TARGET]
_, X_test, _, y_test = train_test_split(X, y, test_size=params["test_size"],
                                        stratify=y, random_state=params["seed"])
model = pickle.load(open("models/model.pkl", "rb"))
pred, proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
m = {"recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
     "roc_auc": roc_auc_score(y_test, proba)}
g = params["gates"]
m["gates_passed"] = bool(m["recall"] >= g["min_recall"] and m["precision"] >= g["min_precision"])
json.dump(m, open("metrics.json", "w"), indent=2)

run_id = json.load(open("models/run_info.json"))["run_id"]
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics({k: v for k, v in m.items() if k != "gates_passed"})
print(m)
