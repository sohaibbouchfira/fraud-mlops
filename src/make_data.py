"""Generate a synthetic, imbalanced card-transaction dataset (no download needed)."""
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate(n_samples=20000, seed=42):
    X, y = make_classification(n_samples=n_samples, n_features=8, n_informative=5,
                               weights=[0.97], flip_y=0.005, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["amount"] = (np.abs(df["f0"]) * 120).round(2)
    df["transaction_id"] = np.arange(len(df)) + 1_000_000   # an ID: must never be a feature
    df["is_fraud"] = y
    return df


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generate(seed=seed).to_csv("data/transactions.csv", index=False)
    print(f"data/transactions.csv written (seed={seed})")
