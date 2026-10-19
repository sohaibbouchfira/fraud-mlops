"""Feature logic shared by training and serving (one module = no training-serving skew)."""
import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(8)] + ["log_amount"]
TARGET = "is_fraud"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    return out[FEATURES]          # selects features only: IDs and target are dropped
