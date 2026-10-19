import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from features import build_features


@pytest.fixture(scope="module")
def trained(df, params):
    X, y = build_features(df), df["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = RandomForestClassifier(**params["model"], class_weight="balanced",
                                   random_state=0).fit(Xtr, ytr)
    return model, Xte, yte


def test_model_smoke_quality(trained, params):
    """CI smoke test: the training code still produces a clearly useful model."""
    model, Xte, yte = trained
    assert roc_auc_score(yte, model.predict_proba(Xte)[:, 1]) >= params["ci"]["min_roc_auc"]


def test_prediction_ignores_transaction_id(trained, df):
    """Invariance test: changing an ID must not change the score."""
    model, _, _ = trained
    rows = df.head(50).copy()
    before = model.predict_proba(build_features(rows))
    rows["transaction_id"] = rows["transaction_id"] + 999
    assert (model.predict_proba(build_features(rows)) == before).all()
