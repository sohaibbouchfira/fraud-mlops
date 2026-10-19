from features import build_features, FEATURES


def test_feature_columns(df):
    assert list(build_features(df).columns) == FEATURES


def test_no_id_or_target_leak(df):
    X = build_features(df)
    assert "transaction_id" not in X.columns and "is_fraud" not in X.columns


def test_features_have_no_nan(df):
    assert not build_features(df).isna().any().any()
