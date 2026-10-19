def test_schema(df):
    assert {"amount", "transaction_id", "is_fraud"} <= set(df.columns)


def test_no_missing_values(df):
    assert df.isna().mean().max() < 0.01


def test_fraud_rate_is_plausible(df):
    assert 0.005 < df["is_fraud"].mean() < 0.10


def test_amount_is_non_negative(df):
    assert (df["amount"] >= 0).all()


def test_transaction_id_is_unique(df): 
    assert df["transaction_id"].is_unique