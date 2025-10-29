import pandas as pd
import xgboost as xgb
from src.preprocessing import basic_clean
from src.feature_engineering import add_rolling_features

def predict_new_transaction(new_tx: dict):
    model = xgb.Booster()
    model.load_model("xgb_model.json")

    tx_df = pd.DataFrame([new_tx])
    tx_df["TX_DATETIME"] = pd.to_datetime(tx_df["TX_DATETIME"])
    tx_df = basic_clean(tx_df)
    tx_df = add_rolling_features(tx_df)

    FEATURES = [
        "log_amount", "hour", "dayofweek",
        "cust_tx_count_7d", "cust_tx_mean_7d",
        "term_fraud_count_28d",
        "amount_ratio_to_cust_mean",
        "cust_amount_dev",
        "term_fraud_rate_28d",
    ]

    dtest = xgb.DMatrix(tx_df[FEATURES])
    proba = model.predict(dtest)[0]
    classification = "FRAUD" if proba > 0.5 else "LEGIT"
    print(f"Fraud Probability: {proba:.3f} → {classification}")
    return classification, proba

if __name__ == "__main__":
    tx = {
        "CUSTOMER_ID": 12345,
        "TERMINAL_ID": 501,
        "TX_AMOUNT": 250.0,
        "TX_DATETIME": "2018-05-02 10:45:00",
        "TX_FRAUD": 0
    }
    predict_new_transaction(tx)