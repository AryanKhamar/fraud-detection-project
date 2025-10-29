import pandas as pd
from src.data_loader import load_data
from src.preprocessing import basic_clean
from src.feature_engineering import add_rolling_features
from src.model import train_logistic, train_xgboost
from src.evaluation import evaluate_model

def main():
    print("🔹 Loading data...")
    df = load_data("data")

    print("🔹 Cleaning data...")
    df = basic_clean(df)

    print("🔹 Creating features...")
    df = add_rolling_features(df)

    FEATURES = [
        "log_amount", "hour", "dayofweek",
        "cust_tx_count_7d", "cust_tx_mean_7d",
        "term_fraud_count_28d",
        "amount_ratio_to_cust_mean",
        "cust_amount_dev",
        "term_fraud_rate_28d",
    ]
    TARGET = "TX_FRAUD"

    split_time = df["TX_DATETIME"].quantile(0.8)
    train_df = df[df["TX_DATETIME"] <= split_time]
    test_df = df[df["TX_DATETIME"] > split_time]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print("🔹 Training Logistic Regression (baseline)...")
    log_model = train_logistic(X_train, y_train)
    evaluate_model(log_model, X_test, y_test)

    print("\n🔹 Training XGBoost (advanced model)...")
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, xgb_model=True)

if __name__ == "__main__":
    main()