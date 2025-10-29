import pandas as pd

def add_rolling_features(df):
    """Create time-based rolling features for customers & terminals."""
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df = df.sort_values("TX_DATETIME")

    cust_roll = (
        df.groupby("CUSTOMER_ID")
          .rolling("7D", on="TX_DATETIME")["TX_AMOUNT"]
          .agg(["count", "mean"])
          .rename(columns={"count": "cust_tx_count_7d", "mean": "cust_tx_mean_7d"})
          .reset_index()
    )
    df = df.merge(cust_roll, on=["CUSTOMER_ID", "TX_DATETIME"], how="left")

    term_roll = (
        df.groupby("TERMINAL_ID")
          .rolling("28D", on="TX_DATETIME")["TX_FRAUD"]
          .sum()
          .rename("term_fraud_count_28d")
          .reset_index()
    )
    df = df.merge(term_roll, on=["TERMINAL_ID", "TX_DATETIME"], how="left")

    df["cust_tx_count_7d"] = df["cust_tx_count_7d"].fillna(0)
    df["cust_tx_mean_7d"] = df["cust_tx_mean_7d"].fillna(0)
    df["term_fraud_count_28d"] = df["term_fraud_count_28d"].fillna(0)

    df["amount_ratio_to_cust_mean"] = df["TX_AMOUNT"] / (df["cust_tx_mean_7d"] + 1e-5)
    df["cust_amount_dev"] = df["TX_AMOUNT"] - df["cust_tx_mean_7d"]
    df["term_fraud_rate_28d"] = df["term_fraud_count_28d"] / (df["term_fraud_count_28d"] + 1)

    return df