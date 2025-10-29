import numpy as np
import pandas as pd

def basic_clean(df):
    """Basic cleaning & derived columns."""
    df = df.copy()
    df["hour"] = df["TX_DATETIME"].dt.hour
    df["dayofweek"] = df["TX_DATETIME"].dt.dayofweek
    df["log_amount"] = np.log1p(df["TX_AMOUNT"])
    return df