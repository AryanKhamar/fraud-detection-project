import pandas as pd
from pathlib import Path

def load_data(data_dir="data"):
    """Load and combine all .pkl transaction files."""
    data_dir = Path(data_dir)
    dfs = []
    for file in sorted(data_dir.glob("*.pkl")):
        print(f"Loading {file.name} ...")
        dfs.append(pd.read_pickle(file))
    df = pd.concat(dfs, ignore_index=True)
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df = df.sort_values("TX_DATETIME").reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} transactions from {len(dfs)} files.")
    return df