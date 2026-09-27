import pandas as pd

def load_dataset(path):
    df = pd.read_parquet(path)
    for col in ["name", "abstract"]:
        if col not in df.columns:
            raise ValueError(f"Dataset is missing required column: {col}")
    df["name"] = df["name"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    return df

def dataset_stats(df):
    return {
        "articles": int(len(df)),
        "columns": int(len(df.columns)),
        "words": int(df["abstract"].str.split().str.len().sum())
    }
