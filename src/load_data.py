"""
load_data.py - Load and validate IMDb dataset before passing to pipeline
"""

import pandas as pd
from pathlib import Path

def load_imdb_data(path: str) -> pd.DataFrame:
    """
    Load IMDb dataset from CSV file, perform basic validity checks.

    Args:
        path: path to CSV file (required columns: 'review', 'sentiment')

    Returns:
        Validated DataFrame, index reset from 0

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if required columns are missing, or severe data issues
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found at: {path}\n"
            f"Download dataset and place it in data/raw/ before running"
        )
    df = pd.read_csv(file_path)
    # Check schema
    requires_cols = {"review", "sentiment"}
    if not requires_cols.issubset(df.columns):
        raise ValueError(
            f"Missing required columns. Required: {requires_cols} got: {set(df.columns)}"
        )
    
    # Check nulls
    n_null = df[["review", "sentiment"]].isnull().sum().sum()
    if n_null > 0:
        print(f"[WARNING] Found {n_null} null values - will be dropped.")
        df = df.dropna(subset=["review", "sentiment"])
    # Check labels (only positive/negative)
    valid_labels = {"positive", "negative"}
    invalid = set(df["sentiment"].unique()) - valid_labels  # Returns elements only in set1 that aren't in set2
    if invalid:
        raise ValueError(f"Invalid labels found: {invalid}")
    
    # Check for duplicates and drop them
    n_dup = df.duplicated(subset=["review"]).sum()
    if n_dup > 0:
        print(f"[INFO] Detected {n_dup} duplicate reviews - will be dropped.")
        df = df.drop_duplicates(subset=["review"], keep="first")
    df = df.reset_index(drop=True)
    return df
def summarize(df: pd.DataFrame) -> None:
    """Print overview information for a quick check after loading"""
    print(f"Shape: {df.shape}")
    print(f"Class balance:\n{df['sentiment'].value_counts(normalize=True)}")
    print(f"Review length (words) — min/mean/max:")
    lengths = df["review"].str.split().str.len()
    print(f"  {lengths.min()} / {lengths.mean():.1f} / {lengths.max()}")

if __name__ == "__main__":
    df = load_imdb_data("data/raw/IMDB Dataset.csv")
    summarize(df)
    


    



