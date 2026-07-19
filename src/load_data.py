"""
load_data.py - Load và validate IMDb dataset trước khi cho vào pipeline
"""

import pandas as pd
from pathlib import Path

def load_imdb_data(path: str) -> pd.DataFrame:
    """
    Load IMDb dataset từ file CSV, kiểm tra tính hợp lệ cơ bản.

    Args:
        path: đường dẫn tới file CSV (cột bắt buộc: 'review', 'sentiment')

    Returns:
        DataFrame đã validate, index reset lại từ 0

    Raises:
        FileNotFoundError: nếu file không tồn tại
        ValueError: nếu thiếu cột bắt buộc, hoặc dữ liệu có vấn đề nghiêm trọng
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file tại: {path}"
            f"Tải dataset và đặt vào data/raw/ trước khi run"
        )
    df = pd.read_csv(file_path)
    #Check schema
    requires_cols = {"review", "sentiment"}
    if not requires_cols.issubset(df.columns):
        raise ValueError(
            f"Thiếu cột bắt buộc. Cần: {requires_cols} có: {set(df.columns)}"
        )
    
    #Check null
    n_null = df[["review", "sentiment"]].isnull().sum().sum()
    if n_null > 0:
        print(f"[WARNING] Có {n_null} giá trị null - sẽ loại bỏ.")
        df = df.dropna(subset=["review", "sentiment"])
    #check label (only positive/negative)
    valid_labels = {"positive", "negative"}
    invalid = set(df["sentiment"].unique()) - valid_labels  #Trả về phần tử chỉ có ở set1 mà không có trong set2
    if invalid:
        raise ValueError(f"Có label không hợp lệ: {invalid}")
    
    #check duplicate và loại bỏ duplicate
    n_dup = df.duplicated(subset=["review"]).sum()
    if n_dup > 0:
        print(f"[INFO] Phát hiện {n_dup} review trùng lặp - sẽ loại bỏ.")
        df = df.drop_duplicates(subset=["review"], keep="first")
    df = df.reset_index(drop=True)
    return df
def summarize(df: pd.DataFrame) -> None:
    """In ra thông tin tổng quan để kiểm tra nhanh sau khi load"""
    print(f"Shape: {df.shape}")
    print(f"Class balance:\n{df['sentiment'].value_counts(normalize=True)}")
    print(f"Review length (words) — min/mean/max:")
    lengths = df["review"].str.split().str.len()
    print(f"  {lengths.min()} / {lengths.mean():.1f} / {lengths.max()}")

if __name__ == "__main__":
    df = load_imdb_data("data/raw/IMDB Dataset.csv")
    summarize(df)
    


    



