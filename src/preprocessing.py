"""
preprocessing.py — các hàm chuẩn hóa text, dùng chung cho cả training và serving.
"""

import json
import re


def load_teencode_dict(path: str) -> dict:
    """Load teencode dictionary từ file JSON."""
    with open(path, "r", encoding="utf-8") as f:    #Mở + tự động đóng tệp
        return json.load(f)


def normalize_teencode(text: str, teencode_map: dict) -> str:
    """
    Thay thế teencode/viết tắt trong text bằng dạng chuẩn.
    Dùng word-boundary regex để tránh match nhầm bên trong từ khác
    (ví dụ "lol" không được match bên trong "lollipop").
    """
    for slang, full_form in teencode_map.items():
        pattern = r"\b" + re.escape(slang) + r"\b"
        text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
    return text