"""
preprocessing.py — các hàm chuẩn hóa text, dùng chung cho cả training và serving.
"""

import json
import re
import emoji


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
def translate_emoji(text: str) -> str:
    text = emoji.demojize(text)
    text = text.replace(":", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text #Cải thiện khả thi ở giai đoạn sau (Map thủ công một vài emoji thường thấy)
def clean_html(text: str) -> str:
    return re.sub(r"<.*?>", " ", text) #Xóa tag HTML
#Bỏ qua xử lí các entity: &amp, &quot, do số lượng quá nhỏ (4), không ảnh hưởng
def preprocess_pipeline(text: str, teencode_map: dict) -> str:
    text = text.lower()
    text = clean_html(text)
    text = normalize_teencode(text, teencode_map)
    text = translate_emoji(text)
    return text
