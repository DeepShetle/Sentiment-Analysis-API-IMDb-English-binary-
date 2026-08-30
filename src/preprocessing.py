"""
preprocessing.py — common text normalization functions used for both training and serving.
"""

import json
import re
import emoji


def load_teencode_dict(path: str) -> dict:
    """Load teencode dictionary from JSON file."""
    with open(path, "r", encoding="utf-8") as f:    # Open and auto-close file
        return json.load(f)


def normalize_teencode(text: str, teencode_map: dict) -> str:
    """
    Replace teencode/abbreviations in text with their full forms.
    Uses word-boundary regex to avoid partial matches inside other words
    (e.g. "lol" won't match inside "lollipop").
    """
    for slang, full_form in teencode_map.items():
        pattern = r"\b" + re.escape(slang) + r"\b"
        text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
    return text
def translate_emoji(text: str) -> str:
    text = emoji.demojize(text)
    text = text.replace(":", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text # Can be improved later (e.g., manual mapping for common emojis)
def clean_html(text: str) -> str:
    return re.sub(r"<.*?>", " ", text) # Remove HTML tags
# Ignore HTML entities like &amp, &quot due to negligible amount
def preprocess_pipeline(text: str, teencode_map: dict) -> str:
    text = text.lower()
    text = clean_html(text)
    text = normalize_teencode(text, teencode_map)
    text = translate_emoji(text)
    return text
