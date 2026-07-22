from src.preprocessing import normalize_teencode, load_teencode_dict
from src.preprocessing import clean_html, translate_emoji, preprocess_pipeline
import pytest

def test_normalize_teencode_basic():
    teencode_map = {"lol": "laugh out loud"}
    result = normalize_teencode("this movie is lol good", teencode_map)
    assert "laugh out loud" in result

def test_normalize_teencode_no_partial_match():
    teencode_map = {"lol": "laugh out loud"}
    result = normalize_teencode("i love lollipop candy", teencode_map)
    assert "laugh out loud" not in result  # không được match nhầm trong "lollipop"

def test_normalize_teencode_case_insensitive():
    teencode_map = {"lol": "laugh out loud"}
    result = normalize_teencode("LOL that was funny", teencode_map)
    assert "laugh out loud" in result.lower()

def test_normalize_teencode_unknown_slang_unchanged():
    teencode_map = {"lol": "laugh out loud"}
    result = normalize_teencode("this uses j4f which is not in dict", teencode_map)
    assert "j4f" in result  # từ không có trong dict giữ nguyên

@pytest.mark.parametrize("raw,expected_removed", [
    ("great movie<br />", "<br"),
    ("<b>bold</b> text", "<b"),
])

def test_clean_html_removes_tags(raw, expected_removed):
    result = clean_html(raw)
    assert expected_removed not in result

def test_translate_emoji_converts_to_text():
    result = translate_emoji("nice 😊")
    assert "smiling" in result.lower()
    assert "😊" not in result

def test_translate_emoji_no_emoji_unchanged():
    assert translate_emoji("great movie") == "great movie"

def test_pipeline_full_flow():
    teencode_map = {"lol": "laugh out loud"}
    result = preprocess_pipeline("Great movie<br /> lol 😊", teencode_map)
    assert "<br" not in result
    assert "laugh out loud" in result
    assert "😊" not in result

def test_pipeline_empty_string():
    teencode_map = {"lol": "laugh out loud"}
    assert preprocess_pipeline("", teencode_map) == ""
