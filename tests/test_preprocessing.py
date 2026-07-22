from src.preprocessing import normalize_teencode, load_teencode_dict

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