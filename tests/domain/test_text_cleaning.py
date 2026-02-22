from spam_detector.domain.text_cleaning import clean_text


def test_clean_text_removes_urls_and_symbols():
    raw = "WIN $$$ now!!! Visit https://example.com today"

    cleaned = clean_text(raw)

    assert "http" not in cleaned
    assert "$" not in cleaned
    assert cleaned == "win now visit today"


def test_clean_text_normalizes_whitespace_and_case():
    raw = "  Hello\n\tWORLD   "

    cleaned = clean_text(raw)

    assert cleaned == "hello world"
