from spamguard.text_preprocessing import WordVocabulary, normalize_text, tokenize


def test_normalize_text_url_phone_email_replaced_with_tokens():
    text = "Call 0909 123 456 or visit http://win.com/abc, mail me a@b.com"
    result = normalize_text(text)
    assert "<phone>" in result and "<url>" in result and "<email>" in result
    assert "0909" not in result


def test_normalize_text_twitter_artifacts_removed():
    result = normalize_text("RT @someone: you are so dumb")
    assert "rt" not in result.split() and "<user>" in result


def test_normalize_text_vietnamese_diacritics_kept_and_nfc():
    decomposed = "Tiếng Việt"  # dạng tổ hợp
    assert normalize_text(decomposed) == "tiếng việt"


def test_normalize_text_cased_mode_keeps_uppercase():
    assert normalize_text("FREE MONEY", lowercase=False) == "FREE MONEY"


def test_normalize_text_repeated_chars_collapsed():
    assert normalize_text("cuteeeeee quáááá") == "cutee quáá"


def test_normalize_text_non_string_returns_empty():
    assert normalize_text(None) == ""


def test_vocabulary_unknown_token_mapped_to_unk_and_roundtrip():
    vocab = WordVocabulary.build(["hello world", "hello there"], vocab_size=10, max_len=5)
    ids = vocab.encode("hello zzz")
    assert ids[0] == vocab.token_to_id["hello"] and ids[1] == vocab.unk_id
    restored = WordVocabulary.from_dict(vocab.to_dict())
    assert restored.encode("hello zzz") == ids


def test_vocabulary_encode_batch_pads_to_same_length():
    vocab = WordVocabulary.build(["a b c", "a"], vocab_size=10, max_len=8)
    padded, lengths = vocab.encode_batch(["a b c", "a"])
    assert len(padded[0]) == len(padded[1]) == 3 and lengths == [3, 1]


def test_tokenize_keeps_special_tokens():
    assert tokenize("<url> hi!") == ["<url>", "hi", "!"]
