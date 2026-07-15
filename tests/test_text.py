from jellyai.text import tokenize, split_sentences, CZECH_STOPWORDS


def test_tokenize_keeps_diacritics_lowercases():
    assert tokenize("Příliš žluťoučký kůň") == ["příliš", "žluťoučký", "kůň"]


def test_tokenize_removes_stopwords():
    toks = tokenize("kdo vyráběl roboty", CZECH_STOPWORDS)
    assert "kdo" not in toks
    assert "roboty" in toks


def test_split_sentences():
    text = "Roboty vyráběla firma. Starý Rossum je vynalezl! Kdo to ví?"
    assert split_sentences(text) == [
        "Roboty vyráběla firma.",
        "Starý Rossum je vynalezl!",
        "Kdo to ví?",
    ]


def test_split_sentences_empty():
    assert split_sentences("   ") == []
