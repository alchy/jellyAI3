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


def test_split_sentences_keeps_dates_and_abbrev():
    assert split_sentences("Narodil se 9. ledna 1890 v Úpici.") == \
        ["Narodil se 9. ledna 1890 v Úpici."]
    assert split_sentences("Ošetřil ho MUDr. Čapek osobně.") == \
        ["Ošetřil ho MUDr. Čapek osobně."]


def test_split_sentences_splits_number_before_capital():
    assert split_sentences("Věta číslo 0. Věta číslo 1.") == \
        ["Věta číslo 0.", "Věta číslo 1."]
