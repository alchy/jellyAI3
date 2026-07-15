from qagen.quality import is_acceptable


def test_good_question_accepted():
    assert is_acceptable("Kdo původně chtěl roboty nazvat laboři?", "Karel Čapek")


def test_leading_punctuation_rejected():
    assert not is_acceptable("Kdo , rodným jménem Karel?", "Karel")


def test_too_few_words_rejected():
    assert not is_acceptable("Kdy 1890 Malé Svatoňovice – 25?", "ledna")


def test_fragment_answer_rejected():
    assert not is_acceptable("Kdo je nejlepší?", "R")


def test_numeric_answer_ok():
    assert is_acceptable("Kolik planet má sluneční soustava?", "8")
