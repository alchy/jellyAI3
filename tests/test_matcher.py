"""Vykonavatel vzorů (#46 fáze 2) — regulární sekvence tříd nad lexerem.

Vzor je PŘÍSNĚ regulární (spec §7): třídy, literály (s alternativami |),
volitelnost (prefix ?). Žádné podmínky, proměnné, vnořování. Match je
ukotvený na CELOU větu (žádné kradení podobných otázek) a vrací vazby
1-based indexů prvků na tokeny (volitelný nenaplněný prvek → None).
"""

from jellyai.lang.lexer import classify
from jellyai.lang.matcher import match_sequence


def test_class_and_literal_elements_bind_tokens():
    tagged = classify("Kdo bydlí v Petrovicích.")
    binding = match_sequence(
        ["otaz:kdo", "sloveso_fin", ":v|ve|na", "jmeno"], tagged)
    assert binding is not None
    assert binding[1].form == "Kdo"
    assert binding[2].form == "bydlí"
    assert binding[4].form == "Petrovicích"


def test_optional_element_binds_or_none():
    seq = ["otaz:kdo", "?:dalsi", "sloveso_fin", ":v|ve|na", "jmeno"]
    with_opt = match_sequence(seq, classify("Kdo další bydlí v Petrovicích."))
    assert with_opt is not None and with_opt[2].form == "další"
    without = match_sequence(seq, classify("Kdo bydlí v Petrovicích."))
    assert without is not None and without[2] is None
    assert without[5].form == "Petrovicích"


def test_match_is_anchored_to_whole_sentence():
    seq = ["otaz:kdo", "sloveso_fin", ":v|ve|na", "jmeno"]
    assert match_sequence(seq, classify("Kdo napsal knihu o Praze.")) is None
    assert match_sequence(seq, classify("Kdo bydlí v Petrovicích rád.")) is None
    assert match_sequence(seq, classify("Prší.")) is None


def test_norm_alternatives_are_deaccented():
    seq = ["otaz:cim", "sloveso_fin", "jmeno", "?jmeno"]
    binding = match_sequence(seq, classify("Čím krmí Marcela Roníka."))
    assert binding is not None
    assert binding[1].form == "Čím"
    assert binding[3].form == "Marcela" and binding[4].form == "Roníka"
