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


def _is_span(span):
    return span in ("Karel Čapek", "Válka s mloky", "Válku s mloky",
                    "Marcel", "Babičku")


def test_span_element_matches_multiword_entity_greedily():
    tagged = classify("Napsal Karel Čapek Válku s mloky.")
    binding = match_sequence(["l_tvar", "uzel+", "?uzel+"], tagged,
                             is_span=_is_span)
    assert binding is not None
    assert [t.form for t in binding[2]] == ["Karel", "Čapek"]
    assert [t.form for t in binding[3]] == ["Válku", "s", "mloky"]


def test_span_element_backtracks_to_let_rest_match():
    """Hladový span musí ustoupit, aby zbytek vzoru vyšel — druhá entita
    existuje, jen ji nesmí spolknout první."""
    tagged = classify("Napsal Karel Čapek Babičku.")
    binding = match_sequence(["l_tvar", "uzel+", "uzel+"], tagged,
                             is_span=_is_span)
    assert binding is not None
    assert [t.form for t in binding[2]] == ["Karel", "Čapek"]
    assert [t.form for t in binding[3]] == ["Babičku"]


def test_span_element_requires_validation():
    tagged = classify("Napsal Karel Čapek Válku s mloky.")
    assert match_sequence(["l_tvar", "uzel+"], tagged,
                          is_span=_is_span) is None      # zbytek nespolkne
    assert match_sequence(["l_tvar", "uzel+", "?uzel+"], tagged,
                          is_span=None) is None          # bez orákula nematchne


def test_optional_span_absent_binds_none():
    tagged = classify("Prší.")
    binding = match_sequence(["sloveso_fin", "?uzel+"], tagged,
                             is_span=_is_span)
    assert binding is not None and binding[2] is None


def test_exclusion_class_keeps_copula_out():
    """„byl" je hypotézově l_tvar I spona — prvek `l_tvar!spona` sponové
    (identitní) otázky nechává poziční šabloně (etalon Kdo byl robot?)."""
    tagged = classify("Kdo byl robot.")
    assert match_sequence(["otaz", "l_tvar!spona", "uzel+"], tagged,
                          is_span=lambda s: s == "robot") is None
    tagged = classify("Kdo napsal Babičku.")
    assert match_sequence(["otaz", "l_tvar!spona", "uzel+"], tagged,
                          is_span=lambda s: s == "Babičku") is not None
