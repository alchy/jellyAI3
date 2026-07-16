"""Rozklad otázky na neúplný fakt — pattern (predikát, známé role, díra).

Holý genitiv (bez předložky) spouští vnořený pod-dotaz; nmod s předložkou
(„Válku **s mloky**", „drama **o robotech**") je součást termínu/fráze, ne
genitivní vztah — jinak by se víceslovný titul rozpadl na nesmyslnou rekurzi.
"""

from jellyai.answerer.pattern import question_pattern, SubQuery
from jellyai.ufal_client import FakeUfalClient


def test_bare_genitive_triggers_subquery():
    q = "Kdo byl bratr Karla Čapka?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "bratr", "lemma": "bratr", "upos": "NOUN", "head": 0, "deprel": "root"},
        {"form": "Karla", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nmod"},
        {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat"},
    ]]})
    pat = question_pattern(q, client)
    assert pat.predicate == "bratr"
    assert pat.known and not isinstance(pat.known[0][1], SubQuery)   # přímý termín


def test_interrogative_root_copula_canonicalizes_identity():
    """„Co je robot?" — parser dává kořen tázacímu „Co"; pattern přesto složí
    identitu být(robot → díra pred), místo predicate=None (wildcard býval
    zdrojem odpovědi „Karel Antonín Čapek" přes nazvat)."""
    q = "Co je robot?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 0, "deprel": "root",
         "feats": {"PronType": "Int"}},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 1, "deprel": "cop"},
        {"form": "robot", "lemma": "robot", "upos": "NOUN", "head": 1,
         "deprel": "nsubj"},
    ]]})
    pat = question_pattern(q, client)
    assert pat.predicate == "být"
    assert pat.known == [("subj", "robot")]
    assert pat.hole_role == "pred"


def test_prepositional_nmod_stays_inside_term():
    """„Kdo napsal Válku s mloky?" — „s mloky" je předložková fráze titulu,
    NE genitivní pod-dotaz válka(mlok)."""
    q = "Kdo napsal Válku s mloky?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Válku", "lemma": "válka", "upos": "NOUN", "head": 2, "deprel": "obj"},
        {"form": "s", "lemma": "s", "upos": "ADP", "head": 5, "deprel": "case"},
        {"form": "mloky", "lemma": "mlok", "upos": "NOUN", "head": 3, "deprel": "nmod"},
    ]]})
    pat = question_pattern(q, client)
    assert pat.predicate == "napsat"
    assert pat.known, "termín se nesmí ztratit"
    role, term = pat.known[0]
    assert role == "obj" and not isinstance(term, SubQuery)
    assert term == "válka"


def test_oblique_participant_becomes_known():
    """„Jak to souviselo s Karlem Čapkem?" — obl osoba je known účastník
    (dřív se zahazovala a pattern zůstal prázdný)."""
    q = "Jak to souviselo s Karlem Čapkem?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Jak", "lemma": "jak", "upos": "ADV", "head": 3, "deprel": "advmod"},
        {"form": "to", "lemma": "ten", "upos": "DET", "head": 3, "deprel": "nsubj",
         "feats": {"PronType": "Dem"}},
        {"form": "souviselo", "lemma": "souviset", "upos": "VERB", "head": 0,
         "deprel": "root"},
        {"form": "s", "lemma": "s", "upos": "ADP", "head": 5, "deprel": "case"},
        {"form": "Karlem", "lemma": "Karel", "upos": "PROPN", "head": 3,
         "deprel": "obl"},
        {"form": "Čapkem", "lemma": "Čapek", "upos": "PROPN", "head": 5,
         "deprel": "flat"},
    ]]})
    pat = question_pattern(q, client)
    assert ("theme", "Karel Čapek") in pat.known


def test_elliptic_question_without_verb_is_copular():
    """„Jaká rodina?" (bez slovesa i spony) = eliptická identita:
    být(rodina → díra attr)."""
    q = "Jaká rodina?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Jaká", "lemma": "jaký", "upos": "DET", "head": 2, "deprel": "amod",
         "feats": {"PronType": "Int"}},
        {"form": "rodina", "lemma": "rodina", "upos": "NOUN", "head": 0,
         "deprel": "root"},
    ]]})
    pat = question_pattern(q, client)
    assert pat.predicate == "být"
    assert pat.known == [("subj", "rodina")]
    assert pat.hole_role == "attr"
