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
