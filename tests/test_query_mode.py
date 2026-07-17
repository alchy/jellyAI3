"""Režimy query cesty: templates = šablony JEDINÁ cesta (bez UDPipe fallbacku)."""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient

_PARSE = {"Kdo stvořil svět?": [[
    {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
    {"form": "stvořil", "lemma": "stvořit", "upos": "VERB", "head": 0, "deprel": "root"},
    {"form": "svět", "lemma": "svět", "upos": "NOUN", "head": 2, "deprel": "obj"},
]]}


def _graph():
    g = FactGraph()
    g.add_fact(make_fact("stvořit", [Participant("subj", "Bůh", "person"),
                                     Participant("obj", "svět", "concept")]))
    return g


def test_templates_mode_answers_without_fallback():
    a = GraphAnswerer(_graph(), FakeUfalClient(parse=_PARSE),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert a.answer("Kdo stvořil svět?", []).text == "Bůh"


def test_templates_mode_ignores_udpipe_pattern():
    """Prázdný graf → šablony predikát neznají → None; UDPipe by odpověď
    složil, ale v templates režimu NESMÍ naskočit — poctivé „nenašel"."""
    a = GraphAnswerer(FactGraph(), FakeUfalClient(parse=_PARSE),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert "nenašel" in a.answer("Kdo stvořil svět?", []).text


def test_hybrid_mode_falls_back_to_udpipe():
    """Hybrid: překlep „sworil" šablony nespárují (prefix 1 znak) → None →
    UDPipe rozbor (fake parse zná správná lemmata) odpověď složí."""
    q = "Kdo sworil svět?"
    parse = {q: _PARSE["Kdo stvořil svět?"]}
    a = GraphAnswerer(_graph(), FakeUfalClient(parse=parse),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="hybrid")
    assert a.answer(q, []).text == "Bůh"
    # tentýž překlep v templates režimu → poctivé nenašel (fallback nesmí naskočit)
    b = GraphAnswerer(_graph(), FakeUfalClient(parse=parse),
                      ExtractiveAnswerer(AnswererConfig()), query_mode="templates")
    assert "nenašel" in b.answer(q, []).text
