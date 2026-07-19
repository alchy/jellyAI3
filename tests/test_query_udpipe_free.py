"""Čistý řez #14: dotazová cesta je BEZ UDPipe.

Šablony (pseudo-QL + vzorové karty) jsou JEDINÝ rozbor otázky — žádný
fallback, žádné režimy. ÚFAL zůstává jen v anotaci korpusu a
nominativizaci Mnemos; `question_pattern` je zakonzervován
(docs/superpowers/conserved-components.md).
"""

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


class _NoParseClient(FakeUfalClient):
    def parse(self, text):
        raise AssertionError("dotazová strana volala UDPipe parse!")


def test_query_path_answers_via_templates():
    a = GraphAnswerer(_graph(), FakeUfalClient(parse=_PARSE),
                      ExtractiveAnswerer(AnswererConfig()))
    assert a.answer("Kdo stvořil svět?", []).text == "Bůh"


def test_empty_graph_is_honest_miss_not_udpipe():
    """Prázdný graf → šablony predikát neznají → poctivé „nenašel";
    UDPipe by odpověď složil, ale naskočit NESMÍ."""
    a = GraphAnswerer(FactGraph(), FakeUfalClient(parse=_PARSE),
                      ExtractiveAnswerer(AnswererConfig()))
    assert "nenašel" in a.answer("Kdo stvořil svět?", []).text


def test_query_path_never_calls_parse():
    """Celá cesta otázka→odpověď bez jediného parse (řez #14)."""
    a = GraphAnswerer(_graph(), _NoParseClient(),
                      ExtractiveAnswerer(AnswererConfig()))
    assert a.answer("Kdo stvořil svět?", []).text == "Bůh"


def test_typo_predicate_is_honest_miss():
    """Překlep „sworil": dřív ho zachraňoval UDPipe fallback — po řezu
    je odpovědí poctivé „nenašel" (nehádat je zákon)."""
    a = GraphAnswerer(_graph(), _NoParseClient(),
                      ExtractiveAnswerer(AnswererConfig()))
    assert "nenašel" in a.answer("Kdo sworil svět?", []).text
