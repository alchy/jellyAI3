"""Zjišťovací (ano/ne) otázky — existence faktu, žádné kontextové hádání."""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _graph():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "válka", "concept")]))
    for _ in range(5):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "Josef Čapek", "person")]))
    return g


def _parse(verb_form, verb_lemma):
    return [[
        {"form": verb_form, "lemma": verb_lemma, "upos": "VERB", "head": 0,
         "deprel": "root"},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 1,
         "deprel": "nsubj"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2,
         "deprel": "flat"},
        {"form": "válku", "lemma": "válka", "upos": "NOUN", "head": 1,
         "deprel": "obj"},
    ]]


def test_yes_when_fact_exists():
    q = "Napsal Karel Čapek válku?"
    client = FakeUfalClient(parse={q: _parse("Napsal", "napsat")})
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Ano"


def test_unknown_when_fact_missing_no_context_guess():
    """Bez faktu se nehádá z kontextu (dialog: „Publikoval…?" → Josef Čapek)."""
    q = "Publikoval Karel Čapek válku?"
    client = FakeUfalClient(parse={q: _parse("Publikoval", "publikovat")})
    a = GraphAnswerer(_graph(), client, ExtractiveAnswerer(AnswererConfig()))
    text = a.answer(q, []).text
    assert "Josef" not in text and "Ano" not in text
