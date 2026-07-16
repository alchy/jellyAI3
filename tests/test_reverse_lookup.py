"""Reverzní dotaz — z data v otázce najde událost (datum → co/kdo se stalo).

Poslední záchrana, když forward průchod nenajde odpověď: v grafu už `facts_of`
indexuje po uzlu, takže z časového uzlu dojdeme na podmět jeho faktu.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def test_reverse_date_to_event():
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("time", "2. května 1818", "time")]))
    q = "Co se stalo 2. května 1818?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 3, "deprel": "obj"},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl:pv"},
        {"form": "stalo", "lemma": "stát", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "2", "lemma": "2", "upos": "NUM", "head": 5, "deprel": "nummod"},
        {"form": "května", "lemma": "květen", "upos": "NOUN", "head": 3, "deprel": "obl"},
        {"form": "1818", "lemma": "1818", "upos": "NUM", "head": 5, "deprel": "nummod"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [])
    assert ans.text == "Božena Němcová"
    assert ans.trace["topic"] == "2. května 1818"
    assert ans.trace["predicate"] == "narodit"


def test_reverse_with_temperature_returns_context_rows():
    """Fuzzy (temperature) → k primární odpovědi i další kontext s menší vahou."""
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena", "person"),
                                     Participant("time", "1818", "time")]))
    g.add_fact(make_fact("zemřít", [Participant("subj", "Jan", "person"),
                                    Participant("time", "1818", "time")]))
    q = "Co se stalo roku 1818?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "stalo", "lemma": "stát", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "roku", "lemma": "rok", "upos": "NOUN", "head": 2, "deprel": "obl"},
        {"form": "1818", "lemma": "1818", "upos": "NUM", "head": 3, "deprel": "nummod"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [], temperature=0.6)
    assert ans.text in ("Božena", "Jan")
    others = {"Božena", "Jan"} - {ans.text}
    assert others <= set(ans.alternatives)   # druhá událost jako kontext s menší vahou


def test_no_year_no_reverse():
    """Otázka bez roku reverzní lookup nespustí (nezmate běžné dotazy)."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "Co se děje?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "děje", "lemma": "dít", "upos": "VERB", "head": 0, "deprel": "root"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [])
    assert ans.trace is None
