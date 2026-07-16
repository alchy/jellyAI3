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
    assert ans.text == "narodit: Božena Němcová"    # děj = sloveso první
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
    subject = ans.text.split(": ")[1]        # text = „děj: účastníci"
    assert subject in ("Božena", "Jan")
    others = {"Božena", "Jan"} - {subject}
    assert others <= set(ans.alternatives)   # druhá událost jako kontext s menší vahou


def test_reverse_includes_neighborhood_of_answer():
    """„Co se stalo <datum>?" → primární podmět + jeho okolí (Božena → Babička)."""
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("time", "1818", "time")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "Co se stalo roku 1818?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "stalo", "lemma": "stát", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "roku", "lemma": "rok", "upos": "NOUN", "head": 2, "deprel": "obl"},
        {"form": "1818", "lemma": "1818", "upos": "NUM", "head": 3, "deprel": "nummod"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [], temperature=0.5)
    assert ans.text == "narodit: Božena Němcová"
    assert "Babička" in ans.alternatives   # okolí odpovědi jako souvislost


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


def test_reverse_lookup_respects_month():
    """„Co se stalo v listopadu 1848?" nesmí vzít uzel „července 1848" —
    měsíc z otázky se páruje s měsícem časového uzlu."""
    g = FactGraph()
    g.add_fact(make_fact("ocitnout", [Participant("subj", "rodina", "concept"),
                                      Participant("time", "července 1848", "time")]))
    g.add_fact(make_fact("vyjít", [Participant("subj", "svazek", "concept"),
                                   Participant("time", "listopadu 1848", "time")]))
    q = "Co se stalo v listopadu 1848?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl"},
        {"form": "stalo", "lemma": "stát", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "v", "lemma": "v", "upos": "ADP", "head": 5, "deprel": "case"},
        {"form": "listopadu", "lemma": "listopad", "upos": "NOUN", "head": 3, "deprel": "obl"},
        {"form": "1848", "lemma": "1848", "upos": "NUM", "head": 5, "deprel": "nummod"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    text = a.answer(q, []).text
    assert "svazek" in text and "rodina" not in text
    assert text.startswith("vyjít")   # děj = sloveso první
