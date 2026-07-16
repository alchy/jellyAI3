"""Attention nad kontextem — navazující dotaz bez tématu vybere z aktivačního pole
uzel, který na otázku *umí odpovědět*, ne slepě nejteplejší."""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _kdy_se_narodila_parse():
    q = "kdy se narodila?"
    return q, FakeUfalClient(parse={q: [[
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod"},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl:pv"},
        {"form": "narodila", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root"},
    ]]})


def test_attention_skips_hotter_irrelevant_node():
    """Nejteplejší uzel (Karel) nemá narození → attention najde Boženu, co narození má."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Karel", "person"),
                                 Participant("pred", "spisovatel", "concept")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena", "person"),
                                     Participant("time", "1818", "time")]))
    q, client = _kdy_se_narodila_parse()
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.context.warm("Karel", 2.0)      # teplejší, ale narození nemá
    a.context.warm("Božena", 0.5)     # chladnější, ale narození má
    ans = a.answer(q, [])
    assert ans.text == "1818"
    assert ans.trace["topic"] == "Božena"


def test_attention_respects_verb_gender():
    """„Kdy se narodil?" (mužský rod) přeskočí ženskou Boženu a najde Karla."""
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("time", "1818", "time")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                                     Participant("time", "1890", "time")]))
    q = "kdy se narodil?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 3, "deprel": "advmod"},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl:pv"},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.context.warm("Božena Němcová", 2.0)   # ženská, teplejší
    a.context.warm("Karel Čapek", 0.5)      # mužský, chladnější
    ans = a.answer(q, [])
    assert ans.text == "1890"               # mužský rod → Karel
    assert ans.trace["topic"] == "Karel Čapek"


def test_named_unknown_topic_does_not_guess_from_context():
    """Pojmenované, ale neznámé téma (RUR) → neodpovídat z kontextu (Boženy)."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Božena", "person"),
                                 Participant("pred", "spisovatelka", "concept")]))
    q = "Kdo je RUR?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "RUR", "lemma": "RUR", "upos": "PROPN", "head": 2, "deprel": "obl"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.context.warm("Božena", 2.0)     # Božena je nejteplejší, ale otázka je o RUR
    ans = a.answer(q, [])
    assert ans.trace is None          # neuhodne Boženu


def test_no_context_no_answer_falls_back():
    """Bez tématu i bez kontextu → fallback (žádná trasa)."""
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena", "person"),
                                     Participant("time", "1818", "time")]))
    q, client = _kdy_se_narodila_parse()
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [])             # prázdný kontext
    assert ans.trace is None
