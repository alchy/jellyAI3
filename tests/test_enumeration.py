"""Enumerativní odpovědi — otázka s víc rovnocennými fakty vrací výčet.

„Co napsal Karel Čapek?" nemá jednu odpověď; _match vrací všechny díry se
shodným top skóre a odpověď je vyjmenuje. Jednohodnotové otázky (bratr,
narození) zůstávají jednohodnotové — výčet vzniká jen z rovnocenných faktů.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def test_multiple_equal_facts_enumerate():
    g = FactGraph()
    for title in ["R.U.R.", "Krakatit", "Matka"]:
        g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                        Participant("obj", title, "dílo")]))
    q = "Co napsal Karel Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    text = a.answer(q, []).text
    assert "R.U.R." in text and "Krakatit" in text and "Matka" in text


def test_heavier_fact_stays_single():
    """Fakt s vyšší vahou výčet nedostane — vyjmenovávají se jen rovnocenné."""
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("bratr", [Participant("subj", "Karel Čapek", "person"),
                                       Participant("obj", "Josef Čapek", "person")]))
    g.add_fact(make_fact("bratr", [Participant("subj", "Karel Čapek", "person"),
                                   Participant("obj", "Omyl Šum", "person")]))
    q = "Kdo byl bratr Karla Čapka?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "bratr", "lemma": "bratr", "upos": "NOUN", "head": 0, "deprel": "root"},
        {"form": "Karla", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nmod"},
        {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Josef Čapek"


def _client_co_napsal():
    q = "Co napsal Karel Čapek?"
    return q, FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "flat"},
    ]]})


def test_enumeration_ties_ignore_activation_for_grouping():
    """Aktivace remízu jen ŘADÍ, nerozbíjí — opakovaná otázka nesmí ztrácet
    hodnoty (dialog: „Co napsal?" → hra, válka; podruhé už jen hra)."""
    g = FactGraph()
    for title in ["hra", "válka"]:
        g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                        Participant("obj", title, "concept")]))
    q, client = _client_co_napsal()
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.context.warm("válka", 3.0)              # dřívější zmínka ve stejné konverzaci
    text = a.answer(q, []).text
    assert "hra" in text and "válka" in text  # výčet drží obě
    assert text.startswith("válka")           # aktivace řadí dopředu


def test_enumeration_caps_and_filters_junk():
    """Strop výčtu (5) a filtr jednoznakových artefaktů NER („V")."""
    g = FactGraph()
    for title in ["a1", "a2", "a3", "a4", "a5", "a6", "V"]:
        g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                        Participant("obj", title, "concept")]))
    q, client = _client_co_napsal()
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    text = a.answer(q, []).text
    assert "V" not in text.split(", ")
    assert len(text.split(", ")) == 5
