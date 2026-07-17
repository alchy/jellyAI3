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


def test_identity_never_echoes_topic_word():
    """„Kdo je Adam stvořitel?" nesmí odpovědět „stvořitel" — hodnota, která
    je slovem id samotného tématu (echo apozice titulu), je tautologie."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Adam stvořitel", "person"),
                                 Participant("pred", "stvořitel", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Adam stvořitel", "person"),
                                 Participant("pred", "hra", "concept")]))
    q = "Kdo je Adam stvořitel?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "Adam", "lemma": "Adam", "upos": "PROPN", "head": 0, "deprel": "root"},
        {"form": "stvořitel", "lemma": "stvořitel", "upos": "NOUN", "head": 3,
         "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "hra"


def test_semantic_hole_requires_role_or_type_match():
    """„Kdy zemřel X?" nesmí odpovědět tématem faktu bez času — sémantická
    díra (time/loc/num) vyžaduje shodu role nebo typu, jinak mlčí."""
    g = FactGraph()
    g.add_fact(make_fact("zemřít", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("theme", "náznak", "concept")]))
    q = "Kdy zemřel Karel Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdy", "lemma": "kdy", "upos": "ADV", "head": 2, "deprel": "advmod"},
        {"form": "zemřel", "lemma": "zemřít", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert "náznak" not in a.answer(q, []).text


def test_relational_noun_is_not_identity_answer():
    """„Kdo je Josef Čapek?" nesmí odpovědět „bratr" — vztahové jméno
    (jazyková tabulka) není identita osoby bez protistrany; profese ano."""
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("být", [Participant("subj", "Josef Čapek", "person"),
                                     Participant("pred", "bratr", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Josef Čapek", "person"),
                                 Participant("pred", "malíř", "concept")]))
    q = "Kdo je Josef Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 0, "deprel": "root"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "malíř"


def test_relational_kind_from_apposition_is_valid_identity():
    """„Kdo je Maria?" — druh(Maria, matka) z apozice JE platná identita
    (jediná, co text nese); filtr vztahových jmen platí jen pro sponové být."""
    g = FactGraph()
    g.add_fact(make_fact("druh", [Participant("subj", "Maria", "person"),
                                  Participant("pred", "matka", "concept")]))
    q = "Kdo je Maria?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "Maria", "lemma": "Maria", "upos": "PROPN", "head": 0, "deprel": "root"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "matka"
