"""Case-insensitive rozřešení tématu — kapitalizovaný uzel („Vějíř") jde dotázat,
i když UDPipe lemmatizuje na malé („vějíř")."""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def test_lowercase_lemma_resolves_capitalized_title():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Jaroslav Seifert", "person"),
                                    Participant("obj", "Vějíř", "dílo")]))
    q = "Kdo napsal Vějíř?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Vějíř", "lemma": "vějíř", "upos": "NOUN", "head": 2, "deprel": "obj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [])
    assert ans.text == "Jaroslav Seifert"          # dřív selhalo (case-sensitive)


def test_case_variant_term_resolves_by_stem():
    """Skloněný termín („Galéna" — lemma, které UDPipe nechá v pádu) najde
    kanonický uzel „Galén" kmenovým fallbackem — týž mechanismus jako
    build-side resolver (canon._stem), takže se query a build nerozejdou."""
    g = FactGraph()
    g.add_fact(make_fact("léčit", [Participant("subj", "Galén", "person"),
                                   Participant("obj", "malomocenství", "concept")]))
    q = "Co léčil Galéna?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "léčil", "lemma": "léčit", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Galéna", "lemma": "Galéna", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "malomocenství"


def test_exact_case_still_distinguishes_book_from_common():
    """Zachovaná velikost (PROPN lemma „Babička") drží knihu odlišenou od „babička"."""
    g = FactGraph()
    for _ in range(9):     # obecná „babička" hodně častá
        g.add_fact(make_fact("péct", [Participant("subj", "babička", "concept"),
                                      Participant("obj", "povídka", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Božena Němcová"    # přesná shoda „Babička" > častá „babička"
