"""Clarify-identity (BACKLOG #43) — identita podmětu výroku bez tichého lepení.

„Emil bydlel v Brně." se na grafu s osobou „Emil Filla" nesmí připsat mlčky
(dialog > figly): automat NABÍDNE existující osobu i založení nové; volba
jménem, „ano" (= první nabídnutý) nebo slovem „nový" (tabulka
new_person_words). Otazník nabídku ruší — text jde jako nová otázka.
Přesné jméno („Emil Filla bydlel…") se nedoptává.
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def _iris():
    g = FactGraph()
    g.add_fact(make_fact("malovat", [
        Participant("subj", "Emil Filla", "person"),
        Participant("obj", "obrazy", "concept")]))
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             clock=lambda: NOW)
    return IrisAutomaton(answerer, clock=lambda: NOW)


def _subjects(graph, predicate):
    return {p.node for f in graph.facts.values() if f.predicate == predicate
            for p in f.participants if p.role == "subj"}


def test_inexact_person_asks_instead_of_gluing():
    iris = _iris()
    response = iris.turn("Emil bydlel v Brně.")
    assert response.kind == "dialog"
    assert "Emil Filla" in response.text
    assert "nov" in response.text.lower()          # nabídka nové osoby
    assert _subjects(iris.answerer.graph, "bydlet") == set()   # nic připsáno


def test_pick_existing_person_by_name():
    iris = _iris()
    iris.turn("Emil bydlel v Brně.")
    response = iris.turn("Filla")
    assert "Zapamatováno" in response.text
    assert _subjects(iris.answerer.graph, "bydlet") == {"Emil Filla"}


def test_pick_new_person_by_word():
    iris = _iris()
    iris.turn("Emil bydlel v Brně.")
    response = iris.turn("nový")
    assert "Zapamatováno" in response.text
    assert _subjects(iris.answerer.graph, "bydlet") == {"Emil"}


def test_ano_means_first_offered():
    iris = _iris()
    iris.turn("Emil bydlel v Brně.")
    iris.turn("ano")
    assert _subjects(iris.answerer.graph, "bydlet") == {"Emil Filla"}


def test_question_cancels_offer():
    iris = _iris()
    iris.turn("Emil bydlel v Brně.")
    response = iris.turn("Kdo maloval obrazy?")
    assert "Zapamatováno" not in response.text
    assert iris.state.pending is None
    assert _subjects(iris.answerer.graph, "bydlet") == set()


def test_exact_name_memorizes_without_asking():
    iris = _iris()
    response = iris.turn("Emil Filla bydlel v Brně.")
    assert "Zapamatováno" in response.text
    assert _subjects(iris.answerer.graph, "bydlet") == {"Emil Filla"}
