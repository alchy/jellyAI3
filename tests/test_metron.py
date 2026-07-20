"""Metron — aritmetický expert (BACKLOG #56, zadání user).

Výpočty z řádku uživatele: „Kolik je 1 plus 1?", „3 * 4", součty pro
účtenku („sto, osmdesátdva, třicet, 50, 32 … dohromady?"). Čísla a
operátory se EXTRAHUJÍ z výroku (cifry, slovní číslovky vč. složenin,
operátory slovem i znakem), počítá bezpečná aritmetika — nikdy eval.
Odpověď nese přepis výrazu (kontrola pro uživatele).
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.subsystems.metron import compute
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def test_expression_with_word_and_symbol_operators():
    assert compute("Kolik je 1 plus 1?") == ("1 + 1", "2")
    assert compute("Kolik je 1 + 1?") == ("1 + 1", "2")
    assert compute("Kolik je 1 * 2?") == ("1 * 2", "2")
    assert compute("Kolik je 10 děleno 4?") == ("10 / 4", "2,5")


def test_operator_precedence_without_eval():
    assert compute("Kolik je 2 plus 3 krát 4?") == ("2 + 3 * 4", "14")


def test_word_numerals_and_compounds_sum():
    """Účtenka: slovní číslovky (i SLOŽENINY: osmdesátdva) a cifry
    smíšeně; fráze „dohromady/celkem" spouští součet."""
    got = compute("Ověř mi součty - sto, osmdesátdva, třicet, 50, 32. "
                  "Kolik je to dohromady?")
    assert got == ("100 + 82 + 30 + 50 + 32", "294")


def test_declines_graph_questions_and_times():
    """Metron se NEhlásí bez výrazu: otázky grafu, časy, jediné číslo."""
    assert compute("Kolik měla dětí Božena Němcová?") is None
    assert compute("Připomeň mi v 18:30 vybrat maso.") is None
    assert compute("Kolik je hodin?") is None
    assert compute("V roce 1818 se narodila.") is None


def test_division_by_zero_declines():
    assert compute("Kolik je 5 děleno 0?") is None


def test_metron_answers_via_iris_card():
    """Automat: metron je přímý expert (jako hodinová otázka Chronos) —
    graf se nedotkne, odpověď nese přepis výrazu kartou."""
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()))
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    out = iris.turn("Kolik je 1 plus 1?")
    assert out.kind == "answer" and "1 + 1 = 2" in out.text
    out = iris.turn("Ověř mi součty - sto, osmdesátdva, třicet, 50, 32. "
                    "Kolik je to dohromady?")
    assert "294" in out.text
