"""Automat Iris — tah uživatele → odpověď, NEBO dialogové doostření.

Nad prahem jistoty (QueryAssurance) automat odpoví; pod prahem vybere
pattern-kartu a vede dialog: nabídne kandidáty, rozsvítí je, a volbu
uživatele přehraje jako zaostřenou otázku. Bez odpovědi s kandidáty
zazní upřímný terminál („nepodařilo se mi zaostřit…").
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.iris.automaton import IrisAutomaton
from jellyai.ufal_client import FakeUfalClient


def _brothers_graph():
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("pred", "spisovatel", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Josef Čapek", "person"),
                                 Participant("pred", "malíř", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    return g


def _iris(graph):
    answerer = GraphAnswerer(graph, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    return IrisAutomaton(answerer)


def test_confident_question_answers_directly():
    iris = _iris(_brothers_graph())
    out = iris.turn("Kdo napsal R.U.R.?")
    assert out.kind == "answer" and "Karel Čapek" in out.text
    assert out.assurance >= 0.6
    jas = [score for _, score in out.activation_window]
    assert jas == sorted(jas, reverse=True)      # okno seřazené sestupně
    assert "graph-answerer" in out.used["components"]


def test_homonym_offers_focus_dialog():
    """„Kdo je Čapek?" — dva rovnocenní kandidáti → dialog z karty, ne tiché
    rozhodnutí vahou; kandidáti se rozsvítí (aktivace > 0)."""
    iris = _iris(_brothers_graph())
    out = iris.turn("Kdo je Čapek?")
    assert out.kind == "dialog"
    assert "Karel Čapek" in out.text and "Josef Čapek" in out.text
    assert "focus-offer-homonym" in out.used["patterns"]
    scores = iris.answerer.context.scores
    assert scores.get("Karel Čapek", 0) > 0 and scores.get("Josef Čapek", 0) > 0


def test_user_pick_resumes_and_answers():
    """Volba kandidáta přehraje původní otázku zaostřenou na vybraného."""
    iris = _iris(_brothers_graph())
    first = iris.turn("Kdo je Čapek?")
    assert first.kind == "dialog"
    out = iris.turn("Josef Čapek")
    assert out.kind == "answer" and "malíř" in out.text


def test_unanswerable_gets_honest_terminal():
    """Rozlišený uzel bez odpovědního faktu → upřímný terminál s kandidáty
    (žádné hádání), kandidát se rozsvítí pro další tah."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Pavel Novák", "person"),
                                    Participant("obj", "kniha", "concept")]))
    iris = _iris(g)
    out = iris.turn("Kdo je Novák?")
    assert out.kind == "dialog"
    assert "Nepodařilo" in out.text and "Pavel Novák" in out.text
    assert "assurance-fail" in out.used["patterns"]


def test_empty_deck_means_mechanism_only():
    """ZÁKON: rozhodnutí nesou karty — bez karet automat vždy odpovídá
    (žádný dialog není zadrátovaný v kódu)."""
    from jellyai.iris.patterns import PatternDeck
    answerer = GraphAnswerer(_brothers_graph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    empty = PatternDeck.__new__(PatternDeck)
    empty.directory, empty.cards = None, []
    iris = IrisAutomaton(answerer, deck=empty)
    out = iris.turn("Kdo je Čapek?")      # homonymum, ale bez karty → odpověď
    assert out.kind == "answer"


def test_focus_shift_warms_domain_and_replays():
    """„v kontextu Bible" není otázka ani konstatování — pokyn k zaostření:
    karta posvítí na doménu a přehraje předchozí otázku (spec §3e)."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Maria", "person"),
                                 Participant("pred", "postava", "concept")]))
    g.add_fact(make_fact("kontext", [Participant("subj", "Maria", "person"),
                                     Participant("obj", "Bible", "dílo")]))
    iris = _iris(g)
    iris.turn("Kdo je Maria?")
    out = iris.turn("v kontextu Bible")
    assert "focus-shift" in out.used["patterns"]
    assert iris.answerer.context.scores.get("Bible", 0) > 0   # doména svítí
    assert "postava" in out.text      # přehraná otázka odpověděla


def _sayings_graph():
    """Sedm výroků Ježíše s různými adresáty — víc, než enumerace unese."""
    g = FactGraph()
    for i, theme in enumerate(["Janovi", "Lukášovi", "Magdaléně", "učedníkům",
                               "Petrovi", "celníkovi", "farizeům"]):
        g.add_fact(make_fact("říci", [
            Participant("subj", "Ježíš", "person"),
            Participant("obj", f"výrok číslo {i}", "výrok"),
            Participant("theme", theme, "person")]))
    return g


def test_overflow_offers_activation_areas():
    """„Co řekl Ježíš?" nad přetékajícím výčtem → karta data.overflow nabídne
    OBLASTI aktivace (adresáty), ne jen ořezaný výčet."""
    iris = _iris(_sayings_graph())
    out = iris.turn("Co řekl Ježíš?")
    assert out.kind == "dialog"
    assert "focus-offer-overflow" in out.used["patterns"]
    assert "učedníkům" in out.text


def test_overflow_pick_narrows_by_activation():
    """Volba oblasti („učedníkům") rozsvítí adresáta a přehraná otázka
    vybere výrok z jeho faktu — zúžení čistě aktivací."""
    iris = _iris(_sayings_graph())
    iris.turn("Co řekl Ježíš?")
    out = iris.turn("učedníkům")
    assert out.kind == "answer"
    assert "výrok číslo 3" in out.text        # výrok patřící učedníkům


def test_card_telemetry_counts_usage_and_gain():
    """Telemetrie karet: použití + měřený nárůst aktivace kandidátů (zisk)."""
    iris = _iris(_brothers_graph())
    iris.turn("Kdo je Čapek?")
    telemetry = iris.telemetry["focus-offer-homonym"]
    assert telemetry["used"] == 1
    assert telemetry["gain"] > 0


def test_plain_miss_keeps_fallback_text():
    """Nic rozlišeného (parser None) → poctivé „nenašel" fallbacku beze změny."""
    iris = _iris(_brothers_graph())
    out = iris.turn("Kdo je Xylofonius Neznámý?")
    assert "nenašel" in out.text
