"""Příkaz „Pošli…" — adresát ≠ čas ≠ zpráva (BACKLOG #54, z produkce).

„Pošli zítra mail Marci" bral „zítra" jako jméno adresáta. Nárok na
časové tokeny má CHRONOS (claim_words — zárodek formálního claim()
z #26 S2), zprávová jména (mail/zprávu/upozornění) drží tabulka
message_nouns; adresát je první NEnárokovaný token a musí vypadat
jako jméno (kapitalizace v původním textu). Skloněné jméno („Marci",
„Marcele") se na osobu s e-mailem rozřeší kmenovým prefixem.
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
EMAIL = "marcela.duffkova@seznam.cz"


def _iris():
    g = FactGraph()
    g.add_fact(make_fact("email", [
        Participant("subj", "Marcela", "person"),
        Participant("obj", EMAIL, "email"),
        Participant("theme", "uživatel", "person")]))
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()))
    return IrisAutomaton(answerer, clock=lambda: NOW)


def test_send_skips_chronos_claim_and_message_noun():
    """„Pošli zítra mail Marcele…" — „zítra" nárokuje Chronos, „mail"
    je zpráva; adresátem je Marcele (jméno s e-mailem v grafu)."""
    out = _iris().turn("Pošli zítra mail Marcele, že ji pozdravuji.")
    assert out is not None
    assert "zitra" not in out.text and "zítra" not in out.text.split()[1:2]
    assert "Pošlu Marcele" in out.text


def test_send_resolves_declined_recipient_stem():
    """Dativ „Marci" → Marcela kmenovým prefixem (marc ⊂ marcel)."""
    out = _iris().turn("Pošli zítra mail Marci, že ji pozdravuji.")
    assert out is not None
    assert "Pošlu Marci" in out.text
    assert "Neznám e-mail" not in out.text


def test_send_without_name_keeps_default_channel():
    """„Pošli mi zítra upozornění…" — zájmeno = bez adresáta (default);
    malopísmenné slovo úkolu se adresátem stát nesmí."""
    iris = _iris()
    out = iris.turn("Pošli mi zítra upozornění vezmi léky.")
    assert out is not None
    assert "Neznám e-mail" not in out.text
    assert "Pošlu" not in out.text          # bez adresáta → „Připomenu…"


def test_send_unknown_name_asks_for_email():
    """Kapitalizované jméno bez e-mailu v grafu → upřímná výzva."""
    out = _iris().turn("Pošli zítra mail Bedřichovi, že ho zdravím.")
    assert out is not None
    assert "Neznám e-mail" in out.text
