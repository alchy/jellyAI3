"""Pattern-karty Iris — chování automatu jako data (1 JSON = 1 vzor).

Karta nese trigger (KDY se použije), dialog (CO říct), akci (CO s aktivací)
a teach (výukové vysvětlení). Přidání karty = rozšíření schopností automatu
bez zásahu do kódu; jiný jazyk = jiný adresář karet.
"""

import json

from jellyai.iris.patterns import PatternDeck


def test_deck_loads_cards_from_directory():
    deck = PatternDeck.for_language("cs")
    assert deck.load() >= 3          # focus-offer, resolve-miss, assurance-fail
    names = {card.name for card in deck.cards}
    assert "focus-offer-homonym" in names


def test_match_ambiguous_with_low_assurance():
    deck = PatternDeck.for_language("cs")
    deck.load()
    card = deck.match("resolve.ambiguous",
                      {"assurance": 0.4, "candidates": ["Karel Čapek",
                                                        "Josef Čapek"]})
    assert card is not None and card.name == "focus-offer-homonym"
    assert "{candidates}" in card.dialog


def test_match_respects_assurance_threshold():
    deck = PatternDeck.for_language("cs")
    deck.load()
    assert deck.match("resolve.ambiguous",
                      {"assurance": 0.9, "candidates": ["A", "B"]}) is None


def test_match_unknown_event_returns_none():
    deck = PatternDeck.for_language("cs")
    deck.load()
    assert deck.match("neexistujici.udalost", {"assurance": 0.0}) is None


def test_custom_directory_extends_behavior(tmp_path):
    """Nový jazyk/chování = nový adresář karet, žádná změna kódu."""
    card = {"name": "test-vzor", "trigger": {"event": "data.empty"},
            "dialog": "Nemám data pro {term}.", "action": {},
            "teach": "Testovací vzor."}
    (tmp_path / "test-vzor.json").write_text(json.dumps(card),
                                             encoding="utf-8")
    deck = PatternDeck(str(tmp_path))
    assert deck.load() == 1
    assert deck.match("data.empty", {}).dialog == "Nemám data pro {term}."
