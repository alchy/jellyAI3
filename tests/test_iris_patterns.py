"""Pattern-karty Iris — chování automatu jako data (1 JSON = 1 vzor).

Karta nese trigger (KDY se použije), dialog (CO říct), akci (CO s aktivací)
a teach (výukové vysvětlení). Přidání karty = rozšíření schopností automatu
bez zásahu do kódu; jiný jazyk = jiný adresář karet.
"""

import json
import pytest

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


def test_best_prefers_specific_card_over_priority(tmp_path):
    """Benefit-výběr (spec §2.6b): těsnější trigger (víc splněných podmínek)
    přebije obecnou kartu i s vyšší prioritou; match() zůstává first-match."""
    import json
    generic = {"name": "obecna", "trigger": {"event": "e", "priority": 99},
               "dialog": "obecně", "action": {}, "teach": ""}
    tight = {"name": "tesna",
             "trigger": {"event": "e", "priority": 1, "min_candidates": 2,
                         "assurance_below": 0.5, "requires": ["rys"]},
             "dialog": "těsně", "action": {}, "teach": ""}
    for card in (generic, tight):
        (tmp_path / f"{card['name']}.json").write_text(
            json.dumps(card), encoding="utf-8")
    deck = PatternDeck(str(tmp_path))
    deck.load()
    context = {"assurance": 0.2, "candidates": ["A", "B"],
               "features": {"rys"}}
    assert deck.best("e", context).name == "tesna"
    assert deck.best("e", {"assurance": 0.9}).name == "obecna"


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


def _familia():
    """Zkušební rodinná karta (kostra × dimenze čas a elipse) — #57 E1."""
    return {
        "name": "q-pokus",
        "trigger": {
            "event": "utterance.query",
            "pattern": ["%{TAZACI}", "<sloveso>", "<ucastnik>"],
            "dimensions": [
                {"slot": "<sloveso>", "values": [
                    {"suffix": "-minuly", "element": "%{SLOVESO_MINULE}"},
                    {"suffix": "-prezens", "element": "%{SLOVESO}"}]},
                {"slot": "<ucastnik>", "values": [
                    {"suffix": "", "element": "%{ENTITA}", "priority": 4},
                    {"suffix": "-prodrop", "element": None, "priority": 3}]},
            ],
        },
        "action": {"query": {"hole": "$1", "predicate": "$2",
                             "known": ["$3"]}},
        "teach": "Zkušební rodina.",
    }


def test_expand_family_rozvine_kostru_krat_dimenze():
    from jellyai.iris.patterns import _expand_family
    cards = _expand_family(_familia())
    by_name = {card["name"]: card for card in cards}
    assert set(by_name) == {"q-pokus-minuly", "q-pokus-prezens",
                            "q-pokus-minuly-prodrop",
                            "q-pokus-prezens-prodrop"}
    plna = by_name["q-pokus-minuly"]
    assert plna["trigger"]["pattern"] == [
        "%{TAZACI}", "%{SLOVESO_MINULE}", "%{ENTITA}"]
    assert plna["trigger"]["priority"] == 4
    assert plna["action"]["query"]["known"] == ["$3"]
    assert "dimensions" not in plna["trigger"]
    prodrop = by_name["q-pokus-prezens-prodrop"]
    assert prodrop["trigger"]["pattern"] == ["%{TAZACI}", "%{SLOVESO}"]
    assert prodrop["trigger"]["priority"] == 3
    assert "known" not in prodrop["action"]["query"]


def test_expand_family_prazdny_slot_jen_na_konci():
    from jellyai.iris.patterns import _expand_family
    data = _familia()
    data["trigger"]["pattern"] = ["<ucastnik>", "%{TAZACI}", "<sloveso>"]
    with pytest.raises(ValueError):
        _expand_family(data)


def test_expand_family_dira_nesmi_mirit_na_prazdny_slot():
    from jellyai.iris.patterns import _expand_family
    data = _familia()
    data["action"]["query"]["predicate"] = "$3"
    with pytest.raises(ValueError):
        _expand_family(data)
