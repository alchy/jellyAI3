"""Pattern-karty Iris — chování automatu jako data (1 JSON = 1 vzor).

Karta nese trigger (KDY se použije), dialog (CO říct), akci (CO s aktivací)
a teach (výukové vysvětlení). Přidání karty = rozšíření schopností automatu
bez zásahu do kódu; jiný jazyk = jiný adresář karet.
"""

import json
import os
import pytest

from jellyai.iris.patterns import PatternDeck


def test_deck_loads_cards_from_directory():
    deck = PatternDeck.for_language("cs")
    assert deck.load() >= 3          # focus-offer, assurance-fail, … (resolve-miss zakonzervována — mrtvý event)
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
    # syntetický event „e" testuje mechaniku výběru — lint vypnut
    deck = PatternDeck(str(tmp_path), known_events=None)
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
    # syntetický event „data.empty" — mechanika, lint vypnut
    deck = PatternDeck(str(tmp_path), known_events=None)
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


def test_deck_rozvine_rodinu_q_otaz():
    deck = PatternDeck.for_language("cs")
    deck.load()
    cards = {card.name: card for card in deck.cards}
    assert cards["q-otaz-minuly"].trigger["pattern"] == [
        "%{TAZACI}", "?%{SE}", "%{SLOVESO_MINULE}", "%{ENTITA}"]
    assert cards["q-otaz-minuly"].trigger["priority"] == 4
    assert cards["q-otaz-minuly-prodrop"].trigger["pattern"] == [
        "%{TAZACI}", "?%{SE}", "%{SLOVESO_MINULE}"]
    assert cards["q-otaz-minuly-prodrop"].trigger["priority"] == 3
    assert cards["q-otaz-minuly-prodrop"].action["query"] == {
        "hole": "$1", "predicate": "$3"}
    # NOVÁ kombinace z rozvinutí — ručně nikdy nenapsaná:
    assert cards["q-otaz-prezens-prodrop"].trigger["pattern"] == [
        "%{TAZACI}", "?%{SE}", "%{SLOVESO}"]
    assert cards["q-otaz-prezens-prodrop"].trigger["priority"] == 3


def test_deck_rozvine_rodinu_q_zjistovaci():
    deck = PatternDeck.for_language("cs")
    deck.load()
    cards = {card.name: card for card in deck.cards}
    for jmeno, sloveso in (("q-zjistovaci-minuly", "%{SLOVESO_MINULE}"),
                           ("q-zjistovaci-prezens", "%{SLOVESO}")):
        assert cards[jmeno].trigger["pattern"] == [
            sloveso, "?%{ENTITA}", "?%{ENTITA}"]
        assert cards[jmeno].trigger["priority"] == 3
        assert cards[jmeno].action["query"]["known"] == [
            ["subj", "$2"], ["obj", "$3"]]
    assert not os.path.exists(os.path.join(
        PatternDeck.for_language("cs").directory,
        "q-zjistovaci-minuly.json"))


def test_deck_odmita_neznamy_event(tmp_path):
    """Lint karet (postřeh 2.5): překlep eventu = karta, která se nikdy
    nevybere, tiše — load musí spadnout NAHLAS (vzor: grok-zkratky)."""
    (tmp_path / "typo.json").write_text(json.dumps({
        "name": "typo-karta",
        "trigger": {"event": "utterance.qeury"},
        "dialog": "", "action": {}, "teach": "překlep"}), encoding="utf-8")
    with pytest.raises(ValueError, match="utterance.qeury"):
        PatternDeck(str(tmp_path)).load()


def test_deck_odmita_neznamy_klic_query_akce(tmp_path):
    """Lint karet (postřeh 3.3): neznámý klíč akce dotazové karty
    (překlep „hole_rol") se dnes tiše ignoruje — load musí spadnout."""
    (tmp_path / "q-typo.json").write_text(json.dumps({
        "name": "q-typo",
        "trigger": {"event": "utterance.query",
                    "pattern": ["%{TAZACI}"], "priority": 1},
        "dialog": "", "action": {"query": {"hole_rol": "subj"}},
        "teach": "překlep"}), encoding="utf-8")
    with pytest.raises(ValueError, match="hole_rol"):
        PatternDeck(str(tmp_path)).load()


def test_zivy_deck_projde_lintem():
    """Všechny skutečné karty používají známé eventy i klíče akcí."""
    deck = PatternDeck.for_language("cs")
    assert deck.load() > 0
