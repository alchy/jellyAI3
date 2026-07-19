"""Vzorové karty dotazů (#46 fáze 2a) — zavírají #44 z živého dialogu.

„Kdo bydlí v Petrovicích?" dosud neměl šablonu (build_query → None) a
fakty paměti (osoba jako obj, místo loc, uživatel theme) nenašel ani
UDPipe fallback. Vzorová karta `q-kdo-sloveso-misto` staví Pattern
z tříd lexeru; answerer navíc dostal guardy: pozorovatel (uživatel
v roli theme) ani časová kotva nejsou odpověď na kdo/co díru.
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def _bydli(person, place):
    return make_fact("bydlí", [
        Participant("obj", person, "concept"),
        Participant("loc", place, "geo"),
        Participant("theme", "uživatel", "person"),
        Participant("time", "17. července 2026 12:00", "time")])


def _answerer(*facts):
    g = FactGraph()
    for fact in facts:
        g.add_fact(fact)
    return GraphAnswerer(g, FakeUfalClient(),
                         ExtractiveAnswerer(AnswererConfig()),
                         clock=lambda: NOW)


def test_kdo_verb_place_finds_inhabitants_not_user():
    a = _answerer(_bydli("Marcela", "Petrovice"), _bydli("Roník", "Petrovice"))
    text = a.answer("Kdo bydlí v Petrovicích?", []).text
    assert "Marcela" in text and "Roník" in text
    assert "uživatel" not in text         # pozorovatel není odpověď (#34)
    assert "července" not in text         # časová kotva není odpověď


def test_kdo_dalsi_variant_matches_same_pattern():
    a = _answerer(_bydli("Marcela", "Petrovice"))
    text = a.answer("Kdo další bydlí v Petrovicích?", []).text
    assert "Marcela" in text


def test_cim_instrument_question():
    a = _answerer(make_fact("krmí", [
        Participant("obj", "Marcela", "concept"),
        Participant("obj", "Roníka", "concept"),
        Participant("obj", "konzervami", "concept"),
        Participant("theme", "uživatel", "person"),
        Participant("time", "17. července 2026 12:00", "time")]))
    text = a.answer("Čím krmí Marcela Roníka?", []).text
    assert "konzervami" in text
    assert "uživatel" not in text and "července" not in text


def test_place_hole_still_works_after_guards():
    """Guardy nesmí rozbít směr „Kde bydlí X?" (loc díra z týchž faktů)."""
    a = _answerer(_bydli("Marcela", "Petrovice"))
    assert a.answer("Kde bydlí Marcela?", []).text == "Petrovice"


def test_existence_cards_build_pattern_via_card_path():
    """Fáze 2b: zjišťovací otázky jako karty — span prvek `uzel+` dělí
    víceslovné entity orákulem grafu; bez účastníků = holá existence."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Napsal Karel Čapek Válku s mloky?", {"napsat"},
                    is_node=lambda s: s in ("Karel Čapek", "Válku s mloky"),
                    is_word=None)
    assert q is not None
    assert q.pattern.predicate == "napsat"      # normalizace _verb_match
    assert q.pattern.known == [("subj", "Karel Čapek"),
                               ("obj", "Válku s mloky")]
    assert q.pattern.hole_role is None and q.qtype is None

    bare = _card_query("Prší?", {"prší"},
                       is_node=lambda s: False, is_word=None)
    assert bare is not None
    assert bare.pattern.predicate == "prší"
    assert bare.pattern.known == []             # holá existence


def test_hole_question_cards_build_pattern():
    """Fáze 2c: „OTAZ (se) SLOVESO ENTITA?" jako karta — díra z interrogatives,
    predikát normalizovaný, entita spanem. Ukotvení chrání složené otázky
    (vztažné věty, „Jakou hru…") — ty jdou dál poziční šablonou."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Kdo napsal R.U.R.?", {"napsat"},
                    is_node=lambda s: s == "R.U.R.", is_word=None)
    assert q is not None
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role == "subj" and q.pattern.hole_type == "person"
    assert q.pattern.known == [("obj", "R.U.R.")]
    assert q.qtype == "Kdo"

    q = _card_query("Kde se narodil Karel Čapek?", {"narodit"},
                    is_node=lambda s: s == "Karel Čapek", is_word=None)
    assert q is not None
    assert q.pattern.hole_role == "loc"
    assert q.pattern.known == [("subj", "Karel Čapek")]

    složená = _card_query("Kdo byl bratr autora, který napsal R.U.R.?",
                          {"napsat"}, is_node=lambda s: s == "R.U.R.",
                          is_word=None)
    assert složená is None       # rekurze zůstává poziční šabloně


def test_selection_question_card_builds_typed_pattern():
    """Fáze 2d: výběrová otázka „Jakou hru napsal Karel Čapek?" jako karta —
    typový filtr díry (hru) je obj, téma subj (join answererova
    _typed_match); elidované téma („Jakou hru napsal?") doplní kontext."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Jakou hru napsal Karel Čapek?", {"napsat"},
                    is_node=lambda s: s in ("hru", "Karel Čapek"),
                    is_word=None)
    assert q is not None
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role == "attr" and q.pattern.hole_type is None
    assert q.pattern.known == [("obj", "hru"), ("subj", "Karel Čapek")]
    assert q.qtype == "Jaký"

    elided = _card_query("Jakou hru napsal?", {"napsat"},
                         is_node=lambda s: s == "hru", is_word=None)
    assert elided is not None
    assert elided.pattern.known == [("obj", "hru")]


def test_relation_operator_card_builds_relation_pattern():
    """Fáze 2d: „Jaký měl X vztah k Y?" jako karta — „vztah" není entita,
    ale OPERÁTOR spojení (třída vztah_dotazu z relation_query_nouns):
    sloveso otázky nic nenese (vzor ho přeskočí), oba účastníci jsou obj,
    díra relation bez predikátu. Výběrový vzor otázku nekrade, i když se
    „měl" orákulem rozřeší na šumový uzel (past 9–11)."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Jaký měl Ježíš vztah k Janu Křtiteli?", {"poslat"},
                    is_node=lambda s: s in ("měl", "vztah", "Ježíš",
                                            "Janu Křtiteli"),
                    is_word=None)
    assert q is not None
    assert q.pattern.predicate is None
    assert q.pattern.hole_role == "relation" and q.pattern.hole_type is None
    assert q.pattern.known == [("obj", "Ježíš"), ("obj", "Janu Křtiteli")]
    assert q.qtype == "Jaký"


def test_first_person_card_maps_user_subject():
    """Fáze 2d: „Kdy jsem měl v tomto roce knedlíky?" — 1. osoba: podmětem
    je IDENTITA UŽIVATELE (uzel Mnemos), jmenovaná entita je předmět;
    časová slova nejsou účastníci (filtr intervalu drží Chronos)."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Kdy jsem měl v tomto roce knedlíky?", {"měl"},
                    is_node=lambda s: s == "knedlíky", is_word=None)
    assert q is not None
    assert q.pattern.predicate == "měl"
    assert q.pattern.hole_role == "time" and q.pattern.hole_type == "time"
    assert q.pattern.known == [("obj", "knedlíky"), ("subj", "uživatel")]
    assert q.qtype == "Kdy"
    assert q.gender == "Masc"

    bez_filtru = _card_query("Kdy jsem měl knedlíky?", {"měl"},
                             is_node=lambda s: s == "knedlíky", is_word=None)
    assert bez_filtru is not None
    assert bez_filtru.pattern.known == [("obj", "knedlíky"),
                                        ("subj", "uživatel")]


def test_copular_identity_card():
    """Fáze 2d: sponová identita „Kdo je X?" / „Jaký je X?" jako karta —
    predikát „být" nese karta LITERÁLEM (data, ne kód), attr díra zůstává
    attr, každá jiná se překlápí na pred (kanonické sponové pravidlo);
    rod nese tvar spony (byla → Fem), is_copula pro odpovědní vrstvu."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Kdo je robot?", set(),
                    is_node=lambda s: s == "robot", is_word=None)
    assert q is not None
    assert q.pattern.predicate == "být"
    assert q.pattern.hole_role == "pred" and q.pattern.hole_type == "person"
    assert q.pattern.known == [("subj", "robot")]
    assert q.qtype == "Kdo" and q.is_copula and q.gender is None

    jaky = _card_query("Jaký je robot?", set(),
                       is_node=lambda s: s == "robot", is_word=None)
    assert jaky is not None
    assert jaky.pattern.hole_role == "attr" and jaky.pattern.hole_type is None
    assert jaky.qtype == "Jaký"

    byla = _card_query("Kdo byla Božena Němcová?", set(),
                       is_node=lambda s: s == "Božena Němcová", is_word=None)
    assert byla is not None
    assert byla.pattern.known == [("subj", "Božena Němcová")]
    assert byla.gender == "Fem" and byla.is_copula


def test_copular_relational_noun_card():
    """Fáze 2d: „Kdo byl bratr Karla Čapka?" jako karta — vztažné jméno
    je predikát (bratr), entita obj, díra subj/person; rod z tvaru spony.
    Rekurze („bratr autora, který napsal…") zůstává poziční šabloně —
    karty nevnořují (spec §7)."""
    from jellyai.answerer.query import _card_query

    q = _card_query("Kdo byl bratr Karla Čapka?", {"bratr"},
                    is_node=lambda s: s == "Karla Čapka", is_word=None)
    assert q is not None
    assert q.pattern.predicate == "bratr"
    assert q.pattern.hole_role == "subj" and q.pattern.hole_type == "person"
    assert q.pattern.known == [("obj", "Karla Čapka")]
    assert q.qtype == "Kdo" and q.is_copula and q.gender == "Masc"

    rekurze = _card_query("Kdo byl bratr autora, který napsal R.U.R.?",
                          {"bratr", "napsat"},
                          is_node=lambda s: s in ("autora", "R.U.R."),
                          is_word=None)
    assert rekurze is None       # SubQuery zůstává poziční šabloně


def test_date_drill_card_builds_time_pattern():
    """Fáze 2d: 2-skokový drill „V kterém roce se narodila BN?" jako karta —
    část data z tabulky date_part_forms (třída cast_data), attr díra se
    s částí data překlápí na time/time; „v TOMTO roce" drill není (tomto
    nemá třídu otaz — filtr intervalu drží Chronos)."""
    from jellyai.answerer.query import _card_query

    q = _card_query("V kterém roce se narodila Božena Němcová?", {"narodit"},
                    is_node=lambda s: s == "Božena Němcová", is_word=None)
    assert q is not None
    assert q.pattern.predicate == "narodit"
    assert q.pattern.hole_role == "time" and q.pattern.hole_type == "time"
    assert q.pattern.date_part == "rok"
    assert q.pattern.known == [("subj", "Božena Němcová")]
    assert q.qtype == "Který"
    assert q.gender == "Fem"

    filtr = _card_query("V tomto roce se narodila Božena Němcová?",
                        {"narodit"},
                        is_node=lambda s: s == "Božena Němcová", is_word=None)
    assert filtr is None         # časový filtr, ne drill — zůstává šabloně
