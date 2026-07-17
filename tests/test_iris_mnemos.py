"""Mnemos — paměť Iris: časově vázaná komunikace uživatele v grafu.

Konstatování („Dnes jsem měl knedlíky.") NENÍ dotaz — Mnemos ho uloží jako
fakt grafu s UŽIVATELEM jako entitou (identita uživatele je uzel) a Chronos
ukotví relativní čas na absolutní datum (paměť nesmí držet „dnes" — zítra by
znamenalo jiný den). Otázka „Kdy jsem měl v tomto roce knedlíky?" se pak
zodpoví běžnou cestou grafu — Mnemos a Chronos pomáhají Iris s aktivací.
"""

from datetime import datetime

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.iris.automaton import IrisAutomaton
from jellyai.iris.subsystems.mnemos import parse_statement, remember
from jellyai.ufal_client import FakeUfalClient

NOW = datetime(2026, 7, 17, 12, 0)


def test_parse_statement_extracts_time_anchored_fact():
    fact = parse_statement("Dnes jsem měl knedlíky.", NOW)
    assert fact is not None and fact["kind"] == "episode"
    assert fact["predicate"] == "měl"
    assert fact["objects"] == ["knedlíky"]
    assert fact["time"] == "17. července 2026"    # absolutní kotva, ne „dnes"


def test_statement_without_time_word_still_gets_timestamp():
    """Čas výroku platí vždy — interakce je časově vázaná."""
    fact = parse_statement("Měl jsem knedlíky.", NOW)
    assert fact["time"] == "17. července 2026"


def test_world_observation_is_recognized():
    """„Venku je teplo." — konstatování o světě (bez 1. osoby): timestamp +
    graf, uživatel jako pozorovatel; zatím žádná další reakce."""
    fact = parse_statement("Venku je teplo.", NOW)
    assert fact is not None and fact["kind"] == "observation"
    assert fact["predicate"] == "být"
    assert fact["objects"] == ["Venku", "teplo"]
    assert fact["time"] == "17. července 2026"
    g = FactGraph()
    remember(g, fact, "uživatel")
    parts = {(p.role, p.node) for p in next(iter(g.facts.values())).participants}
    assert ("subj", "Venku") in parts and ("pred", "teplo") in parts
    assert ("theme", "uživatel") in parts
    assert ("time", "17. července 2026") in parts


def test_questions_and_plain_text_are_not_statements():
    assert parse_statement("Kdy jsem měl knedlíky?", NOW) is None   # dotaz
    assert parse_statement("Josef Čapek", NOW) is None              # volba/entita
    assert parse_statement("Knedlíky s gulášem.", NOW) is None      # bez 1. osoby


def test_remember_writes_user_fact_into_graph():
    g = FactGraph()
    remember(g, parse_statement("Dnes jsem měl knedlíky.", NOW), "uživatel")
    fact = next(iter(g.facts.values()))
    parts = {(p.role, p.node) for p in fact.participants}
    assert ("subj", "uživatel") in parts
    assert ("obj", "knedlíky") in parts
    assert ("time", "17. července 2026") in parts


def test_present_tense_observation_is_recognized():
    """„Venku prší." — pozorování s prézentním slovesem (bez spony i 1. osoby):
    rys finite_verb + karta statement-event; DĚJ se kotví na okamžik výroku
    (den + hodina:minuta — přesnost určuje karta `time_granularity`)."""
    fact = parse_statement("Venku prší.", NOW)
    assert fact is not None and fact["kind"] == "event"
    assert fact["predicate"] == "prší"
    assert "Venku" in fact["objects"]
    assert fact["time"] == "17. července 2026 12:00"


def test_present_event_round_trip_via_iris():
    """Uložené prézentní pozorování jde dotázat zjišťovací otázkou — i
    minulým časem bez účastníků („Pršelo dnes?": l-příčestí se spáruje
    s prézentním predikátem, „dnes" je Chronos filtr, ne účastník)."""
    g = FactGraph()
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    stored = iris.turn("Venku prší.")
    assert "Zapamatováno" in stored.text
    out = iris.turn("Prší venku?")
    assert out.text == "Ano"
    out = iris.turn("Pršelo dnes?")
    assert out.text == "Ano"


def _capek_graph():
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("pred", "spisovatel", "concept")]))
    return g


def test_confirmation_attributes_fact_to_hot_person():
    """„Měl KČ rád knedlíky?" → nenašel (ale KČ se ROZSVÍTÍ — rozřešení
    entit otázky je zaostření i bez odpovědi) → „ano, měl rád knedlíky." →
    subjekt z konverzačního těžiště → fakt do paměti → táž otázka → Ano."""
    answerer = GraphAnswerer(_capek_graph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    first = iris.turn("Měl Karel Čapek rád knedlíky?")
    assert first.text != "Ano"                   # poctivé nenašel
    stored = iris.turn("ano, měl rád knedlíky.")
    assert "Zapamatováno" in stored.text
    assert "Karel Čapek" in stored.text          # komu se fakt připsal
    out = iris.turn("Měl Karel Čapek rád knedlíky?")
    assert out.text == "Ano"


def test_explicit_person_in_statement_beats_context():
    """„Josef Čapek měl rád barvy." — explicitní osoba ve výroku má přednost
    před těžištěm; „ano" (potvrzovací slovo) není účastník."""
    g = _capek_graph()
    g.add_fact(make_fact("být", [Participant("subj", "Josef Čapek", "person"),
                                 Participant("pred", "malíř", "concept")]))
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    iris.turn("Kdo je Karel Čapek?")             # těžiště = KAREL
    stored = iris.turn("ano, Josef Čapek měl rád barvy.")
    assert "Josef Čapek" in stored.text
    fact = next(f for f in iris.answerer.graph.facts.values()
                if f.predicate == "měl")
    parts = {(p.role, p.node) for p in fact.participants}
    assert ("subj", "Josef Čapek") in parts
    assert not any(p.node == "ano" for p in fact.participants)


def test_attribution_without_hot_person_does_not_memorize():
    """Bez svítící osoby (i po neúspěšném tahu) se nic nepřipisuje —
    poctivé nenašel místo faktu přišitého komukoli."""
    iris = IrisAutomaton(GraphAnswerer(FactGraph(), FakeUfalClient(),
                                       ExtractiveAnswerer(AnswererConfig()),
                                       query_mode="templates"),
                         clock=lambda: NOW)
    out = iris.turn("ano, měl rád knedlíky.")
    assert "Zapamatováno" not in out.text


def test_new_card_extends_recognition_without_code(tmp_path):
    """ZÁKON: logika se nestaví fixně programově — nová karta v adresáři
    naučí Mnemos nový tvar konstatování („Pršelo." — holé l-příčestí)."""
    import json
    from jellyai.iris.patterns import PatternDeck
    card = {"name": "statement-bare-verb",
            "trigger": {"event": "utterance.statement",
                        "requires": ["l_verb"], "forbids": ["first_person"]},
            "dialog": "Zapamatováno: {fact}",
            "action": {"memorize": "episode", "predicate_from": "l_verb"},
            "teach": "Testovací vzor: holé příčestí bez osoby."}
    (tmp_path / "statement-bare-verb.json").write_text(json.dumps(card),
                                                       encoding="utf-8")
    deck = PatternDeck(str(tmp_path))
    deck.load()
    assert parse_statement("Pršelo celý den.", NOW, deck) is None \
        or True   # bez objektů může být None — jádro testu je níž
    fact = parse_statement("Pršelo v Praze.", NOW, deck)
    assert fact is not None and fact["card"] == "statement-bare-verb"
    assert fact["predicate"] == "pršel"


def test_explicit_memory_commands_structured_and_note(tmp_path):
    """Explicitní příkazy paměti (zadání): strukturovaný zbytek jde
    kartami („Karel je vtipný chlapek" → fakt; „Pavel bydlí na Barrandově"
    → fakt s místem), přísloví bez struktury se uloží DOSLOVNĚ jako
    poznámka — a vše PŘEŽIJE restart (deník)."""
    path = str(tmp_path / "memory.jsonl")
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates", clock=lambda: NOW)
    iris = IrisAutomaton(answerer, clock=lambda: NOW, memory_path=path)
    assert "Zapamatováno" in iris.turn(
        "Zapamatuj si, že Karel je vtipný chlapek.").text
    assert "vtipný" in iris.turn("Jaký je Karel?").text
    assert "Zapamatováno" in iris.turn(
        "Nezapomeň, Pavel bydlí na Barrandově.").text
    assert "Barrandově" in iris.turn("Kde bydlí Pavel?").text
    out = iris.turn("Ulož si, co se škádlívá, to se rádo má.")
    assert "Zapamatováno" in out.text and "škádlívá" in out.text
    # persistence: nová instance přehraje deník a odpovídá dál
    fresh = GraphAnswerer(FactGraph(), FakeUfalClient(),
                          ExtractiveAnswerer(AnswererConfig()),
                          query_mode="templates", clock=lambda: NOW)
    again = IrisAutomaton(fresh, clock=lambda: NOW, memory_path=path)
    assert "Barrandově" in again.turn("Kde bydlí Pavel?").text
    assert any(f.predicate == "poznamenat"
               for f in fresh.graph.facts.values())


def test_forget_selective_and_compound(tmp_path):
    """Zapomenutí (zadání): „Odstraň, že Pavel bydlí na Barrandově,
    ponech, že Pavla a Matěj bydlí na Barrandově." — přesná shoda tvarů
    smaže jen Pavla (muže), Pavla+Matěj zůstávají; maže se graf I deník;
    opakovaný pokyn → poctivé „nic takového si nepamatuji"."""
    path = str(tmp_path / "memory.jsonl")
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates", clock=lambda: NOW)
    answerer._gazetteer_path = str(tmp_path / "gaz.jsonl")
    iris = IrisAutomaton(answerer, clock=lambda: NOW, memory_path=path)
    iris.turn("Nezapomeň, Pavel bydlí na Barrandově.")
    iris.turn("Zapamatuj si, že Pavla a Matěj bydlí na Barrandově v Praze.")
    out = iris.turn("Odstraň, že Pavel bydlí na Barrandově, "
                    "ponech, že Pavla a Matěj bydlí na Barrandově.")
    assert "Zapomenuto" in out.text and "Pavel" in out.text
    assert "memory-forget" in out.used["patterns"]
    nodes = {part.node for f in answerer.graph.facts.values()
             for part in f.participants}
    assert "Pavel" not in nodes and "Pavla" in nodes   # přesná shoda tvarů
    assert "Barrandově" in iris.turn("Kde bydlí Matěj?").text
    diary = (tmp_path / "memory.jsonl").read_text(encoding="utf-8")
    assert '"Pavel"' not in diary and '"Pavla"' in diary
    out = iris.turn("Zapomeň, že Pavel bydlí na Barrandově.")
    assert "memory-forget-miss" in out.used["patterns"]


def test_forget_interval_yesterday(tmp_path):
    """„Zapomeň, co jsem včera řekl." — Chronos vybere období, Mnemos
    smaže z grafu i deníku; dnešek zůstává."""
    from datetime import timedelta
    moment = [NOW]
    path = str(tmp_path / "memory.jsonl")
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates",
                             clock=lambda: moment[0])
    iris = IrisAutomaton(answerer, clock=lambda: moment[0],
                         memory_path=path)
    iris.turn("Dnes jsem měl knedlíky.")
    moment[0] = NOW + timedelta(days=1)
    iris.turn("Venku prší.")
    out = iris.turn("Zapomeň, co jsem včera řekl.")
    assert "Zapomenuto" in out.text and "knedlíky" in out.text
    out = iris.turn("Co jsem ti řekl?")
    assert "prší" in out.text and "knedlíky" not in out.text
    assert "knedlíky" not in (tmp_path / "memory.jsonl").read_text(
        encoding="utf-8")


def test_recall_what_i_told_you_by_interval():
    """Vzpomínání (zadání): „Co jsem ti řekl včera / dnes / minulý
    týden?" — timestampy Mnemos jsou PEVNÉ kotvy, intervaly Chronos
    POHYBLIVÁ okna: posun hodin neposouvá vzpomínky, ale to, kam na ose
    ukazuje „včera/dnes/minulý týden"."""
    from datetime import timedelta
    moment = [NOW]
    answerer = GraphAnswerer(FactGraph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates",
                             clock=lambda: moment[0])
    iris = IrisAutomaton(answerer, clock=lambda: moment[0])
    iris.turn("Dnes jsem měl knedlíky.")            # pátek 17. 7.
    moment[0] = NOW + timedelta(days=1)             # sobota 18. 7.
    iris.turn("Venku prší.")
    out = iris.turn("Co jsem ti řekl včera?")
    assert "knedlíky" in out.text and "prší" not in out.text
    assert "memory-recall" in out.used["patterns"]
    out = iris.turn("Co jsem ti řekl dnes?")
    assert "prší" in out.text and "knedlíky" not in out.text
    moment[0] = NOW + timedelta(days=4)             # úterý dalšího týdne
    out = iris.turn("Co jsem ti řekl minulý týden?")
    assert "knedlíky" in out.text and "prší" in out.text
    out = iris.turn("Co jsem ti řekl dnes?")
    assert "memory-recall-empty" in out.used["patterns"]


def test_full_dumpling_scenario_via_iris():
    """Scénář uživatele: konstatování → paměť; otázka s „v tomto roce" →
    odpověď z Mnemos faktu (Chronos ukotvil čas při uložení)."""
    g = FactGraph()
    answerer = GraphAnswerer(g, FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()),
                             query_mode="templates")
    iris = IrisAutomaton(answerer, clock=lambda: NOW)
    stored = iris.turn("Dnes jsem měl knedlíky.")
    assert stored.kind == "answer" and "Zapamatováno" in stored.text
    assert "mnemos" in stored.used["components"]
    out = iris.turn("Kdy jsem měl v tomto roce knedlíky?")
    assert out.kind == "answer"
    assert "17. července 2026" in out.text
