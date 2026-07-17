"""Testy detailu uzlu — řádky (popisek, hodnota) čistě z grafu, bez viewBase."""

from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.viz.detail import node_detail_rows, fact_detail_rows


def _graph():
    """Malý graf: Božena napsala Babičku a narodila se 1820."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("time", "1820", "time")]))
    return g


def test_entity_rows_type_weight_and_facts():
    rows = dict(node_detail_rows(_graph(), "Božena Němcová"))
    assert rows["typ"] == "osoba"
    assert rows["váha (výskyty v textu)"] == "2"   # figuruje ve 2 faktech
    assert rows["aktivace (attention)"] == "0"     # živě přepisuje web
    assert rows["napsat →"] == "Babička"
    assert rows["narodit →"] == "1820"


def test_object_node_shows_incoming_direction():
    rows = dict(node_detail_rows(_graph(), "Babička"))
    assert rows["typ"] == "pojem"
    assert rows["napsat ←"] == "Božena Němcová"   # Babička je předmět → šipka dovnitř


def test_same_predicate_aggregates_partners():
    g = FactGraph()
    for dilo in ("Babička", "Divá Bára"):
        g.add_fact(make_fact("napsat", [Participant("subj", "B. N.", "person"),
                                        Participant("obj", dilo, "concept")]))
    rows = dict(node_detail_rows(g, "B. N."))
    assert rows["napsat →"] == "Babička, Divá Bára"   # jeden řádek, spojené hodnoty


def test_unknown_node_is_graceful():
    rows = dict(node_detail_rows(_graph(), "Neexistuje"))
    assert rows == {"uzel": "Neexistuje"}


def test_caps_to_five_most_frequent_connections():
    """Uzel s mnoha spojeními → typ, váha + jen 5 nejfrekventovanějších."""
    g = FactGraph()
    # 7 různých predikátů; „psát" opakováno 3x → nejsilnější spojení
    for i in range(7):
        for _ in range(3 if i == 0 else 1):
            g.add_fact(make_fact(f"pred{i}",
                                 [Participant("subj", "Autor", "person"),
                                  Participant("obj", f"cíl{i}", "concept")]))
    rows = node_detail_rows(g, "Autor")
    labels = [k for k, _ in rows]
    assert labels[0] == "typ"
    # typ, váha, aktivace + morfologie osoby (rod, kmen) + top 5 spojení
    assert len(rows) == 3 + 2 + 5
    assert "pred0 →" in labels                   # nejčastější (3x) se vejde
    assert dict(rows)["pred0 →"] == "cíl0"


def test_caps_partners_per_row():
    """Jeden predikát s mnoha partnery → řádek se ořízne a doplní '…'."""
    g = FactGraph()
    for i in range(8):
        g.add_fact(make_fact("mít", [Participant("subj", "X", "person"),
                                     Participant("obj", f"věc{i}", "concept")]))
    value = dict(node_detail_rows(g, "X"))["mít →"]
    assert value.endswith("…")
    assert value.count(",") < 8                  # ne všech 8 partnerů


def test_fact_rows_predicate_weight_participants():
    g = _graph()
    fact = next(f for f in g.facts.values() if f.predicate == "napsat")
    rows = dict(fact_detail_rows(fact))
    assert rows["typ"] == "fakt"
    assert rows["predikát"] == "napsat"
    assert rows["váha"] == "1"
    assert rows["subj"] == "Božena Němcová"
    assert rows["obj"] == "Babička"


def test_person_rows_carry_morphology():
    """Detail osoby nese morfologii: rod (z tvaru jména), kmenový klíč a
    sloučené pádové tvary z resolveru (pokyn: rysy v popisu uzlů vizualizace)."""
    from jellyai.graph.graph import resolve_entities
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.add_fact(make_fact("číst", [Participant("subj", "lid", "concept"),
                                  Participant("obj", "Boženy Němcové", "person")]))
    resolve_entities(g)
    rows = dict(node_detail_rows(g, "Božena Němcová"))
    assert rows["rod (tvar jména)"] == "ženský"
    assert rows["kmen"] == "božn němc"
    assert "Boženy Němcové" in rows["sloučené tvary"]
