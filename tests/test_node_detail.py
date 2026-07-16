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
    assert rows["váha"] == "2"                 # figuruje ve 2 faktech
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


def test_fact_rows_predicate_weight_participants():
    g = _graph()
    fact = next(f for f in g.facts.values() if f.predicate == "napsat")
    rows = dict(fact_detail_rows(fact))
    assert rows["typ"] == "fakt"
    assert rows["predikát"] == "napsat"
    assert rows["váha"] == "1"
    assert rows["subj"] == "Božena Němcová"
    assert rows["obj"] == "Babička"
