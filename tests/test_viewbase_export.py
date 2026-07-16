from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.viewbase_export import to_json


def test_to_json_has_fact_nodes_and_role_edges():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Čapek", "person"),
                                    Participant("obj", "R.U.R.", "concept")]))
    data = to_json(g)
    ids = {n["id"] for n in data["nodes"]}
    types = {n["type"] for n in data["nodes"]}
    assert "Čapek" in ids and "R.U.R." in ids
    assert "fact" in types                       # faktový uzel je taky uzel
    roles = {e["role"] for e in data["edges"]}
    assert {"subj", "obj"} <= roles              # role-hrany


def test_context_facts_marked_as_weak_association():
    """Kontextový fakt je ve viz odlišený: label „souvislost", kind=context na
    uzlu i hranách — hrana KAČ—Ludvík Němec nesmí vypadat jako přímý vztah."""
    g = FactGraph()
    g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                     Participant("obj", "Ludvík Němec", "person")]))
    data = to_json(g)
    fact_node = next(n for n in data["nodes"] if n["type"] == "fact")
    assert fact_node["label"] == "souvislost"
    assert fact_node["kind"] == "context"
    assert all(e.get("kind") == "context" for e in data["edges"])
