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
