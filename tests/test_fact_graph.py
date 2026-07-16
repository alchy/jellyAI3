from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph, build_graph


def _born(year):
    return make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                 Participant("num", year, "number")])


def test_fact_weight_aggregates_repetition():
    g = FactGraph()
    for _ in range(3):
        g.add_fact(_born("1890"))
    g.add_fact(_born("1915"))
    born90 = [f for f in g.facts.values()
              if ("num", "1890", "number") in
              [(p.role, p.node, p.type) for p in f.participants]][0]
    assert born90.weight == 3
    facts = g.facts_of("Čapek", role="subj", predicate="narodit")
    assert len(facts) == 2
    assert g.participants(born90, "num") == ["1890"]


def test_build_graph_from_annotations():
    ann = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                      "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    g = build_graph(ann)
    assert g.facts_of("Božena Němcová", role="subj", predicate="napsat")


def test_graph_save_load_roundtrip(tmp_path):
    g = FactGraph()
    g.add_fact(_born("1890"))
    path = str(tmp_path / "graph.pkl")
    g.save(path)
    loaded = FactGraph.load(path)
    assert list(loaded.facts.keys()) == list(g.facts.keys())
    assert loaded.facts_of("Čapek", predicate="narodit")[0].weight == 1
