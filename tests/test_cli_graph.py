import pickle
from config import Config, GraphConfig, ServicesConfig
from cli import cmd_graph


def test_cmd_graph_builds_from_annotations(tmp_path):
    ann_path = str(tmp_path / "ann.pkl")
    graph_path = str(tmp_path / "graph.pkl")
    annotations = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                              "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    with open(ann_path, "wb") as f:
        pickle.dump(annotations, f)
    cfg = Config()
    cfg.services = ServicesConfig(annotations_path=ann_path)
    cfg.graph = GraphConfig(graph_path=graph_path)
    n = cmd_graph(cfg)
    assert n >= 2
    from jellyai.graph.graph import FactGraph
    g = FactGraph.load(graph_path)
    assert g.facts_of("Božena Němcová", predicate="napsat")
