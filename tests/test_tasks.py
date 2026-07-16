import pickle


def _write_annotations(path):
    ann = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                      "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    with open(path, "wb") as f:
        pickle.dump(ann, f)


def test_build_and_load_fact_graph(tmp_path):
    from config import Config, ServicesConfig, GraphConfig
    from jellyai.tasks import build_fact_graph, load_fact_graph
    ann_path = str(tmp_path / "ann.pkl")
    graph_path = str(tmp_path / "graph.pkl")
    _write_annotations(ann_path)
    cfg = Config()
    cfg.services = ServicesConfig(annotations_path=ann_path)
    cfg.graph = GraphConfig(graph_path=graph_path)
    graph = build_fact_graph(cfg)
    assert graph.facts_of("Božena Němcová", predicate="napsat")
    assert load_fact_graph(cfg).facts_of("Božena Němcová", predicate="napsat")
