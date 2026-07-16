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


def test_build_fact_graph_resolves_person_case_variants(tmp_path):
    # celá pipeline (build + recover + resolve): akuzativ „Boženu Němcovou"
    # z druhého dokumentu se slije do nominativního uzlu
    from config import Config, ServicesConfig, GraphConfig
    from jellyai.tasks import build_fact_graph
    ann = {
        ("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                   "sentences": [[
            {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
            {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
            {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
            {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
        ]]},
        ("d", 1): {"entities": [{"text": "Josef", "type": "P", "start": 0, "end": 5},
                                {"text": "Boženu Němcovou", "type": "P", "start": 11, "end": 26}],
                   "sentences": [[
            {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "ctil", "lemma": "ctít", "upos": "VERB", "head": 0, "deprel": "root", "start": 6, "end": 10},
            {"form": "Boženu", "lemma": "Božena", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 17},
            {"form": "Němcovou", "lemma": "Němcová", "upos": "PROPN", "head": 3, "deprel": "flat", "start": 18, "end": 26},
        ]]},
    }
    ann_path = str(tmp_path / "ann.pkl")
    with open(ann_path, "wb") as f:
        pickle.dump(ann, f)
    cfg = Config()
    cfg.services = ServicesConfig(annotations_path=ann_path)
    cfg.graph = GraphConfig(graph_path=str(tmp_path / "graph.pkl"))
    graph = build_fact_graph(cfg)
    assert "Boženu Němcovou" not in graph.nodes
    assert graph.facts_of("Božena Němcová", role="obj", predicate="ctít")
