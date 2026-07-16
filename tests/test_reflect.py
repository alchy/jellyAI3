def test_reflect_pushes_activation_and_flow():
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient
    from jellyai.viz.reflect import reflect

    class FakeView:
        def __init__(self):
            self.updated = {}
            self.flows = []

        def add_node(self, node_id, **meta): pass
        def add_edge(self, src, dst, **meta): pass
        def update_node(self, node_id, **attrs): self.updated[node_id] = attrs
        def flow(self, path): self.flows.append(path)
        def on_prompt(self, callback): pass
        def serve(self, open_browser=True): pass
        def stop(self): pass

    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.answer(q, [])                       # rozsvítí těžiště + nastaví trasu
    view = FakeView()
    reflect(view, a)
    assert "Božena Němcová" in view.updated          # aktivace nodu
    assert view.flows and "Babička" in view.flows[0] and "Božena Němcová" in view.flows[0]
