def test_cmd_web_wires_prompt_to_ask(tmp_path, monkeypatch):
    from config import Config, GraphConfig
    from cli import cmd_web
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient

    graph_path = str(tmp_path / "graph.pkl")
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    g.save(graph_path)
    cfg = Config()
    cfg.graph = GraphConfig(graph_path=graph_path)

    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    # make_graph_answerer vytváří UfalClient přes `from jellyai.ufal_client import UfalClient`
    monkeypatch.setattr("jellyai.ufal_client.UfalClient", lambda services: client)

    class FakeView:
        def __init__(self):
            self.cb = None
            self.tick = None
            self.served = False
            self.updated = {}
            self.packets = []
            self.written = []

        def from_graph(self, graph): return self
        def add_node(self, *a, **k): pass
        def add_edge(self, *a, **k): pass
        def update_node(self, node_id, **attrs): self.updated[node_id] = attrs
        def packet(self, path): self.packets.append(path)
        def every(self, seconds, callback): self.tick = callback
        def open_terminal(self, callback): self.cb = callback
        def write(self, text): self.written.append(text)
        def serve(self, open_browser=True): self.served = True
        def stop(self): pass

    view = FakeView()
    cmd_web(cfg, view=view)
    assert view.served is True and view.cb is not None and view.tick is not None
    view.cb(q)                                   # simuluj dotaz z konzole
    assert any("Božena Němcová" in line for line in view.written)  # odpověď v konzoli
    assert "Božena Němcová" in view.updated      # aktivace se promítla do velikosti
    for _ in range(10):                          # pár animačních tiků
        view.tick()
    assert any(pkt for pkt in view.packets)      # provoz po trase se rozjel
