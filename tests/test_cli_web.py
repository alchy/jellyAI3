def test_cmd_web_wires_prompt_through_iris_client(tmp_path, monkeypatch):
    """Web mluví s Iris VÝHRADNĚ přes klienta (REST): odpověď, aktivační okno
    uzlů i dokumentů jdou z API odpovědi; graf se rozsvěcí podle nich."""
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
    # popisky uzlů (nominativizace) používají morfologii — fake bez sítě
    monkeypatch.setattr("jellyai.ufal_client.UfalClient",
                        lambda services: FakeUfalClient())

    class FakeIrisClient:
        def query(self, question, temperature=0.0):
            return {"answer": "Božena Němcová", "kind": "answer",
                    "assurance": 1.0,
                    "trace": {"topic": "Babička", "predicate": "napsat",
                              "fact": {}, "answer": "Božena Němcová"},
                    "alternatives": ["Ratibořice"],
                    "used": {"components": ["graph-answerer"], "patterns": []},
                    "activation": {"nodes": [["Božena Němcová", 2.0],
                                             ["Babička", 1.4]],
                                   "docs": [["kniha_babicka", 2.1]]}}

        def close(self):
            pass

    class FakeView:
        def __init__(self):
            self.cb = None
            self.ticks = []          # web registruje VÍC smyček (animace, inbox)
            self.served = False
            self.updated = {}
            self.packets = []
            self.written = []
            self.docs = None
            self.nodes = None

        def tick(self):
            for callback in self.ticks:
                callback()

        def from_graph(self, graph): return self
        def add_node(self, *a, **k): pass
        def add_edge(self, *a, **k): pass
        def update_node(self, node_id, **attrs): self.updated[node_id] = attrs
        def packet(self, path): self.packets.append(path)
        def every(self, seconds, callback): self.ticks.append(callback)
        def open_terminal(self, callback): self.cb = callback
        def open_docs_panel(self): pass
        def open_nodes_panel(self): pass
        def write(self, text): self.written.append(text)
        def write_docs(self, ranked): self.docs = ranked
        def write_nodes(self, ranked): self.nodes = ranked
        def serve(self, open_browser=True): self.served = True
        def stop(self): pass

    view = FakeView()
    cmd_web(cfg, view=view, client=FakeIrisClient())
    assert view.served is True and view.cb is not None and view.ticks
    view.cb("kdo napsal Babičku?")               # simuluj dotaz z konzole
    assert any("Božena Němcová" in line for line in view.written)
    assert "Božena Němcová" in view.updated      # aktivace → velikost uzlu
    assert view.nodes[0][0] == "Božena Němcová"  # ⚡ aktivační okno seřazené
    assert view.docs == [("kniha_babicka", 2.1)]  # 📄 aktivní dokumenty
    for _ in range(10):                          # pár animačních tiků
        view.tick()
    assert any(pkt for pkt in view.packets)      # provoz po trase se rozjel


def test_docs_panel_writes_ranked_documents():
    """Panel dokumentů vypíše top zdroje dle aktivace (attention nad soubory)."""
    class FakeCanvas:
        def __init__(self): self.written = []
        def terminal_write(self, wid, text): self.written.append(text)
    from jellyai.viz.viewbase_view import ViewBaseView
    v = ViewBaseView.__new__(ViewBaseView)
    v._canvas = FakeCanvas()
    v._docs_id = "dokumenty"
    v.write_docs([("bible_matous", 2.4), ("bible_genesis", 0.84)])
    out = v._canvas.written[0]
    assert "bible_matous" in out and "2.40" in out
    assert out.index("bible_matous") < out.index("bible_genesis")   # sestupně


def test_update_node_skips_unknown_node():
    """Uzel, který plátno nezná (nenakrmen / ZAPOMENUT), update PŘESKOČÍ —
    nepřidá ho zpět (jinak by aktivace vzkřísila zapomenuté uzly). Přidávání
    patří výhradně feed_fact."""
    class FakeCanvas:
        def __init__(self): self.ensured = []
        def update_node(self, node_id, **attrs):
            raise ValueError(f"Uzel '{node_id}' neexistuje")
        def ensure_node(self, node_id, **meta): self.ensured.append(node_id)
    from jellyai.viz.viewbase_view import ViewBaseView
    v = ViewBaseView.__new__(ViewBaseView)
    v._canvas = FakeCanvas()
    v.update_node("uživatel", size=1.5)
    assert v._canvas.ensured == []    # neznámý uzel se NEpřidává


def test_nodes_panel_writes_ranked_activation():
    """⚡ Aktivační okno: uzly podle jasu, největší nahoře, žádný dialog."""
    class FakeCanvas:
        def __init__(self): self.written = []
        def terminal_write(self, wid, text): self.written.append(text)
    from jellyai.viz.viewbase_view import ViewBaseView
    v = ViewBaseView.__new__(ViewBaseView)
    v._canvas = FakeCanvas()
    v._nodes_id = "aktivace"
    v.write_nodes([("Josef Čapek", 4.47), ("malíř", 2.34)])
    out = v._canvas.written[0]
    assert out.index("Josef Čapek") < out.index("malíř")
    assert "4.47" in out
