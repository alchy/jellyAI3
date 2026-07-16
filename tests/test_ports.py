def test_ports_are_structural():
    from jellyai.ports import QuestionAnalyzer, Composer

    class MyAnalyzer:
        def analyze(self, question):
            return None
    assert isinstance(MyAnalyzer(), QuestionAnalyzer)

    class MyComposer:
        def compose(self, question, facts):
            return "text"
    assert isinstance(MyComposer(), Composer)


def test_graphview_port_structural():
    from jellyai.ports import GraphView

    class FakeView:
        def add_node(self, node_id, **meta): pass
        def add_edge(self, src, dst, **meta): pass
        def update_node(self, node_id, **attrs): pass
        def flow(self, path): pass
        def on_prompt(self, callback): pass
        def serve(self, open_browser=True): pass
        def stop(self): pass
    assert isinstance(FakeView(), GraphView)
