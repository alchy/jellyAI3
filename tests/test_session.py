def test_activation_field_json_roundtrip():
    from jellyai.graph.activation import ActivationField
    f = ActivationField(); f.warm("A", 1.5); f.warm("B", 0.5)
    g = ActivationField.from_dict(f.to_dict())
    assert g.hottest() == "A"


def test_session_save_load_continues_weights(tmp_path):
    import json
    from jellyai.session import save_session, load_session
    from jellyai.graph.activation import ActivationField

    class Dummy:
        def __init__(self):
            self.context = ActivationField(); self.history = []
    a = Dummy(); a.context.warm("Božena Němcová", 2.0)
    a.history.append({"question": "…", "topic": "Božena Němcová", "answer": "1818"})
    path = save_session("test", a, graph_path="data/graph.pkl", directory=str(tmp_path))

    with open(path, encoding="utf-8") as f:
        assert json.load(f)["name"] == "test"          # čitelný JSON

    b = Dummy()
    load_session("test", b, directory=str(tmp_path))
    assert b.context.hottest() == "Božena Němcová"      # pokračuje od vah
    assert b.history[-1]["answer"] == "1818"
