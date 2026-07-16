def test_graph_answer_explains_path():
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 10},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj", "start": 11, "end": 18},
    ]]})
    ans = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig())).answer(q, [])
    ex = ans.explain()
    assert "Babička" in ex and "napsat" in ex and "Božena Němcová" in ex


def test_explain_without_trace_returns_text():
    from jellyai.answerer.base import Answer
    assert Answer(text="ahoj").explain() == "ahoj"
