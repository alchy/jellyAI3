def test_retriever_temperature_widens_candidates():
    from config import RetrieverConfig
    from jellyai.chunker import Passage
    from jellyai.retriever import Retriever
    passages = [Passage("d", i, t, i, i + 1) for i, t in enumerate(
        ["robot robot robot", "robot pracuje", "moře je modré"])]
    r = Retriever(RetrieverConfig()).build(passages)
    tight = r.search("robot", temperature=0.0)
    wide = r.search("robot", temperature=1.0)
    assert len(wide) >= len(tight)
    assert len(wide) >= 2                      # široce pustí i slabší shodu


def test_graph_temperature_returns_alternatives():
    from config import AnswererConfig
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.ufal_client import FakeUfalClient
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                         Participant("num", "1890", "number")]))
    g.add_fact(make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                     Participant("num", "1915", "number")]))
    q = "kdy se narodil Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdy", "lemma": "kdy", "upos": "ADV", "head": 2, "deprel": "advmod", "start": 0, "end": 3},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 4, "end": 11},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 12, "end": 17},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [], temperature=0.7)
    assert ans.text == "1890"
    assert "1915" in ans.alternatives          # alternativa dle teploty
