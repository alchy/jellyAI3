import pytest

from config import RetrieverConfig
from jellyai.chunker import Passage
from jellyai.retriever import Retriever, explain


def _passages():
    return [
        Passage("rur", 0,
                "Roboty vyráběla firma Rossumovy univerzální roboty. "
                "Starý Rossum je vynalezl.", 0, 2),
        Passage("rur", 1,
                "Helena přijela na ostrov. Domin jí ukázal továrnu.", 0, 2),
        Passage("rur", 2,
                "Nána vařila oběd v kuchyni.", 0, 1),
    ]


@pytest.mark.parametrize("method", ["bm25", "tfidf"])
def test_retriever_finds_relevant_passage(method):
    r = Retriever(RetrieverConfig(method=method, top_k=2)).build(_passages())
    results = r.search("roboty")
    assert results, "měl by najít aspoň jednu pasáž"
    assert results[0][0].index == 0  # pasáž o robotech je první


def test_search_no_match_returns_empty():
    r = Retriever(RetrieverConfig(method="bm25")).build(_passages())
    assert r.search("kvantová chromodynamika") == []


def test_search_empty_index():
    r = Retriever(RetrieverConfig()).build([])
    assert r.search("cokoliv") == []


def test_save_load_roundtrip(tmp_path):
    r = Retriever(RetrieverConfig(method="bm25")).build(_passages())
    path = str(tmp_path / "index.pkl")
    r.save(path)
    loaded = Retriever.load(path)
    res_orig = r.search("roboty")
    res_loaded = loaded.search("roboty")
    # Načtený index dává stejné pořadí i stejnou top pasáž jako originál.
    assert [p.index for p, _ in res_orig] == [p.index for p, _ in res_loaded]
    assert res_loaded[0][0].index == 0


def test_score_all_covers_all_passages_and_matches_search():
    from config import RetrieverConfig
    from jellyai.chunker import Passage
    from jellyai.retriever import Retriever
    passages = [
        Passage("d", 0, "roboti pracují v továrně", 0, 1),
        Passage("d", 1, "Helena přišla do továrny", 1, 2),
        Passage("d", 2, "moře je modré", 2, 3),
    ]
    r = Retriever(RetrieverConfig()).build(passages)
    scores = r.score_all("roboti v továrně")
    assert scores.shape == (3,)
    import numpy as np
    top = int(np.argmax(scores))
    assert r.search("roboti v továrně")[0][0].index == passages[top].index


def test_explain_nonempty():
    assert explain().strip()
