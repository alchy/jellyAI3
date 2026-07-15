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


def test_explain_nonempty():
    assert explain().strip()
