import pytest

from jellyai.explain import explain_block, list_blocks


def test_list_blocks():
    assert list_blocks() == ["answerer", "chunker", "graph", "iris", "loader",
                             "pipeline", "retriever", "template"]


def test_explain_block_returns_text():
    for name in list_blocks():
        text = explain_block(name)
        assert isinstance(text, str) and text.strip()


def test_explain_unknown_block():
    with pytest.raises(KeyError):
        explain_block("neexistuje")
