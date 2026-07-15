from config import ChunkerConfig
from jellyai.loader import Document
from jellyai.chunker import chunk, Passage, explain


def _doc(n):
    text = " ".join(f"Věta číslo {i}." for i in range(n))
    return Document(doc_id="d", title="d", text=text)


def test_chunk_overlap_and_coverage():
    cfg = ChunkerConfig(size=3, overlap=1)  # step = 2
    passages = chunk(_doc(6), cfg)
    # okna vět: [0:3], [2:5], [4:6]
    assert [(p.start, p.end) for p in passages] == [(0, 3), (2, 5), (4, 6)]
    assert [p.index for p in passages] == [0, 1, 2]
    assert isinstance(passages[0], Passage)
    assert passages[0].doc_id == "d"


def test_chunk_short_document():
    cfg = ChunkerConfig(size=3, overlap=1)
    passages = chunk(_doc(2), cfg)
    assert len(passages) == 1
    assert (passages[0].start, passages[0].end) == (0, 2)


def test_chunk_empty_document():
    cfg = ChunkerConfig(size=3, overlap=1)
    assert chunk(Document("d", "d", "   "), cfg) == []


def test_explain_nonempty():
    assert explain().strip()
