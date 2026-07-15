from jellyai.loader import load_documents, Document, explain


def test_load_documents(tmp_path):
    (tmp_path / "rur.txt").write_text("Text R.U.R.", encoding="utf-8")
    (tmp_path / "babicka.txt").write_text("Text Babičky", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("neload", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert [d.doc_id for d in docs] == ["babicka", "rur"]  # seřazené
    assert isinstance(docs[0], Document)
    assert docs[1].text == "Text R.U.R."


def test_explain_nonempty():
    assert isinstance(explain(), str) and explain().strip()
