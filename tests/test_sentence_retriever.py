import numpy as np
from config import RetrieverConfig
from jellyai.loader import Document
from jellyai.sentence_retriever import distance_activation, SentenceRetriever


def test_distance_activation_decays_and_respects_file_boundary():
    base = [0.0, 1.0, 0.0, 5.0]
    sent_doc = ["a", "a", "a", "b"]
    sent_local = [0, 1, 2, 0]
    finals = distance_activation(base, sent_doc, sent_local, tau=1.0)
    assert finals[1] == 1.0                          # vrchol si drží své
    assert finals[0] == finals[2]                    # symetrie sever/jih
    assert abs(finals[0] - np.exp(-1.0)) < 1e-9      # útlum o 1 krok
    assert finals[3] == 5.0                           # jiný soubor: 5 nikam nezasáhne


def test_build_indexes_sentences_per_document():
    docs = [Document("da", "da", "Alfa jedna. Klíč je tady. Gama tři."),
            Document("db", "db", "Delta prší. Epsilon svítí.")]
    sr = SentenceRetriever(RetrieverConfig()).build(docs)
    assert len(sr.sent_text) == 5
    assert sr.sent_doc == ["da", "da", "da", "db", "db"]
    assert sr.sent_local == [0, 1, 2, 0, 1]          # lokální index se resetuje per dokument
    assert sr._bounds["da"] == (0, 3) and sr._bounds["db"] == (3, 5)


def test_search_focuses_on_matching_sentence():
    docs = [Document("da", "da", "Alfa jedna. Klíč leží tady. Gama tři."),
            Document("db", "db", "Delta prší. Epsilon svítí.")]
    cfg = RetrieverConfig(granularity="sentence", focus_radius=1, decay_tau=1.5)
    sr = SentenceRetriever(cfg).build(docs)
    results = sr.search("klíč", top_k=2)
    assert results, "něco se má najít"
    top_passage, top_score = results[0]
    assert top_passage.doc_id == "da"
    assert "Klíč leží tady" in top_passage.text
    assert top_passage.start <= 1 < top_passage.end
    assert top_score > 0


def test_save_load_roundtrip(tmp_path):
    docs = [Document("da", "da", "Alfa jedna. Klíč leží tady. Gama tři.")]
    sr = SentenceRetriever(RetrieverConfig(granularity="sentence")).build(docs)
    path = str(tmp_path / "sent_index.pkl")
    sr.save(path)
    loaded = SentenceRetriever.load(path)
    assert loaded.sent_text == sr.sent_text
    assert loaded.search("klíč")[0][0].text == sr.search("klíč")[0][0].text


def test_len_counts_indexed_units():
    from jellyai.chunker import Passage
    from jellyai.retriever import Retriever
    docs = [Document("d", "d", "Alfa jedna. Klíč leží tady. Gama tři.")]
    assert len(SentenceRetriever(RetrieverConfig()).build(docs)) == 3
    r = Retriever(RetrieverConfig()).build([Passage("d", 0, "text sem", 0, 1)])
    assert len(r) == 1
