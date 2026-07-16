from config import Config, ChunkerConfig, RetrieverConfig


def test_config_defaults():
    cfg = Config()
    assert cfg.chunker.size == 3
    assert cfg.chunker.overlap == 1
    assert cfg.chunker.unit == "sentences"
    assert cfg.retriever.method == "bm25"
    assert cfg.retriever.top_k == 5
    assert cfg.retriever.use_stopwords is True
    assert cfg.answerer.template is True
    assert cfg.data.processed_dir == "data/processed"


def test_config_override():
    cfg = Config(retriever=RetrieverConfig(method="tfidf", top_k=3))
    assert cfg.retriever.method == "tfidf"
    assert cfg.retriever.top_k == 3
    # ostatní bloky zůstávají výchozí
    assert cfg.chunker.size == 3


def test_config_has_qagen_defaults():
    cfg = Config()
    assert cfg.qagen.qa_path == "data/qa/qapairs.jsonl"
    assert cfg.qagen.min_tokens == 5
    assert cfg.qagen.max_answers_per_sentence == 2
    assert "Kdo" in cfg.qagen.types


def test_config_has_v3_defaults():
    cfg = Config()
    assert cfg.answerer.mode == "extractive"
    assert cfg.services.host == "127.0.0.1"
    assert cfg.services.udpipe_port == 8082
    assert cfg.services.annotations_path == "data/annotations.pkl"
