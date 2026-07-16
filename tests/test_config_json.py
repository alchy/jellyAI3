def test_config_json_roundtrip(tmp_path):
    from config import Config
    cfg = Config()
    cfg.retriever.method = "tfidf"
    path = str(tmp_path / "config.json")
    cfg.to_json(path)
    loaded = Config.from_json(path)
    assert loaded.retriever.method == "tfidf"
    assert loaded.chunker.size == cfg.chunker.size


def test_config_json_is_readable(tmp_path):
    import json
    from config import Config
    path = str(tmp_path / "config.json")
    Config().to_json(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)                     # čitelný JSON, ne pickle
    assert data["retriever"]["method"] == "bm25"
