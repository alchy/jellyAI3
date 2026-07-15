from model.tokenizer import train_tokenizer, SPTokenizer


def test_tokenizer_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Roboty vyráběla firma. Helena přijela na ostrov. Domin řídil továrnu. " * 30,
        encoding="utf-8",
    )
    prefix = str(tmp_path / "sp")
    train_tokenizer(str(corpus), prefix, vocab_size=300)

    tok = SPTokenizer.load(prefix)
    ids = tok.encode("Roboty přijela")
    assert isinstance(ids, list) and ids
    decoded = tok.decode(ids)
    assert "Roboty" in decoded and "přijela" in decoded
    assert tok.eos_id >= 0
    assert tok.vocab_size > 0
