import json

from config import Config, DataConfig, QagenConfig
from qagen.tagger import Entity, FakeTagger
from qagen.build import build_dataset


def test_build_dataset(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    s = "Roboty vynalezl starý Rossum."
    (proc / "rur.txt").write_text(s + " Helena přijela na ostrov.", encoding="utf-8")
    qa = tmp_path / "qa" / "qapairs.jsonl"
    cfg = Config(
        data=DataConfig(processed_dir=str(proc)),
        qagen=QagenConfig(qa_path=str(qa), min_tokens=3),
    )
    ft = FakeTagger(entities={s: [Entity("starý Rossum", "P", 16, 28)]})
    pairs = build_dataset(cfg, ft)

    assert len(pairs) == 1
    assert pairs[0]["question"] == "Kdo roboty vynalezl?"
    assert pairs[0]["answer"] == "starý Rossum"
    assert pairs[0]["type"] == "Kdo"
    assert pairs[0]["doc_id"] == "rur"
    assert qa.exists()
    first = json.loads(qa.read_text(encoding="utf-8").splitlines()[0])
    assert first["question"] == "Kdo roboty vynalezl?"


def test_build_dataset_skips_long_sentences(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    # Run-on „věta" (přes max_tokens) se přeskočí i s přítomnou entitou.
    sentence = "Roboty vynalezl starý Rossum " + "a " * 50 + "konec."
    (proc / "rur.txt").write_text(sentence, encoding="utf-8")
    cfg = Config(
        data=DataConfig(processed_dir=str(proc)),
        qagen=QagenConfig(qa_path=str(tmp_path / "qa" / "p.jsonl"),
                          min_tokens=3, max_tokens=10),
    )
    ft = FakeTagger(entities={sentence: [Entity("starý Rossum", "P", 16, 28)]})
    assert build_dataset(cfg, ft) == []
