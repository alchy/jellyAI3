import json
import os

from config import Config, GeneratorConfig
from model.train import train


def test_train_smoke_decreases_loss(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Roboty vyráběla firma Rossum. Helena přijela na ostrov. Domin řídil továrnu. " * 60,
        encoding="utf-8",
    )
    jsonl = tmp_path / "qa.jsonl"
    rows = [{"context": "Roboty vyráběla firma Rossum.",
             "question": "kdo vyráběl roboty", "answer": "firma Rossum"}] * 8
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                     encoding="utf-8")

    cfg = Config(generator=GeneratorConfig(
        vocab_size=400, n_layer=2, n_head=2, n_embd=32, block_size=64,
        epochs=4, batch_size=4, warmup_steps=2, lr=1e-3,
        sp_prefix=str(tmp_path / "sp"), ckpt_path=str(tmp_path / "ckpt.pt"),
    ))
    path, history = train(cfg, str(jsonl), str(corpus), device="cpu",
                          log_fn=lambda m: None)

    assert os.path.exists(path)
    assert len(history) == 4
    assert history[-1] <= history[0]  # loss na opakovaných datech klesla
