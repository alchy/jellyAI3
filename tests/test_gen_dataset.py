import json

from model.tokenizer import train_tokenizer, SPTokenizer
from model.dataset import QADataset, make_collate


def _tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Kontext Otázka Odpověď Roboty vyráběla firma Rossum Helena Domin. " * 40,
        encoding="utf-8",
    )
    prefix = str(tmp_path / "sp")
    # byte_fallback vyžaduje vocab ≥ ~290 (256 bajtů + znaky + speciály)
    train_tokenizer(str(corpus), prefix, vocab_size=400)
    return SPTokenizer.load(prefix)


def test_dataset_masks_prompt(tmp_path):
    tok = _tokenizer(tmp_path)
    jsonl = tmp_path / "qa.jsonl"
    jsonl.write_text(
        json.dumps({"context": "Roboty vyráběla firma.",
                    "question": "kdo vyráběl roboty", "answer": "firma"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ds = QADataset(str(jsonl), tok, block_size=64)
    assert len(ds) == 1
    idx, targets = ds[0]
    assert idx.shape == targets.shape
    assert (targets == -100).any()   # prompt je maskovaný
    assert (targets != -100).any()   # odpověď se do loss počítá


def test_collate_pads_to_max(tmp_path):
    tok = _tokenizer(tmp_path)
    jsonl = tmp_path / "qa.jsonl"
    lines = [
        {"context": "Krátký.", "question": "co", "answer": "nic"},
        {"context": "Delší kontext o robotech a firmě.", "question": "kdo", "answer": "Rossum"},
    ]
    jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines),
                     encoding="utf-8")
    ds = QADataset(str(jsonl), tok, block_size=64)
    collate = make_collate(tok.pad_id)
    idxs, tgts = collate([ds[0], ds[1]])
    assert idxs.shape == tgts.shape
    assert idxs.shape[0] == 2  # dvě položky, stejná délka po paddingu
