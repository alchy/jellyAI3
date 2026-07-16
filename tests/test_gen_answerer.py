import torch

from config import GeneratorConfig
from model.tokenizer import train_tokenizer, SPTokenizer
from model.gpt import GPT
from jellyai.chunker import Passage
from jellyai.answerer.base import Answer
from jellyai.answerer.generative import GenerativeAnswerer, explain


def _tiny_generator(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Roboty vyráběla firma Rossum Helena Domin.\n" * 30, encoding="utf-8")
    prefix = str(tmp_path / "sp")
    train_tokenizer(str(corpus), prefix, vocab_size=400)
    tok = SPTokenizer.load(prefix)
    gcfg = GeneratorConfig(
        vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=32, block_size=64,
        sp_prefix=prefix, ckpt_path=str(tmp_path / "ckpt.pt"), max_new_tokens=5,
    )
    model = GPT(gcfg)
    torch.save({"model_state": model.state_dict(), "gen_config": gcfg}, gcfg.ckpt_path)
    return gcfg


def test_generative_answerer_returns_answer_with_source(tmp_path):
    gcfg = _tiny_generator(tmp_path)
    answerer = GenerativeAnswerer(gcfg)
    passage = Passage("rur", 3, "Roboty vyráběla firma Rossum.", 0, 1)
    ans = answerer.answer("kdo vyráběl roboty", [(passage, 1.0)])
    assert isinstance(ans, Answer)
    assert ans.sources == ["rur#3"]


def test_generative_answerer_no_results():
    answerer = GenerativeAnswerer(None)  # bez retrievalu se model ani nenačítá
    ans = answerer.answer("cokoliv", [])
    assert ans.sources == []
    assert "nenašel" in ans.text.lower()


def test_explain_nonempty():
    assert explain().strip()
