import torch

from config import GeneratorConfig
from model.tokenizer import train_tokenizer, SPTokenizer
from model.gpt import GPT
from model.generate import load_generator, generate_answer


def test_generate_answer_returns_string(tmp_path):
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

    m, t, device = load_generator(gcfg, device="cpu")
    answer = generate_answer(m, t, "Roboty vyráběla firma.", "kdo vyráběl roboty",
                             gcfg, device)
    assert isinstance(answer, str)
