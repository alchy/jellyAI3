import torch

from config import Config, DataConfig, AnswererConfig, GeneratorConfig
from model.tokenizer import train_tokenizer, SPTokenizer
from model.gpt import GPT
from jellyai.pipeline import QAPipeline
from jellyai.answerer.base import Answer


def test_pipeline_generative_mode(tmp_path):
    # maličký tokenizer + model uložený jako checkpoint
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Roboty vyráběla firma Rossum Helena Domin.\n" * 30, encoding="utf-8")
    prefix = str(tmp_path / "sp")
    train_tokenizer(str(corpus), prefix, vocab_size=400)
    tok = SPTokenizer.load(prefix)
    gcfg = GeneratorConfig(
        vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=32, block_size=64,
        sp_prefix=prefix, ckpt_path=str(tmp_path / "ckpt.pt"), max_new_tokens=5,
    )
    torch.save({"model_state": GPT(gcfg).state_dict(), "gen_config": gcfg}, gcfg.ckpt_path)

    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "rur.txt").write_text(
        "Roboty vyráběla firma Rossum. Helena přijela na ostrov.", encoding="utf-8")

    cfg = Config(
        data=DataConfig(processed_dir=str(proc)),
        answerer=AnswererConfig(mode="generative"),
        generator=gcfg,
    )
    pipe = QAPipeline.from_corpus(str(proc), cfg)
    ans = pipe.ask("kdo vyráběl roboty")
    assert isinstance(ans, Answer)
    assert ans.sources  # generativní answerer uvádí zdroj top pasáže
