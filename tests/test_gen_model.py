import torch

from config import GeneratorConfig
from model.gpt import GPT


def _tiny():
    return GeneratorConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32,
                           block_size=16, dropout=0.0)


def test_forward_shape():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is None


def test_forward_loss_is_scalar():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    assert loss.ndim == 0 and loss.item() > 0


def test_loss_ignores_masked_positions():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    targets = torch.full((1, 6), -100)  # vše maskováno → loss = NaN? ne, ignoruje se
    targets[0, -1] = int(idx[0, 0])
    _, loss = model(idx, targets)
    assert torch.isfinite(loss)


def test_generate_length_and_eos():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=5, top_p=0.9)
    assert 3 <= out.shape[1] <= 3 + 5
