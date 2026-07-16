"""Trénovací smyčka generátoru.

Vezme QA dataset a naučí malý transformer předpovídat tokeny odpovědi. Nic
exotického: AdamW, cosine rozvrh learning rate s krátkým warmupem, ořezávání
gradientu a cross-entropy jen na tokenech odpovědi (zbytek je maskovaný). Běží
na MPS (Apple GPU), a když není, spadne na CPU. Tokenizer se natrénuje automaticky,
pokud ještě neexistuje.
"""

import math
import os
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from model.tokenizer import train_tokenizer, SPTokenizer
from model.dataset import QADataset, make_collate
from model.gpt import GPT


def pick_device():
    """Vrátí nejlepší dostupné zařízení: MPS (Apple GPU), jinak CPU.

    Returns:
        str: "mps" nebo "cpu".
    """
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train(config, jsonl_path, corpus_path, device=None, log_fn=print):
    """Natrénuje generátor na QA datech a uloží checkpoint.

    Když ještě neexistuje SentencePiece model, natrénuje ho na `corpus_path`.
    Skutečná velikost slovníku se převezme z tokenizeru (ne z configu).

    Args:
        config (Config): Konfigurace (bere se `config.generator`).
        jsonl_path (str): Cesta k QA datasetu (qapairs.jsonl).
        corpus_path (str): Textový korpus pro trénink tokenizeru.
        device (str | None): Zařízení; None = automaticky (MPS/CPU).
        log_fn (callable): Kam logovat průběh (výchozí print).

    Returns:
        tuple[str, list[float]]: (cesta k checkpointu, průměrné loss po epochách).
    """
    gcfg = config.generator
    device = device or pick_device()

    sp_model = gcfg.sp_prefix + ".model"
    if not os.path.exists(sp_model):
        os.makedirs(os.path.dirname(gcfg.sp_prefix) or ".", exist_ok=True)
        log_fn(f"Trénuji tokenizer na {corpus_path} …")
        train_tokenizer(corpus_path, gcfg.sp_prefix, gcfg.vocab_size)
    tok = SPTokenizer.load(gcfg.sp_prefix)

    # skutečná velikost slovníku z tokenizeru (SentencePiece může dát méně pieců)
    gcfg = replace(gcfg, vocab_size=tok.vocab_size)

    dataset = QADataset(jsonl_path, tok, gcfg.block_size)
    if len(dataset) == 0:
        raise ValueError(f"Dataset {jsonl_path} je prázdný — spusť nejdřív gen-qa.")
    loader = DataLoader(dataset, batch_size=gcfg.batch_size, shuffle=True,
                        collate_fn=make_collate(tok.pad_id))

    model = GPT(gcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gcfg.lr,
                                  weight_decay=gcfg.weight_decay)
    total_steps = max(1, gcfg.epochs * len(loader))

    def lr_scale(step):
        """Lineární warmup, pak cosine pokles k nule."""
        if step < gcfg.warmup_steps:
            return (step + 1) / max(1, gcfg.warmup_steps)
        progress = (step - gcfg.warmup_steps) / max(1, total_steps - gcfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    model.train()
    history = []
    for epoch in range(gcfg.epochs):
        running = 0.0
        for idx, targets in loader:
            idx, targets = idx.to(device), targets.to(device)
            _, loss = model(idx, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gcfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        avg = running / len(loader)
        history.append(avg)
        log_fn(f"epocha {epoch + 1}/{gcfg.epochs}  loss {avg:.4f}")

    os.makedirs(os.path.dirname(gcfg.ckpt_path) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "gen_config": gcfg}, gcfg.ckpt_path)
    log_fn(f"Model uložen → {gcfg.ckpt_path}")
    return gcfg.ckpt_path, history
