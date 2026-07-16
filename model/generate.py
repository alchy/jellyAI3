"""Generování odpovědi natrénovaným modelem.

Sestaví stejný prompt jako při tréninku (`Kontext: … Otázka: … Odpověď: `) a nechá
model dopsat část za „Odpověď:". Dopisování je autoregresivní sampling — token po
tokenu, dokud nepřijde `<eos>` nebo se nevyčerpá limit. Vrací se jen ta nově
vygenerovaná část (prompt zahazujeme, ten už uživatel zná).
"""

import os

import torch

from model.tokenizer import SPTokenizer
from model.gpt import GPT
from model.train import pick_device

_PROMPT = "Kontext: {context}\nOtázka: {question}\nOdpověď: "


def load_generator(gen_config, device=None):
    """Načte natrénovaný model a tokenizer z checkpointu.

    Args:
        gen_config (GeneratorConfig): Konfigurace (cesty sp_prefix, ckpt_path).
        device (str | None): Zařízení; None = automaticky (MPS/CPU).

    Returns:
        tuple: (model v eval režimu, tokenizer, device).

    Raises:
        FileNotFoundError: Když checkpoint neexistuje.
    """
    device = device or pick_device()
    if not os.path.exists(gen_config.ckpt_path):
        raise FileNotFoundError(
            f"Chybí checkpoint {gen_config.ckpt_path} — spusť nejdřív `train-gen`."
        )
    tokenizer = SPTokenizer.load(gen_config.sp_prefix)
    ckpt = torch.load(gen_config.ckpt_path, map_location=device, weights_only=False)
    model = GPT(ckpt["gen_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, device


def generate_answer(model, tokenizer, context, question, gen_config, device):
    """Vygeneruje odpověď na otázku nad daným kontextem.

    Args:
        model (GPT): Natrénovaný model.
        tokenizer (SPTokenizer): Tokenizer.
        context (str): Nalezená pasáž (kontext).
        question (str): Otázka uživatele.
        gen_config (GeneratorConfig): Parametry samplingu (temperature/top_k/top_p/…).
        device (str): Zařízení.

    Returns:
        str: Vygenerovaná odpověď (bez promptu, oříznutá na `<eos>`).
    """
    prompt = _PROMPT.format(context=context, question=question)
    prompt_ids = tokenizer.encode(prompt)
    # ať se prompt vejde a zbyde místo na generování — ořízneme ho zleva
    max_prompt = model.block_size - 1
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=gen_config.max_new_tokens,
        temperature=gen_config.temperature, top_k=gen_config.top_k,
        top_p=gen_config.top_p, eos_id=tokenizer.eos_id,
    )
    new_ids = out[0, len(prompt_ids):].tolist()
    if tokenizer.eos_id in new_ids:                 # odřízni vše od <eos>
        new_ids = new_ids[:new_ids.index(tokenizer.eos_id)]
    return tokenizer.decode(new_ids).strip()
