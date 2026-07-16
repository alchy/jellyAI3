"""Dataset QA párů pro trénink generátoru.

Model se má naučit jedno: „když dostaneš kontext a otázku, napiš odpověď". Proto
každý pár složíme do jedné sekvence `Kontext: … Otázka: … Odpověď: <odpověď><eos>`
a **loss počítáme jen na tokenech odpovědi** — na kontextu a otázce ne (ty jsou
jen zadání, ne to, co se má model naučit produkovat). Maskování dělá právě tohle:
pozicím promptu dá label -100, který cross-entropy ignoruje.
"""

import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

_PROMPT = "Kontext: {context}\nOtázka: {question}\nOdpověď: "


def _build_example(prompt_ids, answer_ids, block_size):
    """Slepí prompt a odpověď do sekvence a labelů; ořízne na block_size.

    Když je to delší než block_size, ořízne se **prompt zleva** (odpověď se
    zachová — je to to podstatné).

    Args:
        prompt_ids (list[int]): Tokeny promptu (Kontext/Otázka/Odpověď:).
        answer_ids (list[int]): Tokeny odpovědi + <eos>.
        block_size (int): Maximální délka sekvence.

    Returns:
        tuple[list[int], list[int]]: (sekvence, labely). Labely mají -100 na promptu.
    """
    if len(prompt_ids) + len(answer_ids) > block_size:
        keep = block_size - len(answer_ids)
        if keep > 0:
            prompt_ids = prompt_ids[-keep:]
        else:
            prompt_ids = []
            answer_ids = answer_ids[:block_size]
    seq = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return seq, labels


class QADataset(Dataset):
    """Tokenizované QA páry se zamaskovaným promptem, připravené pro DataLoader."""

    def __init__(self, jsonl_path, tokenizer, block_size):
        """Načte JSONL, každý pár složí do formátu a zamaskuje prompt.

        Args:
            jsonl_path (str): Cesta k datasetu (řádky s question/context/answer).
            tokenizer (SPTokenizer): Tokenizer.
            block_size (int): Maximální délka sekvence.
        """
        self.examples = []
        eos = tokenizer.eos_id
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt = _PROMPT.format(context=row["context"], question=row["question"])
                prompt_ids = tokenizer.encode(prompt)
                answer_ids = tokenizer.encode(row["answer"]) + [eos]
                seq, labels = _build_example(prompt_ids, answer_ids, block_size)
                if len(seq) >= 2:  # potřebujeme aspoň vstup + jeden cíl
                    self.examples.append((seq, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """Vrátí (input_ids, targets) posunuté o 1 pro jazykové modelování.

        Args:
            i (int): Index příkladu.

        Returns:
            tuple[Tensor, Tensor]: (vstupní tokeny, cílové tokeny) stejné délky;
                cíle mají -100 na pozicích, které se do loss nezapočítávají.
        """
        seq, labels = self.examples[i]
        idx = torch.tensor(seq[:-1], dtype=torch.long)
        targets = torch.tensor(labels[1:], dtype=torch.long)
        return idx, targets


def make_collate(pad_id):
    """Vytvoří collate funkci, která dávku doplní (padding) na stejnou délku.

    Padding je na konci: vstupy se doplní `pad_id`, cíle `-100` (ignorují se).
    Kauzální maska zajistí, že reálné tokeny na koncový padding nekoukají.

    Args:
        pad_id (int): Id výplňového tokenu.

    Returns:
        callable: Funkce (batch) → (input_ids, targets) tvaru (B, Tmax).
    """
    def collate(batch):
        max_len = max(idx.size(0) for idx, _ in batch)
        idxs, tgts = [], []
        for idx, targets in batch:
            pad = max_len - idx.size(0)
            idxs.append(F.pad(idx, (0, pad), value=pad_id))
            tgts.append(F.pad(targets, (0, pad), value=-100))
        return torch.stack(idxs), torch.stack(tgts)

    return collate
