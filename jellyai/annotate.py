"""Offline anotace pasáží — entity (NameTag) + syntaktický rozbor (UDPipe).

Parsovat každou pasáž za běhu dotazu by bylo pomalé, tak to uděláme jednou předem
a uložíme k indexu. Query-time se pak jen čte. Anotace pasáže = její entity a věty
s tokeny (lemma, slovní druh, závislostní role) — přesně to, co potřebuje výběr
odpovědi (kdo je podmět, co je předmět).
"""

import os
import pickle


def annotate_passages(passages, client):
    """Obohatí pasáže o entity a syntaktický rozbor.

    Args:
        passages (list[Passage]): Pasáže k anotaci.
        client: ÚFAL klient (`UfalClient` nebo `FakeUfalClient`).

    Returns:
        dict: (doc_id, index) → {"entities": [...], "sentences": [[token,...],...]}.
    """
    annotations = {}
    for passage in passages:
        annotations[(passage.doc_id, passage.index)] = {
            "entities": client.entities(passage.text),
            "sentences": client.parse(passage.text),
        }
    return annotations


def save_annotations(annotations, path):
    """Uloží anotace na disk (pickle).

    Args:
        annotations (dict): Výstup :func:`annotate_passages`.
        path (str): Cílová cesta.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(annotations, f)


def load_annotations(path):
    """Načte anotace z disku.

    Args:
        path (str): Cesta k souboru s anotacemi.

    Returns:
        dict: (doc_id, index) → anotace.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
