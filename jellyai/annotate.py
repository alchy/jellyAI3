"""Offline anotace pasáží — entity (NameTag) + syntaktický rozbor (UDPipe).

Parsovat každou pasáž za běhu dotazu by bylo pomalé, tak to uděláme jednou předem
a uložíme k indexu. Query-time se pak jen čte. Anotace pasáže = její entity a věty
s tokeny (lemma, slovní druh, závislostní role) — přesně to, co potřebuje výběr
odpovědi (kdo je podmět, co je předmět).
"""

import os
import pickle

from jellyai.text import split_sentences


def _shift(item, base):
    """Vrátí kopii tokenu/entity s offsety start/end posunutými o `base`.

    Posun do rámce celého dokumentu zajistí, že se offsety vět nepřekrývají —
    po složení ostřicího okna z více vět pak entita jedné věty nesedne na token
    jiné (viz `selection._tokens_in_span`).

    Args:
        item (dict): Token nebo entita s klíči start/end.
        base (int): O kolik posunout.

    Returns:
        dict: Kopie s posunutými start/end (None se nechá být).
    """
    out = dict(item)
    if out.get("start") is not None:
        out["start"] = out["start"] + base
    if out.get("end") is not None:
        out["end"] = out["end"] + base
    return out


def _trim_case_mismatch(entities, sentences):
    """Usekne osobní entitu v místě PÁDOVÉ NESHODY jejích tokenů.

    NER na volném slovosledu lepí jméno s okolím („Ježíš Duchem" Nom+Ins,
    „Kain Hospodinu" Nom+Dat) — vzniklý paskvil pak tříští identitu osoby.
    Tvar rozhoduje: entita drží jen pádově konzistentní prefix.

    Args:
        entities (list[dict]): Entity věty (start/end/text/type).
        sentences (list[list[dict]]): Věty s tokeny (feats.Case, start/end).

    Returns:
        list[dict]: Entity s useknutými paskvily (kopie měněných).
    """
    tokens = [t for sent in sentences for t in sent]
    out = []
    for entity in entities:
        if entity.get("type", "")[:1].lower() != "p" \
                or entity.get("start") is None:
            out.append(entity)
            continue
        covered = [t for t in tokens
                   if t.get("start") is not None
                   and entity["start"] <= t["start"] and t["end"] <= entity["end"]
                   and t.get("feats", {}).get("Case")]
        first_case, end = None, None
        for tok in covered:
            case = tok["feats"]["Case"]
            if first_case is None:
                first_case = case
            if case != first_case:
                break
            end = tok["end"]
        if end is not None and end < entity["end"]:
            trimmed = dict(entity)
            trimmed["end"] = end
            trimmed["text"] = entity["text"][:end - entity["start"]].strip()
            out.append(trimmed)
        else:
            out.append(entity)
    return out


def annotate_documents(documents, client):
    """Obohatí dokumenty o entity a rozbor **po větách** (klíč = index věty).

    Každý dokument se rozseká `split_sentences`; každá věta se zvlášť anotuje
    (entity + syntaktický rozbor) a její offsety se posunou do rámce dokumentu,
    takže jsou napříč větami disjunktní. Answerer si pak složí anotaci libovolné
    pasáže z rozsahu jejích vět (funguje pro chunkerová i ostřicí okna).

    Args:
        documents (list[Document]): Dokumenty korpusu.
        client: ÚFAL klient (`UfalClient` nebo `FakeUfalClient`).

    Returns:
        dict: (doc_id, index věty) → {"entities": [...], "sentences": [[token,...],...]}.
    """
    annotations = {}
    for doc in documents:
        base = 0
        for i, sent in enumerate(split_sentences(doc.text)):
            parsed = client.parse(sent)
            sentences = [[_shift(tok, base) for tok in s] for s in parsed]
            entities = [_shift(e, base) for e in client.entities(sent)]
            entities = _trim_case_mismatch(entities, sentences)
            annotations[(doc.doc_id, i)] = {"entities": entities, "sentences": sentences}
            base += len(sent) + 1
    return annotations


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
