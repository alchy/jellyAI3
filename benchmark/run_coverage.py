"""Coverage audit — kolik vět korpusu nedává žádný typovaný fakt a proč.

`.venv/bin/python benchmark/run_coverage.py`

Doplněk etalonu: etalon měří SPRÁVNOST odpovědí, coverage měří VÝTĚŽNOST
extrakce (kde text mlčí). Kbelíky příčin ukazují, kterou metodu přidat příště.
Spouštět po každé změně extrakce a reportovat spolu s etalonem.
"""
import pickle

from config import Config
from jellyai.graph.graph import _canonical_persons, _warm_persons
from jellyai.graph.extract import extract_facts
from jellyai.graph.activation import ActivationField

_BUCKETS = ("bez slovesa i spony (výčty, nadpisy, fragmenty)",
            "sloveso, ale žádný účastník",
            "zájmenný podmět/předmět (anafora)",
            "spona, ale nic nevytěženo",
            "jiné")


def _bucket(sent):
    """Zařadí větu bez faktu do kbelíku příčiny (hrubá diagnostika)."""
    upos = [t.get("upos") for t in sent]
    deprels = [t.get("deprel") for t in sent]
    if "VERB" not in upos and "cop" not in deprels:
        return _BUCKETS[0]
    if any(u in ("PRON", "DET") and d in ("nsubj", "nsubj:pass", "obj", "iobj")
           for u, d in zip(upos, deprels)):
        return _BUCKETS[2]
    if "cop" in deprels:
        return _BUCKETS[3]
    if "VERB" in upos:
        return _BUCKETS[1]
    return _BUCKETS[4]


def _audit_document(items, buckets):
    """Projde dokument stejně jako build; vrátí (vět, bez faktu) a plní kbelíky."""
    total, empty = 0, 0
    canon = _canonical_persons(items)
    field = ActivationField()
    for _, annotation in items:
        subject = field.hottest()
        for sent in annotation.get("sentences", []):
            total += 1
            facts = extract_facts(
                {"entities": annotation.get("entities", []), "sentences": [sent]},
                default_subject=subject, canon=canon)
            if not facts:
                empty += 1
                buckets[_bucket(sent)] += 1
        _warm_persons(field, annotation, canon)
        field.step()
    return total, empty


def main():
    """Spočítá věty bez jediného faktu napříč korpusem (viz docstring modulu)."""
    with open(Config().services.annotations_path, "rb") as fh:
        annotations = pickle.load(fh)
    by_doc = {}
    for key, annotation in annotations.items():
        doc_id, idx = key if isinstance(key, tuple) else (key, 0)
        by_doc.setdefault(doc_id, []).append((idx, annotation))

    total, empty = 0, 0
    buckets = dict.fromkeys(_BUCKETS, 0)
    for _, items in sorted(by_doc.items()):
        items.sort(key=lambda t: t[0])
        doc_total, doc_empty = _audit_document(items, buckets)
        total += doc_total
        empty += doc_empty

    pct = 100 * empty // total if total else 0
    print(f"COVERAGE: {total - empty}/{total} vět s faktem "
          f"({empty} bez faktu = {pct} %)")
    for name in _BUCKETS:
        if buckets[name]:
            print(f"  {buckets[name]:4}  {name}")
    return total, empty, buckets


if __name__ == "__main__":
    main()
