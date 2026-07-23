#!/usr/bin/env python3
"""Koreference — přesnost resolveru na gold_coref.json (28 ručních děr, zatím bible).

Spustí `FillHoles.resolve_document_identity` per dokument a porovná resoluci díry
(doc, sent, token) s očekávaným antecedentem. Tvrdé číslo pro měř-first: každá změna
resolveru (např. doc-protagonista prior) se poměřuje proti tomuto baseline.
"""
import os
import json
import pickle
from collections import Counter

from fill_holes import FillHoles

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.join(HERE, "../../data/corpus")
GOLD = os.path.join(HERE, "gold_coref.json")


def _load(doc):
    shard = pickle.load(open(os.path.join(CORP, f"{doc}.pkl"), "rb"))
    return [s for (_d, _i), rec in sorted(shard.items(), key=lambda x: x[0][1])
            for s in rec["sentences"]]


def main():
    fh = FillHoles()
    gold = json.load(open(GOLD, encoding="utf-8"))["items"]
    by_doc = {}
    for it in gold:
        by_doc.setdefault(it["key"][0], []).append(it)
    correct = tot = 0
    by_cat = Counter()
    by_cat_ok = Counter()
    rows = []
    for doc, items in by_doc.items():
        res = {(k, i): lem for (k, i, _c, lem, *_r) in fh.resolve_document_identity(_load(doc))}
        for it in items:
            _d, k, i = it["key"]
            got = res.get((k, i))
            ok = bool(got) and got.lower() == it["expected"].lower()
            correct += ok
            tot += 1
            by_cat[it["cat"]] += 1
            by_cat_ok[it["cat"]] += ok
            rows.append((ok, it["form"], it["cat"], it["expected"], got or "—"))
    print(f"\n=== KOREFERENCE gold — {correct}/{tot} = {correct/tot*100:.0f}% (identity resolver) ===\n")
    for ok, form, cat, exp, got in rows:
        print(f"  {'✓' if ok else '×'} „{form:8}\" [{cat:7}] čekáno {exp:12} → {got}")
    print("\n-- dle kategorie --", {c: f"{by_cat_ok[c]}/{by_cat[c]}" for c in sorted(by_cat)})


if __name__ == "__main__":
    main()
