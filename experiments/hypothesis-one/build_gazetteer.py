#!/usr/bin/env python3
"""Postaví GAZETTEER míst = lemmata GEO entit (NER) z korpusu → data/gazetteer.json.

Anotace ① nese entity s typem (P osoba / G místo, z ÚFAL NameType). Token, který padne
do rozsahu GEO entity, je MÍSTO — jeho canon_lemma jde do gazetteeru. Expert Topos pak
`is_place(lemma)` rozliší skutečné místo (Svatoňovice) od distraktoru (rodina) při výběru
where-slotu. Deterministické, bez sítě.

Vstup: data/corpus/*.pkl. Výstup: data/gazetteer.json (seřazený seznam lemmat míst).
"""
import os
import json
import pickle

from grammar_vzor import GrammarVzor

HERE = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.join(HERE, "../../data/corpus")
OUT = os.path.join(HERE, "../../data/gazetteer.json")


def _overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def main():
    """Projde korpus, sesbírá lemmata tokenů uvnitř GEO entit → gazetteer.json."""
    g = GrammarVzor()
    places = set()
    docs = 0
    for fn in sorted(os.listdir(CORP)):
        if not fn.endswith(".pkl"):
            continue
        docs += 1
        shard = pickle.load(open(os.path.join(CORP, fn), "rb"))
        for rec in shard.values():
            geos = [(e["start"], e["end"]) for e in rec.get("entities", []) if e["type"] == "G"]
            if not geos:
                continue
            for sent in rec["sentences"]:
                for t in sent:
                    ts, te = t.get("start"), t.get("end")
                    if ts is None or t["upos"] not in ("PROPN", "NOUN"):
                        continue
                    if any(_overlaps(ts, te, s, e) for s, e in geos):
                        places.add(g.canon_lemma(t).lower())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(sorted(places), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"gazetteer: {len(places)} míst z {docs} souborů → {OUT}")


if __name__ == "__main__":
    main()
