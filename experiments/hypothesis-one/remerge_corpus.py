#!/usr/bin/env python3
"""Re-merge existujícího korpusu přes merge_abbreviations (lokálně, BEZ UDPipe).

Korpus byl postaven PŘED zapojením merge → zkratky rozsekané (R.U.R. → R/./U/./R/.).
Tenhle skript přežene existující shardy přes `merge_abbreviations` (parent chokepoint
princip) → zkratka = jeden PROPN token. Idempotentní (už mergnuté se nezmění). Poté je
nutné přestavět registry (④) + indexy (②) + fakty — staví se z korpusu.
"""
import os
import sys
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from jellyai.normalize import merge_abbreviations     # noqa: E402
from logger import logger                             # noqa: E402

CORP = os.path.join(HERE, "../../data/corpus")


def main():
    docs = sents = 0
    for fn in sorted(os.listdir(CORP)):
        if not fn.endswith(".pkl"):
            continue
        path = os.path.join(CORP, fn)
        rec = pickle.load(open(path, "rb"))
        changed = False
        for v in rec.values():
            after = merge_abbreviations(v["sentences"])
            if any(len(a) != len(b) for a, b in zip(after, v["sentences"])):
                v["sentences"] = after
                changed = True
                sents += 1
        if changed:
            pickle.dump(rec, open(path, "wb"))
            docs += 1
    logger("i", f"re-merge korpusu: {docs} souborů, {sents} vět sloučeno")
    print(f"re-merge: {docs} souborů, {sents} vět změněno")


if __name__ == "__main__":
    main()
