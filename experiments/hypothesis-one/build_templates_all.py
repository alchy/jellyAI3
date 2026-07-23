#!/usr/bin/env python3
"""Postaví DETERMINISTICKÉ šablony (QueryTemplate) pro CELÝ korpus z registry.jsonl.

Bez Ollamy: každý answer-slot s rolí + fact_vzor se transponuje na query-VZOR
(SynthQueries._deterministic_vzor) a uloží jako shadow-pár do per-doc shardu
(data/templates/<doc>.jsonl). Tím je VZOR matcher (answering.Answering) použitelný
NAPŘÍČ všemi doménami — nutná půda pro gold baseline (K0 fúze). Dosud byly na disku
šablony jen 4 souborů (demo), takže matcher mimo ně nemohl nic vrátit.

Vstup: registry.jsonl (④ synth_registry). Výstup: data/templates/<doc>.jsonl + počet.
Idempotentní: reset() smaže staré shardy, dedup dle id v rámci běhu.
"""
import os
import json

from answering import Answering
from synth_queries import SynthQueries
from template_store import QueryTemplate
from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "registry.jsonl")


def main():
    """Přestaví celý template store z registry (deterministicky, bez sítě)."""
    a = Answering()                       # reuse přesných objektů (g, rc, store) jako demo
    sq = SynthQueries(a.g, a.rc)
    a.store.reset()
    n = 0
    docs = set()
    lines = 0
    for line in open(REG, encoding="utf-8"):
        e = json.loads(line)
        lines += 1
        for ans in e["answers"]:
            if not ans.get("role") or not ans.get("fact_vzor"):
                continue
            qvz = sq._deterministic_vzor(ans["fact_vzor"], ans["role"])
            if not qvz:
                continue
            if a.store.append(e["doc"], QueryTemplate(
                    ans["fact_vzor"], qvz, ans["role"], ans["lemma"], None,
                    [e["doc"], e["sent"]], "deterministic")):
                n += 1
                docs.add(e["doc"])
    logger("i", f"deterministické šablony: {n} nad {len(docs)} soubory ({lines} bindings)")
    print(f"deterministických šablon: {n}  soubory: {len(docs)}")


if __name__ == "__main__":
    main()
