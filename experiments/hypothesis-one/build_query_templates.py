#!/usr/bin/env python3
"""Vzorkový build query šablon přes Ollamu (build-time, offline).

NEjede celý korpus (15 912 faktů × Ollama = ~150 h) — query šablony jsou VZOR-úrovňové
a univerzální, takže stačí VARIABILITA query-VZORů. Vzorkujeme deduplikovaně dle
predikátu (jeden reprezentant s nejvíc rolemi na predikát), rovnoměrně napříč, a měříme
SATURACI: kolik distinktních query-VZORů přibývá. Když přestanou chodit nové → dost.

Na fakt bereme jen ≤ max_roles různých rolí (kontrola nákladů). Zapisuje query_templates.jsonl.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_queries import SynthQueries
from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_N = 300
MAX_ROLES = 3


def main():
    sq = SynthQueries()
    # dedup dle predikátu — reprezentant s nejvíc různými rolemi
    reps = {}
    for line in open(os.path.join(HERE, "registry.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        nr = len({a["role"] for a in e["answers"] if a["role"]})
        if e["predicate"] not in reps or nr > reps[e["predicate"]][0]:
            reps[e["predicate"]] = (nr, e)
    facts = [v[1] for v in reps.values()]
    step = max(1, len(facts) // SAMPLE_N)
    sample = facts[::step][:SAMPLE_N]
    logger("i", f"vzorek {len(sample)} faktů (dedup dle predikátu z {len(facts)}), "
                f"≤{MAX_ROLES} rolí/fakt")

    vzors, complete, total_q = set(), 0, 0
    out = open(os.path.join(HERE, "query_templates.jsonl"), "w", encoding="utf-8")
    for i, fact in enumerate(sample, 1):
        # ořízni na ≤ MAX_ROLES různých rolí (jeden slot na roli)
        seen, trimmed = set(), []
        for a in fact["answers"]:
            if a["role"] and a["role"] not in seen:
                seen.add(a["role"]); trimmed.append(a)
            if len(trimmed) >= MAX_ROLES:
                break
        fact = {**fact, "answers": trimmed}
        b = sq.synthesize_fact(fact, n=2)
        out.write(json.dumps(b, ensure_ascii=False) + "\n"); out.flush()
        complete += b["completeness"]["complete"]
        total_q += len(b["queries"])
        for q in b["queries"]:
            if q["vzor"]:
                vzors.add(q["vzor"])
        if i % 20 == 0:
            logger("i", f"  {i}/{len(sample)}  distinktních query-VZORů: {len(vzors)}  "
                        f"kompletních: {complete}  query: {total_q}")
    out.close()
    logger("i", f"HOTOVO: {len(sample)} faktů → {len(vzors)} distinktních query-VZORů, "
                f"kompletních {complete}, query {total_q}")


if __name__ == "__main__":
    main()
