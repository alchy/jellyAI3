#!/usr/bin/env python3
"""Velký etalon (128+ otázek) — mode-aware skórování + rozpad po KATEGORIÍCH.

Na rozdíl od `eval_answers.py` (25 položek) měří ŠIROKÉ pokrytí: polární (ano/ne),
autorství děl, vztahy, taxonomii, formulační variace a POCTIVÉ „nevím" (honest-negative).
Každá položka nese očekávaný `mode` — poctivé doptání/mlčení u honest-negative je PASS,
ne propad. Cíl: vidět, KTERÁ třída otázek propadá (kde je výtěžnost nízká).

Vstup: gold_large.json (pole položek {q, expect, mode, kind, entity, src, factual, quote}).
Spuštění:  python3 eval_large.py
"""
import os
import sys
import json
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from answering import Answering                                   # noqa: E402

GOLD = os.path.join(HERE, "gold_large.json")


def norm(s):
    return (s or "").strip().lower()


def accept(got, answers):
    """Alias-aware: odpověď (lower) == některý povrch, nebo podřetězec (obojí směr)."""
    a = norm(got)
    if not a:
        return False
    for g in answers:
        gg = norm(g)
        if gg and (a == gg or gg in a or a in gg):
            return True
    return False


def _clear(a):
    a.store.mounted.clear()
    a.facts.mounted.clear()
    a.field.words.clear()
    a.field.files.clear()
    a.field.adj.clear()


def score(item, r):
    """PASS/FAIL jednoho tahu podle OČEKÁVANÉHO režimu.

    honest-negative / mode=unsure → PASS, když systém NEodpoví sebevědomě (nehádá).
    mode=answer → PASS, když answer + odpověď sedí. mode=clarify → PASS, když doptání.
    Vrací (ok, got, got_mode).
    """
    got = r["answer"] if r else None
    got_mode = r["mode"] if r else "unsure"
    exp_mode = item.get("mode", "answer")
    kind = item.get("kind", "")
    if kind == "honest-negative" or exp_mode == "unsure":
        ok = got_mode != "answer"                                 # nesmí sebevědomě hádat
    elif exp_mode == "clarify":
        ok = got_mode == "clarify"
    else:                                                         # answer
        ok = got_mode == "answer" and accept(got, item["expect"])
    return ok, got, got_mode


def main():
    gold = json.load(open(GOLD, encoding="utf-8"))
    a = Answering()
    rows = []
    by_kind = defaultdict(lambda: [0, 0])                         # kind -> [pass, total]
    pgroups = defaultdict(list)                                   # pgroup -> [ok, ...]
    npass = 0
    for it in gold:
        _clear(a)
        r = a.answer(it["q"])                                     # samostatný tah
        ok, got, got_mode = score(it, r)
        rows.append((it, ok, got, got_mode))
        by_kind[it.get("kind", "?")][0] += int(ok)
        by_kind[it.get("kind", "?")][1] += 1
        if it.get("pgroup"):
            pgroups[it["pgroup"]].append(ok)
        npass += int(ok)

    n = len(gold)
    print(f"\n{'='*70}\nVELKÝ ETALON — {n} otázek\n{'='*70}")
    for it, ok, got, gm in rows:                                 # jen PROPADY (pro čitelnost)
        if not ok:
            exp = "/".join(it["expect"][:3])
            print(f"  ✗ [{it.get('kind',''):15}] {it['q'][:42]:42} → {str(got)[:16]:16} "
                  f"[{gm}]  chtěli {exp} ({it.get('mode')})")
    print(f"\n-- CELKEM -- {npass}/{n} = {100*npass//max(1,n)} %")
    print("-- dle KATEGORIE (pass/total) --")
    for k in sorted(by_kind, key=lambda k: -by_kind[k][1]):
        p, t = by_kind[k]
        bar = "█" * round(10 * p / t) if t else ""
        print(f"   {k:16} {p:3}/{t:<3} {100*p//max(1,t):3}%  {bar}")
    # formulační variace: shodné výsledky napříč parafrázemi téhož faktu?
    consistent = sum(1 for v in pgroups.values() if len(set(v)) == 1)
    print(f"-- FORMULAČNÍ ROBUSTNOST -- {consistent}/{len(pgroups)} skupin parafrází konzistentních")
    out = os.path.join(HERE, "baseline_large.json")
    json.dump([{"q": it["q"], "ok": ok, "got": got, "mode": gm, "kind": it.get("kind")}
               for it, ok, got, gm in rows], open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
