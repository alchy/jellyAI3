#!/usr/bin/env python3
"""Konzolový MOMENT query šablon — tabulka fakt-VZOR × query-VZOR + nové distinktní VZORy.

Čte rostoucí `query_templates.jsonl` a vypíše NOVÉ (distinktní) query-VZORy jako tabulku
šablona faktu × šablona otázky. Spouštěj opakovaně = momenty, jak batch roste.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "query_templates.jsonl")


def main(last=24):
    if not os.path.exists(PATH):
        print("(query_templates.jsonl ještě není)")
        return
    seen, rows = set(), []
    facts = complete = total_q = 0
    for line in open(PATH, encoding="utf-8"):
        try:
            b = json.loads(line)
        except Exception:
            continue
        facts += 1
        complete += b["completeness"]["complete"]
        for q in b["queries"]:
            total_q += 1
            vz = q.get("vzor")
            if vz and vz not in seen:
                seen.add(vz)
                rows.append((q.get("fact_vzor") or "—", vz,
                             f"{q['target']}→{q['role']}", q["surface"]))
    fw, qw, rw = 32, 32, 15
    print(f"=== MOMENT: {facts} faktů · {len(seen)} distinktních query-VZORů · "
          f"kompletních {complete}/{facts} · query {total_q} ===")
    print(f"  {'FAKT-VZOR (díra)':{fw}}  {'QUERY-VZOR':{qw}}  {'role':{rw}} otázka")
    print("  " + "-" * (fw + qw + rw + 30))
    for fvz, qvz, role, surf in rows[-last:]:
        print(f"  {fvz[:fw]:{fw}}  {qvz[:qw]:{qw}}  {role[:rw]:{rw}} {surf[:36]}")
    print(f"  … (posledních {min(last, len(rows))} nových z {len(rows)} distinktních)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
