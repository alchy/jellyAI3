#!/usr/bin/env python3
"""Doménový etalon — collision (routing scoreboard) + coverage (per-entita pokrytí).

Na rozdíl od eval_large (široké pokrytí) tento etalon CÍLENĚ stresuje výběr souboru:
  • kind=collision — lexikálně matoucí otázka (sdílené příjmení, autor × jeho dílo, víc
    entit z téhož města). Kromě odpovědi se kontroluje i VÍTĚZNÝ SOUBOR (`expect_doc`):
    jádro „retrieval po uzlech" je vybrat SPRÁVNÝ soubor mezi kolidujícími kandidáty.
  • coverage — systematická per-entita fakta (rok/místo/dílo/povolání/taxonomie), ověřená
    ručně proti raw textu (ne proti extrakci → netestuje se extrakce sama sebou).

Ground truth je kurátorovaný (gold_domain.json). Odpovědi ověřeny proti ZDROJI.

Spuštění:
  python3 eval_domain.py            # současný config (jeden mode), plný rozpad
  python3 eval_domain.py --sweep    # porovná off/replace/union/union_cap = scoreboard mountu
"""
import os
import sys
import json
import datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from answering import Answering                      # noqa: E402
from eval_large import accept, _clear                # noqa: E402
import test_report                                   # noqa: E402

GOLD = os.path.join(HERE, "gold_domain.json")
DOCS = os.path.join(HERE, "docs", "last-test.html")

# módy kompozice mountu (answering._compose_mount) pro --sweep
SWEEP = {
    "off":       {"enabled": False},
    "replace":   {"enabled": True, "max_docs": 4, "max_fact_refs": 24, "mode": "replace"},
    "union":     {"enabled": True, "max_docs": 4, "max_fact_refs": 24, "mode": "union"},
    "union_cap": {"enabled": True, "max_docs": 4, "max_fact_refs": 24, "mode": "union_cap"},
}


def load_items():
    """gold_domain.json → plochý seznam položek (podporuje i {collision:[...], coverage:[...]})."""
    data = json.load(open(GOLD, encoding="utf-8"))
    if isinstance(data, dict):
        items = []
        for group in data.values():
            items.extend(group)
        return items
    return data


def won_doc(r):
    """Vítězný soubor z výsledku (fact_ref = [doc, sent]) nebo None."""
    ref = r.get("fact_ref") if r else None
    if isinstance(ref, (list, tuple)) and ref:
        return ref[0]
    return None


def score(item, r):
    """(ans_ok, doc_ok, got, got_mode). doc_ok je None u položek bez `expect_doc`.

    Mode-aware jako eval_large: honest-negative/unsure → PASS když NEodpoví sebevědomě;
    clarify → PASS když se doptá; answer → PASS když answer + odpověď sedí. Navíc u
    collision (expect_doc) ověří, že vyhrál SPRÁVNÝ soubor.
    """
    got = r["answer"] if r else None
    got_mode = r["mode"] if r else "unsure"
    exp_mode = item.get("mode", "answer")
    kind = item.get("kind", "")
    if kind == "honest-negative" or exp_mode == "unsure":
        ans_ok = got_mode != "answer"
    elif exp_mode == "clarify":
        ans_ok = got_mode == "clarify"
    else:
        ans_ok = got_mode == "answer" and accept(got, item["expect"])
    doc_ok = None
    if item.get("expect_doc") and got_mode == "answer":
        doc_ok = won_doc(r) == item["expect_doc"]
    return ans_ok, doc_ok, got, got_mode


def run(a, items):
    """Odpoví na všechny položky (samostatné tahy) → seznam (item, ans_ok, doc_ok, got, mode)."""
    rows = []
    for it in items:
        _clear(a)
        r = a.answer(it["q"])
        ans_ok, doc_ok, got, gm = score(it, r)
        rows.append((it, ans_ok, doc_ok, got, gm))
    return rows


def _norm(rows):
    """(item, ans_ok, doc_ok, got, mode) → normalizované dict řádky pro test_report."""
    return [{"kind": it.get("kind", "?"), "q": it["q"], "got": got,
             "expect": it.get("expect", []), "mode": gm, "ok": ans_ok,
             "doc_ok": doc_ok, "expect_doc": it.get("expect_doc")}
            for it, ans_ok, doc_ok, got, gm in rows]


def _stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def summarize(rows, label=""):
    n = len(rows)
    npass = sum(1 for _, ok, _, _, _ in rows if ok)
    coll = [(it, ok, dok, got, gm) for it, ok, dok, got, gm in rows if it.get("kind") == "collision"]
    coll_ans = sum(1 for _, ok, _, _, _ in coll if ok)
    coll_doc = sum(1 for _, _, dok, _, _ in coll if dok)
    coll_doc_tot = sum(1 for _, _, dok, _, _ in coll if dok is not None)
    print(f"\n=== DOMÉNOVÝ ETALON {label}— {npass}/{n} = {100*npass//max(1,n)}% "
          f"| collision odpověď {coll_ans}/{len(coll)} | správný SOUBOR {coll_doc}/{coll_doc_tot} ===")
    by_kind = defaultdict(lambda: [0, 0])
    for it, ok, _, _, _ in rows:
        k = it.get("kind", "?")
        by_kind[k][0] += int(ok); by_kind[k][1] += 1
    print("-- dle KIND --", {k: f"{p}/{t}" for k, (p, t) in
                             sorted(by_kind.items(), key=lambda x: -x[1][1])})
    return npass, coll_ans, coll_doc


def main_single():
    a = Answering()
    items = load_items()
    rows = run(a, items)
    summarize(rows)
    mode = a.entity_retrieval.get("mode", "union") if a.entity_retrieval.get("enabled") else "off"
    test_report.write_scoreboard(
        DOCS, "Poslední test — doménový etalon", _norm(rows),
        subtitle="Collision (routing scoreboard) + coverage (per-entita). Ground truth ověřený "
                 "proti raw korpusu. Otázka → odpověď systému → očekávaná odpověď, po doménách.",
        config_label=f"mount mode = {mode}", timestamp=_stamp(),
        source=f"gold_domain.json · {len(items)} otázek")
    print(f"→ {DOCS}")
    print("\n-- PROPADY (ANS=špatná odpověď, DOC=špatný soubor) --")
    for it, ok, dok, got, gm in rows:
        if ok and dok is not False:
            continue
        flag = ("" if ok else "ANS") + ("+DOC" if dok is False else "")
        exp = "/".join(it["expect"][:2]) if it.get("expect") else "?"
        tail = f"  (soubor: chtěli {it.get('expect_doc')})" if dok is False else ""
        print(f"  ✗[{flag.strip('+'):7}] [{it.get('kind',''):11}] {it['q'][:40]:40} → "
              f"{str(got)[:16]:16} [{gm}] chtěli {exp}{tail}")


def main_sweep():
    a = Answering()
    items = load_items()
    # parse cache napříč módy (UDPipe je bottleneck)
    _orig = a._parse
    _pc = {}
    a._parse = lambda q: _pc[q] if q in _pc else _pc.setdefault(q, _orig(q))
    n = len(items)
    print(f"\n{'='*66}\nSWEEP módů kompozice mountu — {n} položek\n{'='*66}")
    res = {}
    for name, er in SWEEP.items():
        a.entity_retrieval = dict(er)
        rows = run(a, items)
        npass, coll_ans, coll_doc = summarize(rows, label=f"[{name}] ")
        res[name] = rows
    # scoreboard = default (union) běh + porovnání módů
    mode_summary = [(m, sum(1 for _, ok, _, _, _ in res[m] if ok), n) for m in SWEEP]
    test_report.write_scoreboard(
        DOCS, "Poslední test — doménový etalon", _norm(res["union"]),
        subtitle="Collision (routing scoreboard) + coverage (per-entita). Sweep přes módy "
                 "kompozice mountu; per-otázka tabulka je pro default mode = union.",
        config_label="mount mode = union (default)", timestamp=_stamp(),
        source=f"gold_domain.json · {n} otázek", mode_summary=mode_summary)
    print(f"\n→ {DOCS}")
    # kde se módy liší (rozhodovací materiál pro finální mode)
    print(f"\n{'='*66}\n-- KDE SE MÓDY LIŠÍ (odpověď/soubor) --")
    ref = res["off"]
    for i, (it, _, _, _, _) in enumerate(ref):
        sig = {m: (res[m][i][1], res[m][i][2], res[m][i][3]) for m in SWEEP}
        oks = {m: sig[m][0] for m in SWEEP}
        docs = {m: sig[m][1] for m in SWEEP}
        if len(set(oks.values())) > 1 or len({d for d in docs.values() if d is not None}) > 1:
            tag = it.get("kind", "")
            print(f"  [{tag:11}] {it['q'][:38]:38}  " +
                  "  ".join(f"{m}:{'✓' if oks[m] else '✗'}"
                           f"{'' if docs[m] is None else ('D' if docs[m] else 'd')}" for m in SWEEP))


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        main_sweep()
    else:
        main_single()
