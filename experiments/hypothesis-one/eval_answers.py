#!/usr/bin/env python3
"""Gold otázka→odpověď — BASELINE harness (K0 fúze).

Změří end-to-end přesnost DNEŠNÍHO VZOR matcheru (answering.Answering) na ručním gold
setu (gold_answers.json), alias-aware. Pro každý tah:
  • VÝSLEDEK po fázích — kde tah spadl: no_qvzor / no_hot / no_cand / WRONG / HIT.
  • STROP — vynutí se gold soubor a matchne query-VZOR: je správná odpověď VŮBEC mezi
    kandidáty? Odděluje chybu VÝBĚRU souboru (select_files) vs EXISTENCE šablony
    (vzor transpozice) vs ŘAZENÍ (glow-orders-ties) vs jen špatní kandidáti.
  • EXTRAKCE — je gold odpověď faktem v registry pro ten soubor? (extrakce vs struktura)

Tvrdé číslo, na kterém další fúzní kroky prokazují zisk. Deterministické (per-tah čisté
pole). Runtime parse otázky jde přes UDPipe 2 (jediná živá anotace) — vyžaduje síť.
"""
import os
import json
from collections import Counter, defaultdict

from answering import Answering
from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold_answers.json")
REG = os.path.join(HERE, "registry.jsonl")
OUT = os.path.join(HERE, "baseline_answers.json")


def norm(s):
    return (s or "").strip().lower()


def accept(got, answers):
    """Alias-aware: vrácený lemma (lower) == některý povrch, nebo podřetězec (obojí směr)."""
    a = norm(got)
    if not a:
        return False
    for g in answers:
        gg = norm(g)
        if a == gg or gg in a or a in gg:
            return True
    return False


def registry_index():
    """{doc: set(answer-lemma lower)} — co extrakce vytěžila (diagnostika extrakce)."""
    idx = defaultdict(set)
    for line in open(REG, encoding="utf-8"):
        e = json.loads(line)
        for x in e["answers"]:
            if x.get("lemma"):
                idx[e["doc"]].add(x["lemma"].lower())
    return idx


def stage_trace(a, toks):
    """Fázová viditelnost přes STEJNÁ primitiva jako answering.answer() (_candidates/_assurance)."""
    q_vzor, lemmas, hole_role = a._question(toks)
    hole_role = a._answer_role(toks, hole_role)         # kopula → komplement (state)
    predicate = a._predicate(toks)
    if not (q_vzor or (predicate and hole_role)):
        return {"stage": "no_qvzor"}
    hot = [d for d, _s in a.dl.select_files(lemmas)]
    if not hot:
        return {"stage": "no_hot"}
    a.store.mount(hot)
    if a.facts_enabled:
        a.facts.mount(hot)
    a.field.build_graph(a.dl.mount(hot), a.g)
    a.field.feed(lemmas, a.dl)
    a.field.spread()
    cands = a._candidates(q_vzor, predicate, hole_role, {l.lower() for l in lemmas})
    if not cands:
        return {"stage": "no_cand", "hot": hot}
    mode, best, offer = a._assurance(cands, lemmas, hole_role)   # jasno vs nejasno (sdílené)
    return {"stage": "answer", "answer": best["answer"], "mode": mode,
            "offer": [c["answer"] for c in offer], "cands": len(cands), "hot": hot, "via": best["kind"]}


def stage_ceiling(a, toks, it):
    """Strop: vynutí gold soubor, sesbírej kandidáty (obě cesty) — je odpověď dosažitelná?"""
    q_vzor, _lemmas, hole_role = a._question(toks)
    hole_role = a._answer_role(toks, hole_role)
    predicate = a._predicate(toks)
    a.store.mounted.clear()
    a.facts.mounted.clear()
    a.store.mount([it["doc"]])
    if a.facts_enabled:
        a.facts.mount([it["doc"]])
    cands = a._candidates(q_vzor, predicate, hole_role, {a.g.canon_lemma(t).lower()
                                                         for t in toks if t["upos"] != "PUNCT"})
    if not cands:
        return "no_template"
    return "retrievable" if any(accept(c["answer"], it["answer"]) for c in cands) else "wrong_only"


def _clear(a):
    """Čisté pole pro nezávislý per-tah zdroj světla (bez kontaminace mezi otázkami)."""
    a.store.mounted.clear()
    a.facts.mounted.clear()
    a.field.words.clear()
    a.field.files.clear()
    a.field.adj.clear()


def main():
    a = Answering()
    reg = registry_index()
    gold = json.load(open(GOLD, encoding="utf-8"))["items"]
    rows = []
    outc = Counter()
    ceilc = Counter()
    role_tot = Counter()
    role_hit = Counter()
    kind_tot = Counter()
    kind_hit = Counter()
    hits = 0
    fhits = fn = 0                                  # faktický etalon (bez teologických)
    for it in gold:
        _clear(a)
        toks = a._parse(it["q"])
        t = stage_trace(a, toks)
        got = t.get("answer")
        offer = t.get("offer", [])
        offer_has = any(accept(o, it["answer"]) for o in offer)
        if t["stage"] == "answer":
            mode = t.get("mode", "answer")
            if mode == "answer":
                outcome = "HIT" if accept(got, it["answer"]) else "WRONG"
            elif mode == "clarify":
                outcome = "CLARIFY"
            else:
                outcome = "UNSURE"
        elif t["stage"] == "no_cand":
            outcome = "NO_CAND"
        elif t["stage"] == "no_hot":
            outcome = "NO_HOT"
        else:
            outcome = "NO_QVZOR"
        if outcome == "HIT":
            hits += 1
        if it.get("factual", True):                # teologické/nefaktické z etalonu vyňaty (user)
            fn += 1
            if outcome == "HIT":
                fhits += 1
        ceil = stage_ceiling(a, toks, it)
        extracted = any(any(norm(g) == al or norm(g) in al for al in reg.get(it["doc"], set()))
                        for g in it["answer"])
        outc[outcome] += 1
        ceilc[ceil] += 1
        role = it.get("role", "?")
        kind = it.get("kind", "?")
        role_tot[role] += 1
        kind_tot[kind] += 1
        if outcome == "HIT":
            role_hit[role] += 1
            kind_hit[kind] += 1
        rows.append({"q": it["q"], "role": role, "kind": kind, "expect": it["answer"],
                     "got": got, "outcome": outcome, "ceiling": ceil, "via": t.get("via", "—"),
                     "offer": offer, "offer_has": offer_has,
                     "extracted": extracted, "cands": t.get("cands", 0), "doc": it["doc"]})
    n = len(gold)
    acc = hits / n if n else 0.0
    conf_wrong = sum(1 for r in rows if r["outcome"] == "WRONG")
    clar = sum(1 for r in rows if r["outcome"] == "CLARIFY")
    clar_good = sum(1 for r in rows if r["outcome"] == "CLARIFY" and r["offer_has"])
    facc = fhits / fn if fn else 0.0
    print(f"\n=== FAKTICKÝ ETALON {fhits}/{fn} = {facc*100:.0f}% (bez {n - fn} teologických) | "
          f"vše {hits}/{n} | sebevědomě ŠPATNĚ {conf_wrong} | doptá se {clar} ===\n")
    mark = {"HIT": "✓", "WRONG": "×", "CLARIFY": "⁇", "UNSURE": "∅",
            "NO_CAND": "·", "NO_HOT": "○", "NO_QVZOR": "?"}
    for r in rows:
        if r["outcome"] == "CLARIFY":
            resp = "doptat: " + ", ".join(str(o) for o in r["offer"][:3])
        elif r["outcome"] in ("HIT", "WRONG"):
            resp = "→ " + str(r["got"] or "—")
        else:
            resp = "(" + r["outcome"].lower() + ")"
        print(f"{mark.get(r['outcome'],'?'):1} {r['q']:31}{resp[:34]:35} [chtěli {'/'.join(r['expect'][:2])}]")
    print("\n-- módy --", dict(outc))
    print("-- strop    --", dict(ceilc))
    print("-- dle role --", {k: f"{role_hit[k]}/{role_tot[k]}" for k in sorted(role_tot)})
    print("-- dle kind --", {k: f"{kind_hit[k]}/{kind_tot[k]}" for k in sorted(kind_tot)})
    logger("i", f"gold baseline {hits}/{n} ({acc*100:.0f}%); výsl {dict(outc)}; strop {dict(ceilc)}")
    json.dump({"accuracy": acc, "hits": hits, "n": n, "outcomes": dict(outc),
               "ceiling": dict(ceilc),
               "by_role": {k: [role_hit[k], role_tot[k]] for k in role_tot},
               "by_kind": {k: [kind_hit[k], kind_tot[k]] for k in kind_tot},
               "rows": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
