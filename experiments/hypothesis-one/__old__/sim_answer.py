#!/usr/bin/env python3
"""hypothesis-two — SIMULACE: výběr odpovědi light-beamem + polish syntetických hran.

Validuje jádro modelu: seed = reálná slova otázky (základní tvar) + kontext
rozhovoru; aktivace se rozteče přes graf; vybere se nejvíc rozsvícený answer-slot.

POLISH: porovná dvě varianty grafu —
  (A) jen bigramový řetěz  → slova otázky dosvítí na odpověď SLABĚ (dlouhý řetěz)
  (B) + zhmotněné syntetické Q→A hrany (kontext faktu → answer-sloty, silná váha)
      → slova otázky rozsvítí answer-sloty PŘÍMO a silně; kontext pak jen rozhodne.

Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_answer.py
"""
import sys, importlib.util
from collections import defaultdict

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb
vb.serve = lambda *a, **k: None
_spec = importlib.util.spec_from_file_location(
    "h1run", "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/run.py")
_run = importlib.util.module_from_spec(_spec); sys.modules["h1run"] = _run
_spec.loader.exec_module(_run)

SYN_W = 8.0            # váha zhmotněné syntetické hrany (silnější než bigram)

def neighbors(adj):
    nbr = defaultdict(list)
    for (a, b), w in adj.items():
        nbr[a].append((b, w)); nbr[b].append((a, w))
    return nbr

def beam(nbr, emis, W, seed, hops=4, decay=0.6):
    act = {x: 1.0 for x in seed if x in W}
    fro = dict(act)
    for h in range(hops):
        nx = defaultdict(float)
        for nd, a in fro.items():
            ws = sum(w for _, w in nbr[nd]) or 1
            for mm, w in nbr[nd]:
                nx[mm] += a * (w / ws) * (decay ** (h + 1)) * (0.4 + 0.6 * emis.get(mm, 0))
        for k, v in nx.items():
            act[k] = act.get(k, 0) + v
        fro = nx
    return act

def main():
    g = _run.build()
    W, tfmax = g["W"], g["tfmax"]
    emis = {l: W[l]["tfidf"] / tfmax for l in W}

    # fakt „Mezi pátečníky patřili …": answer-sloty + kotva (mezi-skupina)
    ANCHOR = "pátečník"                       # distinktivní slovo otázky = kotva faktu
    ANSWERS = ["bratr", "Čapek", "prezident"]  # fillery answer-slotu
    CANDS = {"bratři Čapkové": ["Čapek", "bratr"], "prezident Masaryk": ["prezident"]}

    # (A) jen bigram
    nbrA = neighbors(dict(g["adj"]))
    # (B) + zhmotněné syntetické hrany kotva ↔ answer-sloty
    adjB = dict(g["adj"])
    for ans in ANSWERS:
        adjB[(ANCHOR, ans)] = SYN_W
    nbrB = neighbors(adjB)

    def score(act, cand): return sum(act.get(l, 0) for l in cand)
    def show(nbr, seed):
        act = beam(nbr, emis, W, seed)
        return {lab: round(score(act, lm), 3) for lab, lm in CANDS.items()}

    QW = ["pátečník", "patřit", "mezi"]        # reálná slova otázky (základní tvar)

    print("① POLISH: seed = jen SLOVA OTÁZKY (bez kontextu) — dosvítí na odpověď?\n")
    a = show(nbrA, QW); b = show(nbrB, QW)
    print(f"   (A) jen bigram:        {a}")
    print(f"   (B) + syntetické hrany:{b}")
    lift = (max(b.values()) / max(max(a.values()), 1e-9))
    print(f"   → zesílení nejjasnějšího kandidáta: {lift:.0f}×\n")

    print("② VALIDACE výběru: seed = slova otázky + KONTEXT rozhovoru (graf B)\n")
    cases = [("bez kontextu", [], None),
             ("Karel Čapek", ["Čapek"], "bratři Čapkové"),
             ("prezidenti", ["prezident"], "prezident Masaryk")]
    ok = 0
    for label, ctx, expect in cases:
        r = show(nbrB, QW + ctx)
        win = max(r, key=r.get)
        verdict = ""
        if expect is not None:
            good = win == expect
            ok += good
            verdict = "✓" if good else f"✗ (čekáno {expect})"
        print(f"   {label:14} → {r}   ★ {win}   {verdict}")
    print(f"\n=== kontextový výběr: {ok}/2 správně ===")

if __name__ == "__main__":
    main()
