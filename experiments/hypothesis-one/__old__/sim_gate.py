#!/usr/bin/env python3
"""hypothesis-two — SIMULACE: validace dokumentové aktivační brány.

Ověřuje, že slova otázky (v základním tvaru) otevřou správný dokument přes jeho
klíčová slova (per-dokument tf·idf, normalizovaný podíl). Cross-dokument (kosinus
klíčových slov) = kandidáti na swap-in (plynulé pohasínání, ne cut-off).

Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_gate.py
"""
import sys, pickle, math
from collections import defaultdict, Counter

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb  # noqa (jen kvůli importu run.py, který ho čeká)
vb.serve = lambda *a, **k: None
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "h1run", "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/run.py")
_run = importlib.util.module_from_spec(_spec); sys.modules["h1run"] = _run
_spec.loader.exec_module(_run)
canon = _run.canon_lemma
ROOT = "/Users/j/Projects/jellyAI3"

# --- korpusový model (per-dokument klíčová slova) ---------------------------
def build_corpus():
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    doc_cnt = defaultdict(Counter)
    for (doc, _si), rec in ann.items():
        for sent in rec["sentences"]:
            for t in sent:
                if t["upos"] != "PUNCT":
                    doc_cnt[doc][canon(t)] += 1
    docs = list(doc_cnt)
    lemdocs = defaultdict(set)
    for doc, c in doc_cnt.items():
        for lem in c:
            lemdocs[lem].add(doc)
    D = len(docs)
    idf = lambda w: math.log((D + 1) / (len(lemdocs.get(w, ())) + 1))
    tfidf = lambda w, d: doc_cnt[d][w] * idf(w)
    wtot = {w: sum(tfidf(w, d) for d in lemdocs[w]) for w in lemdocs}
    return {"docs": docs, "doc_cnt": doc_cnt, "idf": idf, "tfidf": tfidf,
            "wtot": wtot, "lemdocs": lemdocs}

def gate(C, qwords):
    """Aktivace dokumentů: každé slovo rozdělí svůj podíl dle klíčovosti (norm.)."""
    out = {}
    for d in C["docs"]:
        out[d] = sum((C["tfidf"](w, d) / C["wtot"][w])
                     for w in qwords if C["wtot"].get(w))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))

def cross(C, doc, k=3):
    """Nejbližší dokumenty (kosinus klíčových slov) = swap-in kandidáti."""
    def vec(d): return {w: C["tfidf"](w, d) for w in C["doc_cnt"][d]}
    vd = vec(doc); nd = math.sqrt(sum(v * v for v in vd.values())) or 1
    res = []
    for o in C["docs"]:
        if o == doc:
            continue
        vo = vec(o); no = math.sqrt(sum(v * v for v in vo.values())) or 1
        num = sum(vd[w] * vo.get(w, 0) for w in vd)
        res.append((o, num / (nd * no)))
    return sorted(res, key=lambda x: -x[1])[:k]

# --- testovací případy (slovo otázky v základu → očekávaný dokument) --------
CASES = [
    ("pátečníci",        ["pátečník"],          "wiki_karel_čapek"),
    ("mloci",            ["mlok"],              "wiki_válka_s_mloky"),
    ("roboti R.U.R.",    ["robot"],             "wiki_r.u.r."),
    ("Babička/Němcová",  ["babička", "Němcová"],"wiki_božena_němcová"),
    ("Neruda fejeton",   ["Neruda", "fejeton"], "wiki_jan_neruda"),
    ("Mojžíš/farao",     ["Mojžíš", "farao"],   "bible_exodus"),
    ("Abraham/Izák",     ["Abraham", "Izák"],   "bible_genesis"),
    ("žalm",             ["žalm"],              "bible_zalmy"),
]

def main():
    C = build_corpus()
    print(f"korpus: {len(C['docs'])} dokumentů\n")
    passed = 0
    for label, qw, expect in CASES:
        present = [w for w in qw if w in C["lemdocs"]]
        g = gate(C, qw)
        top, sc = next(iter(g.items()))
        ok = top == expect
        passed += ok
        miss = [w for w in qw if w not in C["lemdocs"]]
        flag = "✓" if ok else "✗"
        print(f"{flag} {label:18} slova={present}{' MIMO:'+str(miss) if miss else ''}")
        print(f"     brána → {top} ({sc:.2f})   očekáváno {expect}"
              f"{'' if ok else '   ← NESHODA'}")
        if not ok:
            print(f"     pořadí: " + ", ".join(f"{d}:{s:.2f}" for d, s in list(g.items())[:3]))
        sw = cross(C, top)
        print(f"     swap-in kandidáti (cross-dok): "
              + ", ".join(f"{d}:{s:.2f}" for d, s in sw))
    print(f"\n=== {passed}/{len(CASES)} bran otevřelo správný dokument ===")

if __name__ == "__main__":
    main()
