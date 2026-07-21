#!/usr/bin/env python3
"""hypothesis-two — SIMULACE: full pipeline (celá smyčka na reálném registru).

Správné pořadí (oprava — kontext se NEcpe předčasně):
  ① otázka → BRÁNA (dokument, ze slov otázky)
  ② IDENTIFIKACE FAKTU (content-match slov otázky proti faktům s predikátem)
  ③ ODPOVĚĎ = answer-sloty identifikovaného faktu  (množina belongerů)
  ④ jen když je belongerů VÍC a je kontext → light-beam rozhodne MEZI nimi

Fakt i množina odpovědí vznikají ze SLOV OTÁZKY; kontext rozhoduje až uvnitř.

Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_pipeline.py
"""
import sys, json, math, importlib.util, pickle
from collections import defaultdict, Counter

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb
vb.serve = lambda *a, **k: None
_spec = importlib.util.spec_from_file_location(
    "h1run", "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/run.py")
_run = importlib.util.module_from_spec(_spec); sys.modules["h1run"] = _run
_spec.loader.exec_module(_run)
canon = _run.canon_lemma
ROOT = "/Users/j/Projects/jellyAI3"

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
    for d, c in doc_cnt.items():
        for lem in c:
            lemdocs[lem].add(d)
    D = len(docs)
    idf = lambda w: math.log((D + 1) / (len(lemdocs.get(w, ())) + 1))
    tfidf = lambda w, d: doc_cnt[d][w] * idf(w)
    wtot = {w: sum(tfidf(w, d) for d in lemdocs[w]) for w in lemdocs}
    smax = max(wtot.values()) or 1
    emis = {w: wtot[w] / smax for w in wtot}
    return {"docs": docs, "tfidf": tfidf, "wtot": wtot, "emis": emis, "idf": idf}

def gate(C, words):
    out = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in words if C["wtot"].get(w))
           for d in C["docs"]}
    return max(out, key=out.get), out

def load_registry():
    reg = defaultdict(list)
    with open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl") as f:
        for line in f:
            e = json.loads(line)
            reg[e["doc"]].append(e)
    return reg

def doc_graph(entries):
    adj = Counter()
    for e in entries:
        words = list(dict.fromkeys(e["context"] + [a["lemma"] for a in e["answers"]]))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                a, b = sorted((words[i], words[j]))
                adj[(a, b)] += 1
    nbr = defaultdict(list)
    for (a, b), w in adj.items():
        nbr[a].append((b, w)); nbr[b].append((a, w))
    return nbr

def beam(nbr, emis, seed, hops=3, decay=0.6):
    act = {x: 1.0 for x in seed if x in nbr}
    fro = dict(act)
    for _ in range(hops):
        nx = defaultdict(float)
        for nd, a in fro.items():
            ws = sum(w for _, w in nbr[nd]) or 1
            for mm, w in nbr[nd]:
                nx[mm] += a * (w / ws) * decay * (0.4 + 0.6 * emis.get(mm, 0))
        for k, v in nx.items():
            act[k] = act.get(k, 0) + v
        fro = nx
    return act

# (label, predikát, slova otázky, kontext, očekávaná odpověď | None=množina, dok)
CASES = [
    ("Kdo patřil mezi pátečníky?  [BEZ kontextu]", "patřit", ["pátečník"], [],           None,        "wiki_karel_čapek"),
    ("   … tentýž dotaz  [ctx Čapek]",              "patřit", ["pátečník"], ["Čapek"],     "Čapek",     "wiki_karel_čapek"),
    ("   … tentýž dotaz  [ctx prezident]",          "patřit", ["pátečník"], ["prezident"], "prezident", "wiki_karel_čapek"),
]

def main():
    C = build_corpus()
    reg = load_registry()
    graphs = {}
    passed = 0
    for label, pred, qw, ctx, expect, expect_doc in CASES:
        # ① brána (jen slova otázky)
        d, _ = gate(C, [pred] + qw)
        # ② identifikace faktu (content-match slov otázky)
        facts = [e for e in reg[d] if e["predicate"] == pred]
        facts.sort(key=lambda e: sum(C["idf"](w) for w in qw if w in e["context"]),
                   reverse=True)
        fakt = facts[0] if facts else None
        # ③ answer-sloty faktu (bez skupiny = slova otázky)
        belong = [a["lemma"] for a in (fakt["answers"] if fakt else [])
                  if a["lemma"] not in qw]
        belong = list(dict.fromkeys(belong))
        # ④ kontext rozhodne MEZI belongery (jen když je jich víc a je kontext)
        if ctx and len(belong) > 1:
            if d not in graphs:
                graphs[d] = doc_graph(reg[d])
            act = beam(graphs[d], C["emis"], qw + ctx)
            belong = sorted(belong, key=lambda w: -act.get(w, 0))
        winner = belong[0] if belong else None

        ok_doc = d == expect_doc
        if expect is None:                       # bez kontextu: ověř FAKT + množinu
            ok = ok_doc and fakt and "pátečníky" in fakt["text"]
            passed += bool(ok)
            print(f"{'✓' if ok else '✗'} {label}")
            print(f"     ① brána → {d}   ② fakt → „{fakt['text'][:60]}…\"")
            print(f"     ③ odpověď (množina belongerů): {', '.join(belong)}  [BEZ kontextu]")
        else:                                    # s kontextem: ověř vítěze
            ok = ok_doc and winner == expect
            passed += bool(ok)
            print(f"{'✓' if ok else '✗'} {label}")
            print(f"     ④ vybráno z {belong}  →  {winner}  {'✓' if winner==expect else '✗'}")
    print(f"\n=== full-pipeline: {passed}/{len(CASES)} ===")

if __name__ == "__main__":
    main()
