#!/usr/bin/env python3
"""hypothesis-two — SIMULACE: full pipeline (celá smyčka na reálném registru).

  otázka → brána (dokument) → registr (kandidátní fakty s predikátem) →
  light-beam (slova otázky + kontext) přes doc-graf → vybraná odpověď.

Doc-graf se staví z registru (co-occurrence context+answer v rámci faktu).
Validuje: brána trefí dokument, a KONTEXT vybere správnou odpověď (táž otázka,
jiná odpověď dle rozhovoru).

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
    # globální salience (tf·idf přes celý korpus, normalizovaná)
    gsal = {w: sum(tfidf(w, d) for d in lemdocs[w]) for w in lemdocs}
    smax = max(gsal.values()) or 1
    emis = {w: gsal[w] / smax for w in gsal}
    return {"docs": docs, "tfidf": tfidf, "wtot": wtot, "emis": emis}

def gate(C, words):
    out = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in words if C["wtot"].get(w))
           for d in C["docs"]}
    return max(out, key=out.get)

def load_registry():
    reg_by_doc = defaultdict(list)
    with open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl") as f:
        for line in f:
            e = json.loads(line)
            reg_by_doc[e["doc"]].append(e)
    return reg_by_doc

def doc_graph(entries):
    """Co-occurrence graf z faktů dokumentu (context ∪ answers v rámci faktu)."""
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

# (label, predikát, slova otázky, kontext rozhovoru, očekávaná odpověď, očekávaný dok)
CASES = [
    ("Kdo patřil mezi pátečníky? [ctx Čapek]",  "patřit", ["pátečník"], ["Čapek"],     "Čapek",     "wiki_karel_čapek"),
    ("Kdo patřil mezi pátečníky? [ctx prezident]","patřit", ["pátečník"], ["prezident"], "prezident", "wiki_karel_čapek"),
]

def main():
    C = build_corpus()
    reg = load_registry()
    graphs = {}
    passed = 0
    for label, pred, qw, ctx, expect_ans, expect_doc in CASES:
        d = gate(C, [pred] + qw)
        if d not in graphs:
            graphs[d] = doc_graph(reg[d])
        nbr = graphs[d]
        # kandidáti = answer lemmata faktů v dok s tímto predikátem
        cands = set()
        for e in reg[d]:
            if e["predicate"] == pred:
                cands.update(a["lemma"] for a in e["answers"])
        act = beam(nbr, C["emis"], [pred] + qw + ctx)
        ranked = sorted(cands, key=lambda w: -act.get(w, 0))
        top = ranked[0] if ranked else None
        ok_doc = d == expect_doc
        ok_ans = top == expect_ans
        passed += ok_doc and ok_ans
        print(f"{'✓' if ok_doc and ok_ans else '✗'} {label}")
        print(f"    brána → {d} {'✓' if ok_doc else '✗ (čekáno '+expect_doc+')'}")
        print(f"    kandidáti (top5 dle aktivace): "
              + ", ".join(f"{w}:{act.get(w,0):.2f}" for w in ranked[:5]))
        print(f"    odpověď → {top}   {'✓' if ok_ans else '✗ (čekáno '+expect_ans+')'}")
    print(f"\n=== full-pipeline: {passed}/{len(CASES)} ===")

if __name__ == "__main__":
    main()
