#!/usr/bin/env python3
"""hypothesis-two — SHOWCASE: celý průchod na vzorku otázek napříč korpusem.

Vezme reálné fakty (diverzně po dokumentech), z každého vyrobí otázku
(predikát + nejdistinktivnější slovo faktu) a pustí je celou pipeline:
  ① brána → dokument   ② identifikace faktu   ③ odpověď (belongeři)
Self-consistency: našel pipeline ZPĚT ten fakt, z něhož otázka vznikla?

Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_showcase.py [K]
"""
import sys, json, math, importlib.util, pickle, random
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
    return {"docs": docs, "tfidf": tfidf, "wtot": wtot, "idf": idf}

def gate(C, words):
    out = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in words if C["wtot"].get(w))
           for d in C["docs"]}
    return max(out, key=out.get)

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    C = build_corpus()
    reg_by_doc = defaultdict(list)
    reg = []
    for line in open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl"):
        e = json.loads(line); reg.append(e); reg_by_doc[e["doc"]].append(e)

    def dword(e):     # nejdistinktivnější slovo faktu (mimo answer-jména = klíč otázky)
        cw = [w for w in e["context"] if w not in
              [a["lemma"] for a in e["answers"] if a["name"]]]
        return max(cw, key=C["idf"]) if cw else None

    # vyber diverzně: po dokumentech fakty s nejdistinktivnějším slovem + belongery
    picks = []
    random.seed(7)
    for doc in C["docs"]:
        cand = [e for e in reg_by_doc[doc]
                if len(e["answers"]) >= 1 and dword(e) and C["idf"](dword(e)) > 1.4]
        cand.sort(key=lambda e: C["idf"](dword(e)), reverse=True)
        picks += cand[:2]     # 1–2 nejdistinktivnější fakty per dokument
    random.shuffle(picks)
    picks = picks[:K]

    ok = 0
    for i, e in enumerate(picks, 1):
        pred, dw = e["predicate"], dword(e)
        d = gate(C, [pred, dw])
        facts = [x for x in reg_by_doc[d] if x["predicate"] == pred]
        facts.sort(key=lambda x: sum(C["idf"](w) for w in [dw] if w in x["context"]),
                   reverse=True)
        found = facts[0] if facts else None
        recovered = found is not None and found["sent"] == e["sent"] and found["doc"] == e["doc"]
        belong = list(dict.fromkeys(a["lemma"] for a in e["answers"] if a["lemma"] != dw))
        ok += recovered
        print(f"{i:2} {'✓' if recovered else '✗'} Q: Kdo {pred} … ?   (klíč: {dw!r})")
        print(f"      ① brána → {d.split('_',1)[-1]}    ② fakt: „{e['text'][:62]}…\"")
        print(f"      ③ odpověď: {', '.join(belong[:6])}")
    print(f"\n=== self-consistency (našel zpět svůj fakt): {ok}/{len(picks)} ===")

if __name__ == "__main__":
    main()
