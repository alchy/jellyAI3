#!/usr/bin/env python3
"""hypothesis-two — SIMULACE: přirozené otázky celou pipeline.

Reálné otázky („Kdo byl Karel Čapek?") → UDPipe (:8092) lemmatizace → obsahová
slova → ① brána (dokument) → ② identifikace faktu (content-overlap) → ③ ODPOVĚĎ
= CELÁ VĚTA faktu. Přesný tvar (syntetická odpověď) je další krok.

Vyžaduje běžící UDPipe na :8092.
Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_ask.py
"""
import sys, json, math, importlib.util, pickle, urllib.request
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
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ"}
STOP = {"kdo", "co", "být", "který", "jaký", "kde", "kdy"}

from jellyai.normalize import merge_abbreviations

def udpipe(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8092/parse", data=data,
                                 headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=10).read())["sentences"]
    return merge_abbreviations(raw)     # R . U . R . → R.U.R. (jako v korpusu)

def q_words(text):
    """Obsahová slova otázky v základním tvaru + predikát (VERB, nebo „být" u spony)."""
    words, pred, cop = [], None, False
    for sent in udpipe(text):
        for t in sent:
            lem = canon(t)
            if lem.lower() == "být":
                cop = True                     # spona: „Kdo je/byl X"
            if t["upos"] == "VERB" and pred is None and lem.lower() not in STOP:
                pred = lem
            if t["upos"] in CONTENT and lem.lower() not in STOP:
                words.append(lem)
    if pred is None and cop:
        pred = "být"                           # copula-otázka → predikát „být"
    return list(dict.fromkeys(words)), pred

def build_corpus():
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    dc = defaultdict(Counter)
    for (doc, _si), rec in ann.items():
        for sent in rec["sentences"]:
            for t in sent:
                if t["upos"] != "PUNCT":
                    dc[doc][canon(t)] += 1
    docs = list(dc)
    lemdocs = defaultdict(set)
    for d, c in dc.items():
        for lem in c:
            lemdocs[lem].add(d)
    D = len(docs)
    idf = lambda w: math.log((D + 1) / (len(lemdocs.get(w, ())) + 1))
    tfidf = lambda w, d: dc[d][w] * idf(w)
    wtot = {w: sum(tfidf(w, d) for d in lemdocs[w]) for w in lemdocs}
    return {"docs": docs, "tfidf": tfidf, "wtot": wtot, "idf": idf}

QUESTIONS = [
    "Kdo patřil mezi pátečníky?",
    "Kdo byl Karel Čapek?",
    "Kdo napsal Bílou nemoc?",
    "Kdo napsal R.U.R.?",
    "Kdo je Božena Němcová?",
    "Kdo objevil mloky?",
    # tvrdší: Ježíš je ve všech evangeliích (nízké idf), různé typy díry, spona
    "Co kázal Ježíš?",
    "Co řekl Ježíš Janovi?",
    "Kdo je Ježíš?",
]

def main():
    C = build_corpus()
    reg = defaultdict(list)
    for line in open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl"):
        e = json.loads(line); reg[e["doc"]].append(e)

    for q in QUESTIONS:
        qw, pred = q_words(q)
        gate = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in qw if C["wtot"].get(w))
                for d in C["docs"]}
        doc = max(gate, key=gate.get)
        # identifikace faktu: content-overlap slov otázky (idf-váženo) + shoda PREDIKÁTU
        # (predikát je z context vyloučen, ale rozliší belonging × naming fakt)
        def score(e):
            ctx = set(e["context"])
            s = sum(C["idf"](w) for w in qw if w in ctx)
            if pred and e["predicate"] == pred:
                s += 3.0
            return s
        facts = sorted(reg[doc], key=score, reverse=True)
        best = facts[0] if facts and score(facts[0]) > 0 else None
        print(f"Q: {q}")
        print(f"   slova: {qw}  · predikát: {pred}")
        print(f"   ① brána → {doc.split('_', 1)[-1]}  ({gate[doc]:.2f})")
        if best:
            print(f"   ② fakt (match {score(best):.1f})")
            print(f"   A: {best['text']}")
        else:
            print(f"   ② žádný fakt nenamatchoval (v dok chybí, nebo predikát je spona)")
        print()

if __name__ == "__main__":
    main()
