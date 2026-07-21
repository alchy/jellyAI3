#!/usr/bin/env python3
"""hypothesis-two — SROVNÁNÍ r=1 vs r=2: frame-overlap otázka ↔ identifikovaný fakt.

Pro každou otázku: pipeline (gate + content+predikát) identifikuje fakt; pak se
spočítá STRUKTURNÍ shoda rámů (bez modality) mezi otázkou a faktem při r=1 a r=2.
Vyšší r = užší gramatický kontext; kolik rámů přežije = míra jistoty matche.

Vyžaduje běžící UDPipe :8092.  Spuštění: .venv/bin/python .../sim_r_compare.py
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
canon, frame_sig, slot = _run.canon_lemma, _run.frame_sig, _run.slot
from jellyai.normalize import merge_abbreviations
ROOT = "/Users/j/Projects/jellyAI3"
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ"}
STOP = set(_run.LANG["stopwords"]); COPULA = _run.LANG["copula_lemma"]

def udpipe(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8092/parse", data=data,
                                 headers={"Content-Type": "application/json"})
    return merge_abbreviations(json.loads(urllib.request.urlopen(req, timeout=10).read())["sentences"])

def q_analyze(text):
    toks = [t for s in udpipe(text) for t in s]
    words, pred, cop = [], None, False
    for t in toks:
        lem = canon(t)
        if lem.lower() == COPULA: cop = True
        if t["upos"] == "VERB" and pred is None and lem.lower() not in STOP: pred = lem
        if t["upos"] in CONTENT and lem.lower() not in STOP: words.append(lem)
    if pred is None and cop: pred = COPULA
    return toks, list(dict.fromkeys(words)), pred

def frames(toks, r):
    """Množina STRUKTURNÍCH rámů (bez modality) obsahových pozic."""
    out = set()
    for i, t in enumerate(toks):
        if t["upos"] in CONTENT:
            out.add(frame_sig(toks, i, "", r))    # prázdná modalita = jen gramatika
    return out

def build_corpus():
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    dc = defaultdict(Counter)
    for (doc, _si), rec in ann.items():
        for s in rec["sentences"]:
            for t in s:
                if t["upos"] != "PUNCT": dc[doc][canon(t)] += 1
    docs = list(dc); lemdocs = defaultdict(set)
    for d, c in dc.items():
        for lem in c: lemdocs[lem].add(d)
    D = len(docs)
    idf = lambda w: math.log((D + 1) / (len(lemdocs.get(w, ())) + 1))
    tfidf = lambda w, d: dc[d][w] * idf(w)
    wtot = {w: sum(tfidf(w, d) for d in lemdocs[w]) for w in lemdocs}
    return {"docs": docs, "tfidf": tfidf, "wtot": wtot, "idf": idf, "ann": ann}

def fact_tokens(ann, doc, si, text):
    rec = ann.get((doc, si))
    if not rec: return []
    for s in rec["sentences"]:
        if " ".join(t["form"] for t in s)[:40] == text[:40]:
            return s
    return rec["sentences"][0] if rec["sentences"] else []

QUESTIONS = [
    "Kdo patřil mezi pátečníky?", "Kdo byl Karel Čapek?", "Kdo napsal Bílou nemoc?",
    "Kdo napsal R.U.R.?", "Kdo je Božena Němcová?", "Kdo objevil mloky?",
    "Co kázal Ježíš?", "Co řekl Ježíš Janovi?", "Kdo je Ježíš?",
    "Kdo je Jan Neruda?", "Kdo napsal Válku s mloky?", "Kde se narodil Karel Čapek?",
]

def main():
    C = build_corpus(); ann = C["ann"]
    reg = defaultdict(list)
    for line in open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl"):
        e = json.loads(line); reg[e["doc"]].append(e)
    print(f"{'otázka':30} {'odpověď (fakt, zkráceno)':44} {'r1':>4} {'r2':>4}")
    print("─" * 88)
    for q in QUESTIONS:
        qtoks, qw, pred = q_analyze(q)
        gate = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in qw if C["wtot"].get(w))
                for d in C["docs"]}
        doc = max(gate, key=gate.get)
        def score(e):
            s = sum(C["idf"](w) for w in qw if w in set(e["context"]))
            return s + (3.0 if pred and e["predicate"] == pred else 0)
        facts = sorted(reg[doc], key=score, reverse=True)
        best = facts[0] if facts and score(facts[0]) > 0 else None
        if best is None:
            print(f"{q:30} {'—':44} {'-':>4} {'-':>4}"); continue
        ftoks = fact_tokens(ann, best["doc"], best["sent"], best["text"])
        q1, f1 = frames(qtoks, 1), frames(ftoks, 1)
        q2, f2 = frames(qtoks, 2), frames(ftoks, 2)
        sh1, sh2 = len(q1 & f1), len(q2 & f2)
        print(f"{q:30} {best['text'][:44]:44} {sh1:>4} {sh2:>4}")

if __name__ == "__main__":
    main()
