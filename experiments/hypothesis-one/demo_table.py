#!/usr/bin/env python3
"""hypothesis-two — DEMO tabulka: fakt · otázka · odpověď (+ proč případné minutí).

Každou otázku prožene celá pipeline:
  ① brána tf·idf → dokument   ② identifikace faktu (obsah + predikát)
  ③ ODPOVĚĎ = rozklad nalezeného faktu do katalogu rolí (roles.decompose),
     zvýrazněna role, na kterou se otázka ptá (díra)

Když se pattern netrefí, řádek nese DŮVOD. Nemáme zatím VÁHY KONVERZACE
(krátkodobá aktivace slov z dialogu) — u ambivalentních otázek („Kdo je Ježíš?")
je proto výběr THE faktu nerozhodnutelný; to je komentář, ne chyba.

Vyžaduje UDPipe :8092.
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
import roles as R
from jellyai.normalize import merge_abbreviations
canon = _run.canon_lemma
ROOT = "/Users/j/Projects/jellyAI3"
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ"}
STOP = set(_run.LANG["stopwords"]); COPULA = _run.LANG["copula_lemma"]

# tázací slovo (lemma[+pád]) → klíč katalogu (díra otázky); data zrcadlí role_ask
ASK_ROLE = {"kdo": "who", "co": "whom_what", "kde": "where", "kam": "where",
            "odkud": "where", "kdy": "when", "jak": "how", "kolik": "how_much",
            "proč": "why", "čí": "which_attribute", "jaký": "which_attribute",
            "který": "which_attribute"}
ASK_ROLE_CASE = {("kdo", "Ins"): "with_whom_what", ("kdo", "Dat"): "to_whom",
                 ("kdo", "Gen"): "whose_of_what", ("kdo", "Acc"): "whom_what",
                 ("co", "Ins"): "with_whom_what", ("co", "Gen"): "whose_of_what"}

def udpipe(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8092/parse", data=data,
                                 headers={"Content-Type": "application/json"})
    return merge_abbreviations(json.loads(urllib.request.urlopen(req, timeout=10).read())["sentences"])

def q_analyze(text):
    toks = [t for s in udpipe(text) for t in s]
    words, pred, cop, ask = [], None, False, None
    for t in toks:
        lem = canon(t)
        low = lem.lower()
        if low == COPULA: cop = True
        if t["upos"] == "VERB" and pred is None and low not in STOP: pred = lem
        if t["upos"] in CONTENT and low not in STOP: words.append(lem)
        if ask is None and (t["lemma"].lower() in ASK_ROLE or "Int" in (t.get("feats", {}).get("PronType", ""))):
            case = t.get("feats", {}).get("Case", "")
            ask = ASK_ROLE_CASE.get((t["lemma"].lower(), case)) or ASK_ROLE.get(t["lemma"].lower())
    if pred is None and cop: pred = COPULA
    # identita „Kdo byl X?" → ptáme se na stav (predikativ), ne na podmět
    if cop and ask == "who" and any(t["upos"] == "PROPN" for t in toks): ask = "state"
    return list(dict.fromkeys(words)), pred, ask

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

def run(C, reg, q):
    qw, pred, ask = q_analyze(q)
    gate = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in qw if C["wtot"].get(w))
            for d in C["docs"]}
    if not gate or max(gate.values()) == 0:
        return {"ask": ask, "why": "žádné slovo otázky není v korpusu (mimo doménu)"}
    doc = max(gate, key=gate.get)
    def score(e):
        s = sum(C["idf"](w) for w in qw if w in set(e["context"]))
        return s + (3.0 if pred and e["predicate"] == pred else 0)
    facts = sorted(reg[doc], key=score, reverse=True)
    top = score(facts[0]) if facts else 0
    if top <= 0:
        return {"ask": ask, "doc": doc, "why": "nulový překryv obsahu/predikátu → díra v registru"}
    ties = sum(1 for e in facts if abs(score(e) - top) < 1e-9)
    best = facts[0]
    clauses = R.decompose(fact_tokens(C["ann"], best["doc"], best["sent"], best["text"]))
    roles = {}
    for c in clauses:
        for k, v in c["roles"].items():
            if not k.startswith("_"): roles.setdefault(k, []).extend(v)
    ans = roles.get(ask) if ask else None
    return {"ask": ask, "doc": doc, "fact": best["text"], "roles": roles,
            "answer": ans, "ties": ties, "top": top}

QUESTIONS = [
    "Kdo napsal R.U.R.?",
    "Kdo byl Karel Čapek?",
    "Kdo patřil mezi pátečníky?",
    "Co kázal Ježíš?",
    "Kde se narodil Ježíš?",
    "S kým mluvil Hospodin?",
    "Kdo je Ježíš?",                 # ambivalentní — potřebuje váhy konverzace
    "Kdo napsal Hamleta?",           # mimo korpus
    "Jaká je hlavní myšlenka kvantové gravitace?",   # mimo doménu
]

def main():
    C = build_corpus()
    reg = defaultdict(list)
    for line in open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl"):
        e = json.loads(line); reg[e["doc"]].append(e)
    rows = []
    for q in QUESTIONS:
        r = run(C, reg, q)
        rows.append((q, r))
    # výpis strojově čitelně (tabulku vyrenderuje volající)
    for q, r in rows:
        print(json.dumps({"q": q, **r}, ensure_ascii=False))

if __name__ == "__main__":
    main()
