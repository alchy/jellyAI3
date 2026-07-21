#!/usr/bin/env python3
"""hypothesis-two — BATCH: qwen generuje parafráze otázek + šablonu odpovědi
(fragment i celá věta) a zpětně verifikuje; pipeline testuje self-consistency.

Pro ≥50 faktů z různých žánrů/dokumentů:
  TAM   — qwen z faktu+odpovědi vyrobí parafráze otázek + odpověď jako větu
  PIPE  — každou variantu prožene naše pipeline (brána+fakt) → trefí zdrojový fakt?
  ZPĚT  — qwen ověří, že odpověď je věcně správná
Průběžný zápis do variants.jsonl. Věty na '?' se NEberou jako zdroj otázek.

Vyžaduje Ollama :11434 (qwen3.6) + UDPipe :8092.
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
import ollama_iface as OL
from jellyai.normalize import merge_abbreviations
ROOT = "/Users/j/Projects/jellyAI3"
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ"}
STOP = set(_run.LANG["stopwords"]); COPULA = _run.LANG["copula_lemma"]

def udpipe(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8092/parse", data=data,
                                 headers={"Content-Type": "application/json"})
    return [t for s in merge_abbreviations(json.loads(
        urllib.request.urlopen(req, timeout=10).read())["sentences"]) for t in s]

def q_analyze(text):
    toks = udpipe(text); words, pred, cop = [], None, False
    for t in toks:
        lem = canon(t)
        if lem.lower() == COPULA: cop = True
        if t["upos"] == "VERB" and pred is None and lem.lower() not in STOP: pred = lem
        if t["upos"] in CONTENT and lem.lower() not in STOP: words.append(lem)
    if pred is None and cop: pred = COPULA
    return list(dict.fromkeys(words)), pred

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
    return {"docs": docs, "tfidf": tfidf, "wtot": wtot, "idf": idf}

def pipeline(C, reg, qtext):
    qw, pred = q_analyze(qtext)
    gate = {d: sum((C["tfidf"](w, d) / C["wtot"][w]) for w in qw if C["wtot"].get(w))
            for d in C["docs"]}
    doc = max(gate, key=gate.get) if gate else None
    def score(e):
        s = sum(C["idf"](w) for w in qw if w in set(e["context"]))
        return s + (3.0 if pred and e["predicate"] == pred else 0)
    facts = sorted(reg.get(doc, []), key=score, reverse=True)
    best = facts[0] if facts and score(facts[0]) > 0 else None
    return doc, best

ROLES = ("who", "what_subject", "state")     # sjednocené klíče katalogu (podmět/predikativ)

def select(reg, docs, K):
    """Diverzní výběr: fakty o PROTAGONISTECH (opakující se entita v answer-slotu),
    obsahově bohaté, střední délky, NE otázky ('?'). Round-robin napříč dokumenty
    kvůli žánrové pestrosti. Answer = nejfrekventovanější entita faktu."""
    ent_df = defaultdict(Counter)
    for doc in reg:
        for e in reg[doc]:
            for a in e["answers"]:
                if a["name"] and a["role"] in ROLES:
                    ent_df[doc][a["lemma"]] += 1
    per_doc = {}
    for doc in docs:
        prot = {ent for ent, c in ent_df[doc].most_common(8) if c >= 3}
        cand = []
        for e in reg[doc]:
            if e["text"].rstrip().endswith("?"): continue
            if not (6 <= len(e["text"].split()) <= 20): continue
            if len(set(e["context"])) < 4: continue
            a = max((a for a in e["answers"]
                     if a["name"] and a["role"] in ROLES and a["lemma"] in prot),
                    key=lambda a: ent_df[doc][a["lemma"]], default=None)
            if a: cand.append((e, a["lemma"]))
        cand.sort(key=lambda ea: len(set(ea[0]["context"])), reverse=True)
        per_doc[doc] = cand
    picks, idx = [], 0
    while len(picks) < K and any(idx < len(per_doc[d]) for d in docs):
        for doc in docs:
            if idx < len(per_doc[doc]):
                picks.append(per_doc[doc][idx])
                if len(picks) >= K: break
        idx += 1
    return picks

def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    NV = 2
    C = build_corpus()
    reg = defaultdict(list)
    for line in open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl"):
        e = json.loads(line); reg[e["doc"]].append(e)
    picks = select(reg, C["docs"], K)

    out = open(f"{ROOT}/experiments/hypothesis-one/variants.jsonl", "w")
    n_q = self_ok = ver_ok = 0
    pos_tot = Counter(); pos_ok = Counter()      # self-consistency dle pozice varianty
    for i, (e, ans) in enumerate(picks, 1):
        try:
            variants = OL.gen_questions(e["text"], ans, NV)
            asent = OL.gen_answer(variants[0], e["text"]) if variants else ""
        except Exception as exc:      # noqa
            print(f"{i}: Ollama chyba {exc}", flush=True); continue
        rec = {"doc": e["doc"], "fact": e["text"], "answer_fragment": ans,
               "answer_sentence": asent, "variants": []}
        for pos, q in enumerate(variants):
            n_q += 1
            _, best = pipeline(C, reg, q)
            hit = best is not None and best["sent"] == e["sent"] and best["doc"] == e["doc"]
            self_ok += hit; pos_tot[pos] += 1; pos_ok[pos] += hit
            rec["variants"].append({"q": q, "self_consistent": hit})
        try:
            v = OL.verify(variants[0], ans) if variants else False
        except Exception:
            v = None
        ver_ok += bool(v)
        rec["verify_ok"] = v
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(f"{i}/{len(picks)} [{e['doc'].split('_')[-1]}] ans={ans} "
              f"var={len(variants)} self={sum(x['self_consistent'] for x in rec['variants'])}"
              f"/{len(variants)} verify={v}", flush=True)
    out.close()
    print(f"\n=== otázek: {n_q} · self-consistent: {self_ok}/{n_q} "
          f"({100*self_ok/max(n_q,1):.0f}%) · verify ANO: {ver_ok}/{len(picks)} ===", flush=True)
    for pos in sorted(pos_tot):
        lbl = "blízká parafráze (v1)" if pos == 0 else f"volnější parafráze (v{pos+1})"
        print(f"    {lbl:26} self-consistent {pos_ok[pos]}/{pos_tot[pos]} "
              f"({100*pos_ok[pos]/max(pos_tot[pos],1):.0f}%)", flush=True)

if __name__ == "__main__":
    main()
