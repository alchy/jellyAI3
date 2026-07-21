#!/usr/bin/env python3
"""hypothesis-two — GENERÁTOR syntetických vazeb fakt ↔ otázka (registr).

Pro každou větu korpusu s predikátem (root VERB) vyrobí syntetickou otázku
„Kdo <predikát> …?" (díra = podmět/účastník) a uloží VAZBU na fakt:
  { predicate, hole, answers[lemma,role,gender], context[lemma], doc, sent, text }

To je data, nad kterými běží match (registr) i light-beam. Answers/context jsou
v ZÁKLADNÍM TVARU (kanonizace přes run.canon_lemma). Ukládá do registry.jsonl.

Spuštění:  .venv/bin/python experiments/hypothesis-one/gen_registry.py
"""
import sys, pickle, json, importlib.util
from collections import Counter

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb
vb.serve = lambda *a, **k: None
_spec = importlib.util.spec_from_file_location(
    "h1run", "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/run.py")
_run = importlib.util.module_from_spec(_spec); sys.modules["h1run"] = _run
_spec.loader.exec_module(_run)
canon = _run.canon_lemma
ROOT = "/Users/j/Projects/jellyAI3"

ANSWER_DEPREL = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:arg"}
CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "ADV"}

def _skip(lem, upos):
    return upos == "PUNCT" or (len(lem) == 1 and upos in ("NOUN", "PROPN", "SYM", "X"))

def gen(sent, doc, si, fold):
    verb = next((t for t in sent if t.get("deprel") == "root" and t["upos"] == "VERB"), None)
    if verb is None:
        verb = next((t for t in sent if t["upos"] == "VERB"), None)
    if verb is None:
        return None
    answers, context = [], []
    for t in sent:
        lem = fold.get(canon(t), canon(t))     # podmíněný epentetický fold PROPN
        if _skip(lem, t["upos"]):
            continue
        fe = t.get("feats") or {}
        if t.get("deprel") in ANSWER_DEPREL and t["upos"] in ("NOUN", "PROPN"):
            answers.append({"lemma": lem, "role": t["deprel"], "gender": fe.get("Gender"),
                            "name": bool(fe.get("NameType"))})
        if t["upos"] in CONTENT_UPOS and t is not verb:
            context.append(lem)
    if not answers:
        return None
    return {"predicate": canon(verb), "hole": "subj/person",
            "answers": answers, "context": context,
            "doc": doc, "sent": si,
            "text": " ".join(t["form"] for t in sent)[:120]}

def main():
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    # PODMÍNĚNÝ epentetický fold PROPN nad korpusem: Čapk→Čapek jen když „Čapek"
    # reálně existuje a je častější (nekorumpuje Egypt→Egypet, Petr→Peter)
    propn = Counter()
    for (_d, _s), rec in ann.items():
        for sent in rec["sentences"]:
            for t in sent:
                if t["upos"] == "PROPN":
                    propn[canon(t)] += 1
    fold = {}
    for lem, c in propn.items():
        cand = _run._epen_stem(lem)
        if cand != lem and propn.get(cand, 0) > c:
            fold[lem] = cand
    print("podmíněný fold PROPN: " + str(len(fold)) + " dvojic "
          + (("(" + ", ".join(f"{a}→{b}" for a, b in list(fold.items())[:5]) + ")") if fold else ""))
    out = open(f"{ROOT}/experiments/hypothesis-one/registry.jsonl", "w")
    n = 0; preds = Counter(); ans_total = 0; by_doc = Counter()
    sample = {}
    for (doc, si), rec in ann.items():
        for sent in rec["sentences"]:
            e = gen(sent, doc, si, fold)
            if e is None:
                continue
            out.write(json.dumps(e, ensure_ascii=False) + "\n")
            n += 1; preds[e["predicate"]] += 1; ans_total += len(e["answers"])
            by_doc[doc] += 1
            if "pátečníky" in e["text"] and "patecnici" not in sample:
                sample["patecnici"] = e
    out.close()
    print(f"vazeb (fakt↔otázka): {n}")
    print(f"unikátních predikátů: {len(preds)}")
    print(f"answer-slotů celkem: {ans_total}  (⌀ {ans_total/max(n,1):.1f} na fakt)")
    print(f"\nnejčastější predikáty: " + ", ".join(f"{p}×{c}" for p, c in preds.most_common(8)))
    print(f"vazeb per dokument: " + ", ".join(f"{d.split('_')[-1]}:{c}" for d, c in by_doc.most_common()))
    print(f"\n„patřit" + '"' + " vazeb: {}".format(preds.get("patřit", 0)))
    if "patecnici" in sample:
        s = sample["patecnici"]
        print("\nukázka (pátečníci):")
        print(f"  predikát: {s['predicate']}  díra: {s['hole']}")
        print(f"  answers:  " + ", ".join(f"{a['lemma']}[{a['role']}]" for a in s['answers']))
        print(f"  context:  {s['context']}")
        print(f"  fakt:     [{s['doc']}] {s['text']}")
    print("\n→ experiments/hypothesis-one/registry.jsonl")

if __name__ == "__main__":
    main()
