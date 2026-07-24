#!/usr/bin/env python3
"""hypothesis-two — VALIDACE generování: lematizace + atributy na náhodném vzorku.

Jde zpět na tokeny anotace a ukazuje řetěz  form → UDPipe lemma → canon_lemma
+ rod/pád/číslo — abychom viděli, jestli je lematizace a kanonizace správná.
Zvlášť AUDIT foldu: která PROPN lemmata canon mění (Čapk→Čapek dobře × korupce jmen).

Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_verify.py [N]
"""
import sys, random, importlib.util, pickle
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

def feats_str(t):
    fe = t.get("feats") or {}
    return " ".join(f"{k[0].lower()}={fe[k]}" for k in ("Gender", "Case", "Number", "Tense") if fe.get(k)) or "—"

def sentences(ann):
    for (doc, si), rec in ann.items():
        for sent in rec["sentences"]:
            yield doc, si, sent

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    sents = [s for s in sentences(ann) if any(t["upos"] == "VERB" for t in s[2])]

    # --- AUDIT foldu: kolik PROPN lemmat canon mění, a jak ---
    changed = Counter(); propn = 0
    for _, _, sent in sents:
        for t in sent:
            if t["upos"] == "PROPN":
                propn += 1
                c = canon(t)
                if c != t["lemma"]:
                    changed[(t["lemma"], c)] += 1
    print(f"AUDIT foldu PROPN: {propn} výskytů, canon mění {sum(changed.values())} "
          f"({len(changed)} unik. dvojic)")
    print("  ukázka změn lemma→canon (má být Čapk→Čapek; korupce jmen = špatně):")
    for (a, b), c in changed.most_common(10):
        print(f"    {a!r:14} → {b!r:14} ×{c}")

    # --- metriky lematizace ---
    tot = badlem = imper = 0
    for _, _, sent in sents:
        for t in sent:
            if t.get("deprel") in ANSWER_DEPREL and t["upos"] in ("NOUN", "PROPN"):
                tot += 1
                lem = t["lemma"]
                # heuristika „podezřelé lemma": velké písmeno uvnitř, končí na sloves. způsob
                if any(ch.isupper() for ch in lem[1:]) or lem != lem.strip("!?."):
                    badlem += 1
    print(f"\nLEMATIZACE answer-slotů: {tot} slotů, podezřelých lemmat: {badlem} "
          f"({100*badlem/max(tot,1):.1f} %)")

    # --- náhodný vzorek: form → lemma → canon + rysy ---
    random.seed(42)
    sample = random.sample(sents, min(N, len(sents)))
    print(f"\nVZOREK ({len(sample)}, seed 42) — predikát a answer-sloty: form → lemma → canon [rysy]:\n")
    for i, (doc, si, sent) in enumerate(sample, 1):
        verb = next((t for t in sent if t.get("deprel") == "root" and t["upos"] == "VERB"),
                    next((t for t in sent if t["upos"] == "VERB"), None))
        txt = " ".join(t["form"] for t in sent)[:90]
        print(f"{i:2} [{doc.split('_')[-1]}] {txt}")
        if verb:
            print(f"      PREDIKÁT  {verb['form']!r} → {verb['lemma']!r} → {canon(verb)!r}  [{feats_str(verb)}]")
        for t in sent:
            if t.get("deprel") in ANSWER_DEPREL and t["upos"] in ("NOUN", "PROPN"):
                mark = "  ⚠fold" if canon(t) != t["lemma"] else ""
                print(f"      answer    {t['form']!r} → {t['lemma']!r} → {canon(t)!r}  "
                      f"[{t['upos']} {t.get('deprel')} · {feats_str(t)}]{mark}")

if __name__ == "__main__":
    main()
