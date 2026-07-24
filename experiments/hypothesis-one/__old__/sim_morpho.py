#!/usr/bin/env python3
"""hypothesis-two — SIMULACE: cross-check lematizace UDPipe × MorphoDiTa.

Naše anotace jsou z UDPipe. Otázka: opraví MorphoDiTa (:8093) lemma-chyby?

NEGATIVNÍ NÁLEZ (změřeno): MorphoDiTa /analyze jak běží produkuje HORŠÍ lemmata —
tvar místo lemmatu (Synové→Synové, koně→koně), halucinace (přišli→„přišnout"),
špatné tagy (koním jako ADJ). Z 203 answer-slotů se lišila v 29 %, a ty rozdíly
byly VĚTŠINOU chyby MorphoDiTy, ne opravy. ⇒ korpusové služby lematizaci nepomohou,
UDPipe cache zůstává nejlepší zdroj. (Tail chyby = biblické imperativy/vzácná jména
by chtěly cílená pravidla, ne MorphoDiTu.)

Vyžaduje běžící MorphoDiTa na :8093.
Spuštění:  .venv/bin/python experiments/hypothesis-one/sim_morpho.py [N]
"""
import sys, json, random, urllib.request, pickle, re, unicodedata
from collections import Counter

ROOT = "/Users/j/Projects/jellyAI3"
ANSWER_DEPREL = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:arg"}

def morpho_analyze(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8093/analyze", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())["tokens"]

def clean(lemma):
    """morfflex lemma → holý základ (bez _technických značek a -čísel smyslu)."""
    return re.split(r"[_-]", lemma)[0] if lemma else lemma

def deacc(s):
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower())
                   if unicodedata.category(c) != "Mn")

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    sents = []
    for (doc, si), rec in ann.items():
        for sent in rec["sentences"]:
            if any(t.get("deprel") in ANSWER_DEPREL and t["upos"] in ("NOUN", "PROPN")
                   for t in sent):
                sents.append((doc, sent))
    random.seed(42)
    sample = random.sample(sents, min(N, len(sents)))

    agree = differ = 0
    diffs = []
    for doc, sent in sample:
        text = " ".join(t["form"] for t in sent)
        try:
            mt = morpho_analyze(text)
        except Exception as exc:            # noqa
            print("MorphoDiTa nedostupná:", exc); return
        m_lem = {}
        for tok in mt:
            m_lem.setdefault(tok["form"], clean(tok["lemma"]))
        for t in sent:
            if t.get("deprel") in ANSWER_DEPREL and t["upos"] in ("NOUN", "PROPN"):
                u = t["lemma"]; m = m_lem.get(t["form"])
                if m is None:
                    continue
                if deacc(u) == deacc(m):
                    agree += 1
                else:
                    differ += 1
                    diffs.append((t["form"], u, m, t["upos"]))
    tot = agree + differ
    print(f"answer-slotů porovnáno: {tot}  (vzorek {len(sample)} vět)")
    print(f"  shoda UDPipe = MorphoDiTa: {agree}/{tot}  ({100*agree/max(tot,1):.0f} %)")
    print(f"  liší se:                   {differ}/{tot}  ({100*differ/max(tot,1):.0f} %)\n")
    print("ukázky rozdílů  form: UDPipe → MorphoDiTa:")
    for form, u, m, up in diffs[:20]:
        print(f"    {form:16} {u!r:14} → {m!r:14} [{up}]")
    # nejčastější dvojice rozdílů
    pair = Counter((u, m) for _, u, m, _ in diffs)
    print("\nnejčastější rozdílové dvojice (UDPipe→MorphoDiTa):")
    for (u, m), c in pair.most_common(10):
        print(f"    {u!r:14} → {m!r:14} ×{c}")

if __name__ == "__main__":
    main()
