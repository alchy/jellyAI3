#!/usr/bin/env python3
"""PILOT: malá NN nad maticí VZORŮ místo ručních if-then pravidel (viz docs/naucene-smerovani.html).

Z každé etalonové otázky vytáhne STRUKTURNÍ VZOR-rysy (pattern dotazu — generické, ne ruční
sémantika) → drobný softmax klasifikátor (numpy) předpoví TYP/roli odpovědi (kind). Leave-one-out
CV. Ptáme se: naučí se z DAT to, co dnes hardcoduju v _answer_role / _is_relation_query (hlavně
copula × relation), aniž bych psal pravidla? Měřeno: 75.9 % vs 17.2 % (majorita).

Trénink OFFLINE, inference by byla deterministická (pevné váhy). Spuštění: python3 nn_pilot.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GOLD = os.path.join(HERE, "data/gold/gold_large.json")
CACHE = os.path.join(HERE, "pilot_parses.json")          # cache rozborů (gitignored)

INTERROG = ["kdo", "co", "kde", "kdy", "kolik", "jaký", "který", "čí", "jak"]


def build_cache():
    """Naparsuje etalon (UDPipe) a uloží tokeny + kind — ať se parsuje jen jednou."""
    from answering import Answering
    a = Answering()
    gold = json.load(open(GOLD, encoding="utf-8"))
    out = []
    for i, it in enumerate(gold):
        out.append({"q": it["q"], "kind": it["kind"], "toks": a._parse(it["q"])})
        if i % 20 == 0:
            print(f"  parsed {i}/{len(gold)}", flush=True)
    json.dump(out, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    return out


def feats(toks):
    """STRUKTURNÍ rysy VZORu dotazu — generické (kopula? tázací? root upos? genitiv u rootu?
    které tázací lemma? počet PROPN, sloveso, objekt, modalita ?). Žádné ruční 'if bratr'."""
    has_cop = int(any(t.get("deprel") == "cop" for t in toks))
    has_int = int(any("Int" in (t.get("feats") or {}).get("PronType", "") for t in toks))
    root_id = next((i + 1 for i, t in enumerate(toks) if t.get("head") == 0), None)
    root = toks[root_id - 1] if root_id else {}
    root_upos = [int(root.get("upos", "") == x) for x in ("NOUN", "PROPN", "VERB", "ADJ")]
    gen = int(any(t.get("head") == root_id
                  and (t.get("feats") or {}).get("Case") == "Gen" for t in toks))
    lems = [(t.get("lemma") or "").lower() for t in toks]
    inter = [int(x in lems) for x in INTERROG]
    n_propn = sum(1 for t in toks if t.get("upos") == "PROPN")
    has_verb = int(any(t.get("upos") == "VERB" for t in toks))
    has_obj = int(any(t.get("deprel") in ("obj", "obl", "obl:arg") for t in toks))
    modq = int(any(t.get("form") == "?" for t in toks))
    return [has_cop, has_int, gen, n_propn, has_verb, has_obj, modq] + root_upos + inter


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(X, Y, k, epochs=600, lr=0.5, l2=1e-3):
    """Drobný softmax (multinomiální log. regrese). OFFLINE → pevné váhy (W, b)."""
    W = np.zeros((X.shape[1], k)); b = np.zeros(k)
    Yoh = np.eye(k)[Y]
    for _ in range(epochs):
        P = softmax(X @ W + b)
        W -= lr * (X.T @ (P - Yoh) / len(X) + l2 * W)
        b -= lr * (P - Yoh).mean(0)
    return W, b


def main():
    data = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else build_cache()
    X = np.array([feats(d["toks"]) for d in data], float)
    kinds = sorted({d["kind"] for d in data})
    ki = {c: i for i, c in enumerate(kinds)}
    Y = np.array([ki[d["kind"]] for d in data])
    k = len(kinds)
    mu, sd = X.mean(0), X.std(0) + 1e-9

    preds = np.zeros(len(Y), int)                        # leave-one-out CV
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        W, b = train((X[tr] - mu) / sd, Y[tr], k)
        preds[i] = np.argmax(softmax(((X[i:i+1] - mu) / sd) @ W + b), 1)[0]

    acc = (preds == Y).mean()
    print(f"\n{'='*60}\nPILOT: NN nad VZOR-rysy → predikce TYPU odpovědi (LOO-CV)\n{'='*60}")
    print(f"vzorků {len(Y)}, rysů {X.shape[1]}, tříd {k}")
    print(f"\n  PŘESNOST (learned)  : {acc:.1%}")
    print(f"  baseline (majorita) : {np.bincount(Y).max() / len(Y):.1%}")
    print("\n  dle třídy (správně/celkem):")
    for c in kinds:
        idx = Y == ki[c]
        print(f"    {c:16} {int((preds[idx] == Y[idx]).sum()):2}/{int(idx.sum()):<2}")


if __name__ == "__main__":
    main()
