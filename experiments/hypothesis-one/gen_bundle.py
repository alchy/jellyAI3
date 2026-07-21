#!/usr/bin/env python3
"""hypothesis-two — DETERMINISTICKÝ svazek stínů: fakt → { synt.otázka }ⁿ.

Fakt je n-ární: vrhá JEDEN stín na KAŽDOU rolovou díru (ne jeden na fakt).
Svazek se staví SYMBOLICKY z roles.decompose (vyjmenuje díry per klauzule) +
role_ask (tázací slovo per role, data v cs.json). Žádná Ollama — to je jádro,
match, který definujeme sami. (Parafráze nad svazkem = druhá vrstva, ta Ollamu
využívá — gen_variants.)

Bonus: přes per-klauzulový rozklad vzniknou i díry z vedlejších vět, které
registr per-věta ztrácel — např. „to Slovo byl Bůh" → stín state → Bůh.

Ukládá bundle.jsonl (jeden řádek = jeden stín).  Deterministické, bez sítě.
"""
import sys, json, pickle, importlib.util
from collections import Counter

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import roles as R                       # roles importuje run (LANG, role_key…)
_run = R._run
ROOT = "/Users/j/Projects/jellyAI3"
ROLE_ASK = _run.LANG["role_ask"]

# díry = askable role (vynech predikát a jeho tvary; ty nejsou nominální díra)
NOT_HOLE = {"action", "past_participle", "passive_participle"}
HOLE_ROLES = [k for k in _run.LANG["role_catalog"] if k not in NOT_HOLE]

def shadows_of(sent, doc, si):
    """Všechny stíny jednoho faktu: (predikát, díra-role, tázací slovo, odpověď)."""
    clauses = R.decompose(sent)
    ctx = [R.canon(t) if hasattr(R, "canon") else _run.canon_lemma(t)
           for t in sent if t["upos"] in ("NOUN", "PROPN", "VERB", "ADJ", "NUM")]
    text = " ".join(t["form"] for t in sent)[:120]
    out = []
    for c in clauses:
        for role, vals in c["roles"].items():
            if role in NOT_HOLE or role.startswith("_"):
                continue
            vals = [v for v in vals if len(v) > 1]        # vyhoď jednopísmenné paskvily
            if not vals:
                continue
            out.append({"predicate": c["predicate"], "hole": role,
                        "q": ROLE_ASK.get(role, "?"), "answers": vals,
                        "context": ctx, "doc": doc, "sent": si, "text": text})
    return out

def main():
    ann = pickle.load(open(f"{ROOT}/data/annotations.pkl", "rb"))
    out = open(f"{ROOT}/experiments/hypothesis-one/bundle.jsonl", "w")
    n_fact = n_shadow = 0
    holes = Counter(); per_fact = Counter()
    for (doc, si), rec in ann.items():
        for sent in rec["sentences"]:
            sh = shadows_of(sent, doc, si)
            if not sh:
                continue
            n_fact += 1; per_fact[len(sh)] += 1
            for s in sh:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_shadow += 1; holes[s["hole"]] += 1
    out.close()
    print(f"faktů: {n_fact}   stínů (otázek): {n_shadow}   ⌀ {n_shadow/max(n_fact,1):.1f} na fakt")
    print("děr per fakt (histogram): " +
          ", ".join(f"{k}:{per_fact[k]}" for k in sorted(per_fact)))
    print("nejčastější díry: " + ", ".join(f"{h}×{c}" for h, c in holes.most_common(10)))
    print("→ bundle.jsonl")

if __name__ == "__main__":
    main()
