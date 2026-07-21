#!/usr/bin/env python3
"""Offline build + začištění CONFIRMED slovníku VZOR→role nad CELÝM korpusem.

Agreguje role per VZOR (SLOT_ARRAY, r=2) přes všechny WORD_W_ATTR_ARRAY-y korpusu.
Povýší jen ZAČIŠTĚNÉ: PURE (jedna role) + FREQUENT (≥ MIN výskytů) + content-role.
Heuristická revize = KONZISTENCE, ne správnost; `curated.json` opravuje známé chyby,
impure/vzácné zůstávají kandidáty k lidské revizi. Zapíše determination.json.

Spuštění:  .venv/bin/python build_determination.py [MIN]
"""
import sys, json, pickle
from collections import defaultdict, Counter
sys.path.insert(0, "/Users/j/Projects/jellyAI3/experiments/hypothesis-one")
import phase1
import roles as R
run = R._run
STRUCT = set(run.LANG["structural_roles"])
R_RADIUS = 2

def main():
    MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    ann = pickle.load(open(phase1._HERE + "../../data/annotations.pkl", "rb"))
    sents = [s for (_d, _i), rec in ann.items() for s in rec["sentences"]]

    vzor_roles = defaultdict(Counter)
    n_tok = 0
    for s in sents:
        byid = R._byid(s); mod = run.sentence_modality(s)
        cop = {x["head"] for x in s if x["deprel"] == "cop"}
        for i, w in enumerate(s):
            if w["upos"] == "PUNCT":
                continue
            role = R.standard_role(w, s, byid, cop)
            if role is None or role in STRUCT:      # jen content-role (kde ÚFAL chybuje)
                continue
            n_tok += 1
            vzor_roles[run.frame_sig(s, i, mod, R_RADIUS)][role] += 1

    # ZAČIŠTĚNÍ: pure (1 role) + frequent (≥ MIN)
    confirmed = {vz: c.most_common(1)[0][0] for vz, c in vzor_roles.items()
                 if len(c) == 1 and sum(c.values()) >= MIN}
    covered = sum(sum(vzor_roles[vz].values()) for vz in confirmed)

    data = {"_comment": ("CONFIRMED VZOR(SLOT_ARRAY)→role — auto-build nad korpusem "
                         f"(PURE + FREQUENT ≥{MIN}); heuristická revize = KONZISTENCE, "
                         "ne správnost. curated.json opravuje chyby; impure/vzácné = kandidáti.")}
    for vz, role in confirmed.items():
        data[vz] = {"role": role, "status": "confirmed", "count": sum(vzor_roles[vz].values())}
    json.dump(data, open(phase1._HERE + "determination.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"vět: {len(sents)}   content-tokenů: {n_tok}")
    print(f"distinct VZORů: {len(vzor_roles)}")
    print(f"CONFIRMED (pure & ≥{MIN}): {len(confirmed)}")
    print(f"pokrytí content-tokenů: {covered}/{n_tok} ({100*covered/max(n_tok,1):.0f}%)")
    print("→ determination.json")

if __name__ == "__main__":
    main()
