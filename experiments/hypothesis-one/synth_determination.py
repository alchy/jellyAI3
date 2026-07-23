#!/usr/bin/env python3
"""④ SYNTHESIS — CONFIRMED slovník VZOR → role (začištěný nad celým korpusem).

Agreguje role per VZOR (SLOT_ARRAY) přes všechny věty korpusu a POVÝŠÍ jen začištěné:
PURE (jediná role na VZOR) + FREQUENT (≥ prahu) + content-role (kde anotátor chybuje).
Heuristická revize = KONZISTENCE, ne správnost; `curated.json` opravuje známé chyby,
impure/vzácné zůstávají kandidáty k lidské revizi.

Konzumuje NOVÝ korpus (shardy ① přes `Dataloader.mount`), role bere přes
`RoleCatalog.standard_role`, VZOR přes `GrammarVzor.frame_sig`. Zapisuje `determination.json`.
"""
import os
import json
from collections import defaultdict, Counter

from grammar_vzor import GrammarVzor
from role_catalog import RoleCatalog
from dataloader import Dataloader
from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class SynthDetermination:
    """Staví CONFIRMED slovník VZOR→role z anotovaného korpusu.

    Instance drží primitiva (`GrammarVzor`), rozklad na role (`RoleCatalog`), zavaděč
    korpusu a prahy z configu. `aggregate` je čistá agregace; `run` projde korpus a zapíše.
    """

    def __init__(self, grammar=None, roles=None, loader=None, config_path=CONFIG_PATH):
        """Vezme (nebo vytvoří) `GrammarVzor`/`RoleCatalog`/`Dataloader` a načte prahy."""
        self.g = grammar or GrammarVzor(config_path)
        self.rc = roles or RoleCatalog(self.g, config_path)
        self.dl = loader or Dataloader(config_path)
        cfg = json.load(open(config_path, encoding="utf-8"))
        self.det_path = os.path.join(HERE, cfg["synthesis"]["determination"])
        self.radius = cfg["synthesis"]["vzor_radius"]
        self.min = cfg["synthesis"]["determination_min"]
        self.STRUCT = set(self.g.LANG["structural_roles"])

    def aggregate(self, sentences):
        """Spočte {VZOR: Counter(role→počet)} přes věty — jen content-role.

        Strukturální role a interpunkce se vynechávají (content-role je to, kde
        anotátor chybuje a kde má smysl VZOR opravovat). Vstup: seznam vět.
        """
        vzor_roles = defaultdict(Counter)
        for s in sentences:
            byid = self.rc._byid(s)
            mod = self.g.sentence_modality(s)
            cop = self.rc._cop_heads(s)
            for i, w in enumerate(s):
                if w["upos"] == "PUNCT":
                    continue
                role = self.rc.standard_role(w, s, byid, cop)
                if role is None or role in self.STRUCT:
                    continue
                vzor_roles[self.g.frame_sig(s, i, mod, self.radius)][role] += 1
        return vzor_roles

    def confirmed(self, vzor_roles):
        """Vybere PURE (1 role) + FREQUENT (≥ prahu) VZORy → {VZOR: role}.

        Konzistence, ne správnost — impure/vzácné jsou kandidáti k revizi.
        """
        return {vz: c.most_common(1)[0][0] for vz, c in vzor_roles.items()
                if len(c) == 1 and sum(c.values()) >= self.min}

    def run(self):
        """Projde celý korpus a zapíše `determination.json`.

        Vrací souhrn (počet VZORů, CONFIRMED, pokrytí content-tokenů).
        """
        corpus = self.dl.mount(self.dl.document_ids())
        sents = [s for rec in corpus.values() for s in rec["sentences"]]
        vzor_roles = self.aggregate(sents)
        confirmed = self.confirmed(vzor_roles)
        n_tok = sum(sum(c.values()) for c in vzor_roles.values())
        covered = sum(sum(vzor_roles[vz].values()) for vz in confirmed)

        data = {"_comment": ("CONFIRMED VZOR(SLOT_ARRAY)→role — auto-build nad korpusem "
                             f"(PURE + FREQUENT ≥{self.min}); revize = KONZISTENCE, ne správnost. "
                             "curated.json opravuje chyby; impure/vzácné = kandidáti.")}
        for vz, role in confirmed.items():
            data[vz] = {"role": role, "status": "confirmed",
                        "count": sum(vzor_roles[vz].values())}
        json.dump(data, open(self.det_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        logger("i", f"VZORů: {len(vzor_roles)}, CONFIRMED (pure & ≥{self.min}): {len(confirmed)}, "
                     f"pokrytí content-tokenů: {covered}/{n_tok} "
                     f"({100*covered/max(n_tok,1):.0f}%)")
        return {"vzors": len(vzor_roles), "confirmed": len(confirmed), "coverage": covered}


if __name__ == "__main__":
    SynthDetermination().run()
