#!/usr/bin/env python3
"""Chronos — expert/podgraf ČAS: rozpozná časový výraz a přiřadí ho k predikátu.

„Kdy" díry patří časovému podgrafu. Rok (4místné NUM v rozsahu) se přiřadí k HLAVNÍMU
predikátu věty (root verb) — tím dostane fakt `when`-slot i z BĚŽNÉHO TEXTU, ne jen z
biografické závorky (kterou řeší extract_bio natvrdo). „Premiéra … měla proběhnout … došlo
k odložení termínu na 25. ledna 1921" → rok 1921 visí přes conj pod root `měla` → fakt
mít dostane when=1921 → „Kdy měla premiéru?" se trefí.

Fúze z parent/chronos (resolve_temporal). Zatím jen ROK; plné datum/interval/relativní čas
(„včera", „v 19. století") + clock/reminder jsou rozšíření — parent je má hotové.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config", "config.json")


class Chronos:
    """Expert ČAS — `temporal_facts(sent)` vytěží roky přiřazené k hlavnímu predikátu věty.

    Instance drží rozsah roků z configu. Metoda je čistá nad jednou větou; přiřazení k root
    verbu je záměrné — rok v podřízené klauzuli („…na 1921") sémanticky patří k události věty.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte rozsah přijímaných roků z configu (aby „123" nebo „9999" nebyl rok)."""
        cfg = json.load(open(config_path, encoding="utf-8")).get("chronos", {})
        self.year_min = cfg.get("year_min", 1000)
        self.year_max = cfg.get("year_max", 2100)

    def _is_year(self, form):
        return form.isdigit() and self.year_min <= int(form) <= self.year_max

    def temporal_facts(self, sent, grammar):
        """Věta → [(predikát, rok)] — roky přiřazené k HLAVNÍMU predikátu (root verb) věty.

        Bez root-verbu nebo bez roku → []. Vstup: anotovaná věta, grammar (kanonizace).
        """
        root = next((t for t in sent if t["deprel"] == "root" and t["upos"] == "VERB"), None)
        if not root:
            return []
        pred = grammar.canon_lemma(root)
        return [(pred, t["form"]) for t in sent if t["upos"] == "NUM" and self._is_year(t["form"])]


if __name__ == "__main__":
    import pickle
    from grammar_vzor import GrammarVzor
    g = GrammarVzor()
    ch = Chronos()
    shard = pickle.load(open(os.path.join(HERE, "data/corpus/wiki_r.u.r..pkl"), "rb"))
    for rec in shard.values():
        for s in rec["sentences"]:
            tf = ch.temporal_facts(s, g)
            if tf:
                print(tf, "::", " ".join(t["form"] for t in s)[:70])
