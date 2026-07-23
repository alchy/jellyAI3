#!/usr/bin/env python3
"""Topos — expert MÍSTO: rozliší, zda lemma je MÍSTO (GEO entita), pro výběr where-slotu.

Fúze z parentu (jellyai/iris/subsystems/topos.py), refaktorováno na minimum, které měření
žádá: fakt „narodit(who=Karel, where=[Svatoňovice, rodina])" má dvě where-hodnoty; správná
odpověď je MÍSTO. `is_place` (gazetteer GEO entit z NER) to rozliší — Svatoňovice ano, rodina
ne. Gazetteer je DATA (build_gazetteer.py), třída je čistý MECHANISMUS.

Zatím jen typová diskriminace (is_place). Containment (Praha ⊂ Čechy) z parenta je pozdější
rozšíření — přidá se, až ho měření vyžádá (otázka „Kde v Čechách…").
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class Topos:
    """Expert MÍSTO — `is_place(lemma)` nad gazetteerem GEO entit z korpusu.

    Instance drží množinu lemmat míst (načtenou z `data/gazetteer.json`). Metoda je
    čistá; sdílená s vrstvou ⑤ (matcher) pro výběr where-slotu dle typu entity.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte gazetteer (lemmata míst) z cesty v configu; prázdný, když chybí."""
        cfg = json.load(open(config_path, encoding="utf-8")).get("topos", {})
        path = os.path.join(HERE, cfg.get("gazetteer", "../../data/gazetteer.json"))
        self.places = set(json.load(open(path, encoding="utf-8"))) if os.path.exists(path) else set()

    def is_place(self, lemma):
        """True, když je lemma MÍSTO (v gazetteeru GEO entit). Svatoňovice ✓, rodina ✗."""
        return (lemma or "").lower() in self.places


if __name__ == "__main__":
    tp = Topos()
    print(f"gazetteer: {len(tp.places)} míst")
    for w in ["Svatoňovice", "Praha", "rodina", "škola"]:
        print(f"  {w:14} is_place={tp.is_place(w)}")
