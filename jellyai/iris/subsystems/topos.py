"""Topos — orientace Iris v PROSTORU: kontejnment míst (spec §5, S3).

Zrcadlo Chronosu na ose prostoru: Chronos odpovídá „spadá okamžik do
intervalu?", Topos „leží místo uvnitř oblasti?" (Praha ⊂ Čechy ⊂ Česko).
Kontejnment nese GAZETTEER — JSONL záznam subsystému
(`data/sub_topos_gazetteer.jsonl`, řádek `{"place", "in"}`): kurátorský
seed + budoucí učení dialogem („Kaufland je v Praze" — S5). Jména se
porovnávají KMENOVĚ (skloněný povrch „v Praze" ↔ „Praha").
"""

import json
import os

from jellyai.graph.canon import _stem, deaccent
from jellyai.lang import current


def _key(name):
    """Kmenový klíč jména místa (bez pádu a diakritiky)."""
    return deaccent(_stem(name.strip().lower()))


def _keys(name):
    """Klíč + PALATALIZOVANÁ varianta („Praze": praz → prah) — česká
    přehláska koncové souhlásky kmene (tabulka `palatal_fold`)."""
    base = _key(name)
    out = {base}
    if len(base) > 5 and base.endswith("ich"):
        out.add(base[:-3])           # lokativ plurálu: „Petrovicích"
    vowels = "aeiouy"
    if len(base) > 2 and base[-1] not in vowels and base[-2] not in vowels:
        out.add(base[:-1] + "e" + base[-1])   # epenteze: „Plzni" ↔ „Plzeň"
    for key in list(out):
        # DRUHÝ ořez (dávka D): nominativ v gazetteeru mohl přijít
        # o delší koncovku než pádový tvar otázky („Jeruzalém"→jeruzal
        # přes -ém, ale „Jeruzalémě"→…→jeruzalem přes -ě + epentezi) —
        # klíče se míjely a místní filtr se u těchto míst nikdy
        # nenárokoval; délková pojistka ≥ 4 proti zmršení
        again = deaccent(_stem(key))
        if again != key and len(again) >= 4:
            out.add(again)
    fold = current().get("palatal_fold", {})
    for key in list(out):
        if key and key[-1] in fold:
            out.add(key[:-1] + fold[key[-1]])
    return out


def load_gazetteer(path):
    """Načte gazetteer: kmen místa → kmen nadřazené oblasti.

    Returns:
        dict: {kmen místa: kmen oblasti}; chybějící soubor = prázdný.
    """
    parents = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if "in" in row:      # {"place","near"} = sousedství
                        parents[_key(row["place"])] = _key(row["in"])
    return parents


def area_keys(parents):
    """Kmeny všech OBLASTÍ (cílů kontejnmentu) — nárok brány Q: slovo
    otázky, které je oblastí, je FILTR, ne účastník."""
    return set(parents.values()) | set(parents)


def learn_containment(parents, path, inner, outer):
    """UČENÍ ZA POCHODU: vnořená místa výroku („na Barrandově v Praze")
    učí kontejnment — vnitřní ⊂ vnější. Zapisuje do gazetteeru (záznam
    subsystému) i do živého slovníku; obrácené pořadí („v Praze na
    Barrandově") pozná podle už známého řetězu a otočí se. Existující
    znalost se nepřepisuje.

    Returns:
        bool: True, když se subsystém něco nového naučil.
    """
    if place_within(outer, inner, parents):
        inner, outer = outer, inner          # už víme, že to je naopak
    key, target = _key(inner), _key(outer)
    if not key or not target or key == target or key in parents:
        return False
    parents[key] = target
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"place": inner, "in": outer},
                                ensure_ascii=False) + "\n")
    return True


def place_within(name, area, parents):
    """Leží místo (libovolný pád) uvnitř oblasti? — výstup kontejnmentu.

    Prochází řetěz rodičů (Praha → Čechy → Česko); místo leží i samo
    v sobě („v Praze" ⊇ Praha). Cyklus jistí limit délky řetězu.
    """
    targets = _keys(area)
    candidates = {k for k in _keys(name) if k}
    if not candidates or not targets:
        return False
    for _ in range(12):
        if candidates & targets:
            return True
        candidates = {parents[k] for k in candidates if k in parents}
        if not candidates:
            return False
    return False
