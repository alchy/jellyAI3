#!/usr/bin/env python3
"""Staví gold_domain.json z RUČNĚ OVĚŘENÉ faktové tabulky (proti raw textu wiki, ne proti
extrakci). Fakta jsou kurátorovaná ground truth (níže, s citacemi ve specu); šablony otázek
jsou deterministické. Výstup pak projde ruční kontrolou + `eval_domain.py --sweep`.

collision = lexikálně matoucí (bratři Čapkové mají RŮZNÉ hodnoty → špatný mount = špatná
hodnota; `expect_doc` = soubor, z něhož má vítězný fakt pocházet).
coverage  = systematické per-entita pokrytí (rok/místo/dílo/povolání/taxonomie).
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data/gold/gold_domain.json")

# lemma-tvary míst + aliasy (jak je answering vrací: holé lemma), pro alias-aware accept
PLACE = {
    "Malé Svatoňovice": ["Svatoňovice", "Malé Svatoňovice"],
    "Bergen-Belsen": ["Bergen-Belsen", "Bergen", "Belsen"],
    "Lipnice nad Sázavou": ["Lipnice", "Lipnice nad Sázavou"],
    "Háj ve Slezsku": ["Háj", "Háj ve Slezsku"],
}
def place(p):
    return PLACE.get(p, [p])

# --- OVĚŘENÁ FAKTA (subagent extrakce z data/raw/wiki_*.txt, doslovné citace ve specu) ---
# (doc, jméno, žena?, rok_nar, místo_nar, rok_úmrtí, místo_úmrtí, povolání[+alt])
AUTHORS = [
    ("wiki_bohumil_hrabal",     "Bohumil Hrabal",     False, "1914", "Židenice",         "1997", "Praha",              ["prozaik", "spisovatel"]),
    ("wiki_božena_němcová",     "Božena Němcová",     True,  "1820", "Vídeň",            "1862", "Praha",              ["spisovatelka"]),
    ("wiki_ivan_olbracht",      "Ivan Olbracht",      False, "1882", "Semily",           "1952", "Praha",              ["spisovatel", "prozaik"]),
    ("wiki_jan_neruda",         "Jan Neruda",         False, "1834", "Praha",            "1891", "Praha",              ["básník", "novinář"]),
    ("wiki_jaroslav_hašek",     "Jaroslav Hašek",     False, "1883", "Praha",            "1923", "Lipnice nad Sázavou",["spisovatel"]),
    ("wiki_jaroslav_seifert",   "Jaroslav Seifert",   False, "1901", "Žižkov",           "1986", "Praha",              ["básník", "spisovatel"]),
    ("wiki_karel_hynek_mácha",  "Karel Hynek Mácha",  False, "1810", "Praha",            "1836", "Litoměřice",         ["básník", "prozaik"]),
    ("wiki_milan_kundera",      "Milan Kundera",      False, "1929", "Brno",             "2023", "Paříž",              ["spisovatel"]),
    ("wiki_vítězslav_nezval",   "Vítězslav Nezval",   False, "1900", "Biskoupky",        "1958", "Praha",              ["básník", "spisovatel"]),
    ("wiki_vladislav_vančura",  "Vladislav Vančura",  False, "1891", "Háj ve Slezsku",   "1942", "Praha",              ["spisovatel", "dramatik"]),
]
# bratři Čapkové — kolizní jádro (různé hodnoty, sdílené příjmení)
BROTHERS = [
    ("wiki_josef_čapek", "Josef Čapek", "1887", "Hronov",           "1945", "Bergen-Belsen", ["malíř", "grafik"]),
    ("wiki_karel_čapek", "Karel Čapek", "1890", "Malé Svatoňovice", "1938", "Praha",         ["spisovatel", "dramatik"]),
]
# taxonomie (doslovný taxon z úvodní věty; prase vynecháno — v úvodní větě žádný taxon;
# skot vynechán — jeho soubor je ROZCESTNÍK, ne článek o druhu)
ANIMALS = [
    ("wiki_pes_domácí",    "pes domácí",    ["šelma"]),
    ("wiki_kočka_domácí",  "kočka domácí",  ["forma"]),
    ("wiki_koza_domácí",   "koza domácí",   ["přežvýkavec"]),
    ("wiki_králík_domácí", "králík domácí", ["forma"]),
    ("wiki_kůň_domácí",    "kůň domácí",    ["lichokopytník"]),
    ("wiki_ovce_domácí",   "ovce domácí",   ["přežvýkavec"]),
]
# díla → autor (oba bratři jsou „Čapek" → answer nerozliší; expect_doc NEnastaven — fakt
# „napsat(Čapek, dílo)" legitimně žije i v článku autora, nejen díla; test cílí provázání
# díla s autorem, ne výběr souboru)
WORKS = [
    ("wiki_r.u.r.",         "R.U.R.",           "1920"),
    ("wiki_válka_s_mloky",  "Válku s Mloky",    "1935"),
    ("wiki_bílá_nemoc",     "Bílou nemoc",      "1937"),
]


def born(fem):
    return "narodila" if fem else "narodil"
def died(fem):
    return "zemřela" if fem else "zemřel"


def coverage():
    items = []
    for doc, name, fem, by, bp, dy, dp, prof in AUTHORS:
        items += [
            {"q": f"Kdy se {born(fem)} {name}?", "expect": [by], "mode": "answer", "kind": "temporal", "src": doc},
            {"q": f"Kdy {died(fem)} {name}?",    "expect": [dy], "mode": "answer", "kind": "temporal", "src": doc},
            {"q": f"Kde se {born(fem)} {name}?", "expect": place(bp), "mode": "answer", "kind": "spatial", "src": doc},
            {"q": f"Kde {died(fem)} {name}?",    "expect": place(dp), "mode": "answer", "kind": "spatial", "src": doc},
            {"q": f"Kdo je {name}?", "expect": prof, "mode": "answer", "kind": "copula", "src": doc},
        ]
    for doc, subj, taxon in ANIMALS:
        items.append({"q": f"Co je {subj}?", "expect": taxon, "mode": "answer", "kind": "taxonomy", "src": doc})
    # dílo → autor (provázání; oba bratři jsou „Čapek", proto bez expect_doc)
    for doc, title, year in WORKS:
        items.append({"q": f"Kdo napsal {title}?", "expect": ["Čapek", "Karel Čapek"],
                      "mode": "answer", "kind": "authorship-who", "src": doc})
    return items


def collision():
    items = []
    # bratři Čapkové — každý fakt má PROTIPÓL v článku druhého bratra
    jo, ka = BROTHERS
    for doc, name, by, bp, dy, dp, prof in BROTHERS:
        other = ka if doc == jo[0] else jo
        items += [
            {"q": f"Kdy se narodil {name}?", "expect": [by], "expect_doc": doc, "mode": "answer",
             "kind": "collision", "trap": f"bratr → {other[2]}"},
            {"q": f"Kde se narodil {name}?", "expect": place(bp), "expect_doc": doc, "mode": "answer",
             "kind": "collision", "trap": f"bratr → {other[3]}"},
            {"q": f"Kdy zemřel {name}?", "expect": [dy], "expect_doc": doc, "mode": "answer",
             "kind": "collision", "trap": f"bratr → {other[4]}"},
            {"q": f"Kde zemřel {name}?", "expect": place(dp), "expect_doc": doc, "mode": "answer",
             "kind": "collision", "trap": f"bratr → {other[5]}"},
            {"q": f"Kdo je {name}?", "expect": prof, "expect_doc": doc, "mode": "answer",
             "kind": "collision", "trap": f"bratr → {other[6][0]}"},
        ]
    # ambiguita holého jména → poctivé doptání/nejistota (ne sebevědomý tip)
    items += [
        {"q": "Kdy se narodil Čapek?", "expect": ["1887", "1890"], "mode": "unsure",
         "kind": "collision", "trap": "Josef 1887 vs Karel 1890 — má doptat"},
        {"q": "Kdy se narodil Jaroslav?", "expect": ["1883", "1901"], "mode": "unsure",
         "kind": "collision", "trap": "Hašek 1883 vs Seifert 1901 — má doptat"},
    ]
    return items


def main():
    data = {"collision": collision(), "coverage": coverage()}
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    nc, nv = len(data["collision"]), len(data["coverage"])
    print(f"→ {OUT}\n  collision {nc} + coverage {nv} = {nc + nv} položek")


if __name__ == "__main__":
    main()
