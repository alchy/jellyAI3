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

# --- NOVÝ ZÁKON — faktické otázky, odpovědi OVĚŘENÉ proti raw textu evangelií (data/raw/bible_*.txt).
# (q, [expect+aliasy], sub, src). Doslovné doklady ve zdroji, ne z (zašuměné) extrakce.
BIBLE = [
    # kdo-udělal-co
    ("Kdo pokřtil Ježíše?",            ["Jan", "Křtitel"],          "who",      "bible_markus"),   # „v Jordánu od Jana pokřtěn"
    ("Kdo zradil Ježíše?",             ["Jidáš", "Iškariotský"],    "who",      "bible_matous"),   # „Jidáš, který ho zradil"
    ("Kdo zapřel Ježíše?",             ["Petr", "Šimon"],           "who",      "bible_matous"),   # „třikrát mě zapřeš" (Petr)
    # geografie (kde/kam/odkud) — jen co je ve vstupním textu
    ("Kde se narodil Ježíš?",          ["Betlém", "Betlémě"],       "spatial",  "bible_matous"),   # „narodil se v Judském Betlémě"
    ("Odkud byl Ježíš?",               ["Nazaret", "Nazareta"],     "spatial",  "bible_markus"),   # „Ježíš z Nazareta"
    ("Kde byl Ježíš pokřtěn?",         ["Jordán", "Jordánu"],       "spatial",  "bible_markus"),   # „byl v Jordánu pokřtěn"
    ("Kde křtil Jan?",                 ["Jordán", "Jordánu"],       "spatial",  "bible_markus"),   # „křtít v řece Jordánu"
    ("Kam přišel Ježíš po křtu?",      ["Galilej", "Galileje"],     "spatial",  "bible_markus"),   # „přišel Ježíš do Galileje"
    # číselné
    ("Kolik dní byl Ježíš na poušti?", ["čtyřicet", "40"],          "num",      "bible_markus"),   # „čtyřicet dní"
    ("Kolik měl Ježíš učedníků?",      ["dvanáct", "12", "dvanácti"], "num",    "bible_markus"),   # „Ustanovil jich dvanáct"
    # vztahové
    ("Kdo byl bratr Šimona?",          ["Ondřej"],                  "relation", "bible_markus"),   # „Šimona a jeho bratra Ondřeje"
    ("Kdo byl bratr Jakuba?",          ["Jan"],                     "relation", "bible_markus"),   # „Jakuba… a jeho bratra Jana"
    ("Kdo byl otec Jakuba a Jana?",    ["Zebedeus", "Zebedea", "Zebedeův"], "relation", "bible_markus"),  # „Jakuba Zebedeova"
    # kázání
    ("Co kázal Jan?",                  ["pokání"],                  "preach",   "bible_markus"),   # „kázal: Čiňte pokání"
    ("Co kázal Ježíš?",                ["evangelium", "evangelia"], "preach",   "bible_markus"),   # „kázal Boží evangelium"
    # kdo-byl (role) + více postav
    ("Kdo byl Pilát Pontský?",         ["vladař", "místodržitel"],  "who",      "bible_matous"),   # „vladaři Pilátovi"
    ("Kdo byl Herodes?",               ["král"],                    "who",      "bible_matous"),   # „krále Heroda"
    ("Kdo byl Jairos?",                ["představený", "synagóg"],  "who",      "bible_markus"),   # „představený synagógy, jménem Jairos"
    # umíme řešit (odpověď je ve vstupním textu), ale zatím nezodpovíme — frontiér
    ("Koho vzkřísil Ježíš?",           ["Lazar", "Lazara"],         "who",      "bible_jan"),      # „Lazar z Betanie"
    ("Co přinesli mudrci Ježíšovi?",   ["zlato", "kadidlo", "myrha"], "whom_what", "bible_matous"), # „zlato, kadidlo a myrhu"
    ("Kolika chleby nasytil Ježíš zástup?", ["pět", "5"],           "num",      "bible_jan"),      # „pět chlebů a dvě ryby"
    ("Kde proměnil Ježíš vodu ve víno?", ["Kána", "Káně", "Kán"],   "spatial",  "bible_jan"),      # „v Káně Galilejské"
    ("Kde byl Ježíš ukřižován?",       ["Golgota", "Golgotu"],      "spatial",  "bible_matous"),   # „Golgota, to znamená Lebka"
]
# NARATIVNÍ INDIKÁTORY — chceme pokryto vše; tyto systém (zatím) neumí → fail = poctivá značka
# mezery (user). Očekávaná odpověď je ověřená ve zdroji, ale je to obsah/parafráze, ne holé lemma.
BIBLE_NARR = [
    ("Kdo byl Ježíš?",                 ["Kristus", "syn", "Nazaretský"], "bible_markus"),          # narativní/otevřené
    ("Co řekl Ježíš učedníkům?",       ["rybář", "rybáře", "pojďte"],    "bible_markus"),          # „Pojďte za mnou… rybáře lidí"
    ("Co řekl Ježíš Janovi?",          ["pokřtít", "spravedlnost", "nech"], "bible_matous"),        # křest — otevřené
    ("Kde se setkal Ježíš s Janem?",   ["Jordán", "Jordánu"],            "bible_markus"),           # u Jordánu (křest)
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


def bible():
    """Novozákonní faktické otázky (kind=bible) + narativní indikátory (kind=bible-narrativa).

    Odpovědi OVĚŘENÉ proti raw evangeliím (ne proti zašuměné extrakci). Narativní jsou
    záměrně těžké (fail = poctivá značka nepokrytého — chceme pokryto vše).
    """
    items = []
    for q, expect, sub, src in BIBLE:
        items.append({"q": q, "expect": expect, "mode": "answer",
                      "kind": "bible", "sub": sub, "src": src})
    for q, expect, src in BIBLE_NARR:
        items.append({"q": q, "expect": expect, "mode": "answer",
                      "kind": "bible-narrativa", "sub": "narrativa", "src": src, "indicator": True})
    return items


def main():
    data = {"collision": collision(), "coverage": coverage(), "bible": bible()}
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = {k: len(v) for k, v in data.items()}
    print(f"→ {OUT}\n  " + " + ".join(f"{k} {v}" for k, v in n.items()) + f" = {sum(n.values())} položek")


if __name__ == "__main__":
    main()
