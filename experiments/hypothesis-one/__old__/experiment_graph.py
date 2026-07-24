#!/usr/bin/env python3
"""Sweep variant VAH UZLŮ × VODIVOSTI HRAN — které rozsvítí obsah, ne funkční slova/bibli.

Měř-first pro graf: na dotazu „pes domácí" nad doménou domácích zvířat (+ bible jako
kontaminace) porovná varianty grafu a světla. Metrika: doteče světlo na OBSAH
(šelma/savec/…), uteče do FUNKČNÍCH slov (být/se) nebo do CIZÍ domény (bible)?
"""
import os
import math
from collections import defaultdict

from grammar_vzor import GrammarVzor
from dataloader import Dataloader

g = GrammarVzor()
dl = Dataloader()
IDF = dl.load_idf()["idf"]
DEF = max(IDF.values())
def idf(w): return IDF.get(w, DEF)

DOCS = ["wiki_pes_domácí", "wiki_kočka_domácí", "wiki_kůň_domácí", "wiki_koza_domácí",
        "wiki_ovce_domácí", "wiki_prase_domácí", "wiki_králík_domácí",
        "bible_jan", "bible_genesis"]
corpus = dl.mount(DOCS)
SENTS = [[g.canon_lemma(t).lower() for t in s if t["upos"] != "PUNCT"]
         for rec in corpus.values() for s in rec["sentences"]]

CONTENT = ["šelma", "savec", "masožravec", "živočich", "tvor", "domestikovaný",
           "zvíře", "plemeno", "chov", "smečka"]
LEAK = ["být", "se", "a", "v", "na", "který", "ten", "mít", "s", "u"]
BIBLE = ["bůh", "ježíš", "hospodin", "syn", "otec", "duch"]


def build(cond, window, tau):
    adj = defaultdict(lambda: defaultdict(float))
    for content in SENTS:
        n = len(content)
        for i in range(n):
            for j in range(i + 1, min(i + 1 + window, n)):
                prox = math.exp(-(j - i - 1) / tau) if tau else 1.0
                w = cond(content[i], content[j]) * prox
                adj[content[i]][content[j]] += w
                adj[content[j]][content[i]] += w
    return adj


def spread(adj, seeds, weight, hops=2, lam=0.5):
    words = {w: weight(w) for w in seeds}
    for _ in range(hops):
        add = defaultdict(float)
        for lem, heat in list(words.items()):
            nb = adj.get(lem)
            if not nb:
                continue
            tot = sum(nb.values())
            for x, wt in nb.items():
                add[x] += heat * lam * (wt / tot)
        for x, h in add.items():
            words[x] = words.get(x, 0.0) + h
    return words


VARIANTS = [
    ("V0 bigram/flat (dnešní)", lambda a, b: 1.0, 1, None, lambda w: 1.0),
    ("V1 idf²/win5/prox",       lambda a, b: idf(a) * idf(b), 5, 2.0, idf),
    ("V2 idf²/win5/BEZ prox",   lambda a, b: idf(a) * idf(b), 5, None, idf),
    ("V3 min-idf/win5/prox",    lambda a, b: min(idf(a), idf(b)), 5, 2.0, idf),
    ("V4 idf²/win8/prox",       lambda a, b: idf(a) * idf(b), 8, 2.0, idf),
    ("V5 idf²/win3/prox",       lambda a, b: idf(a) * idf(b), 3, 2.0, idf),
    ("V6 flat-cond/win5/idf-váha", lambda a, b: 1.0, 5, 2.0, idf),
]

print(f"\n{'varianta':26}{'šelma':>7}{'OBSAH':>8}{'únik-fn':>9}{'BIBLE':>7}{'obsah/únik':>12}  top-obsah rank")
print("-" * 92)
for name, cond, win, tau, weight in VARIANTS:
    adj = build(cond, win, tau)
    words = spread(adj, ["pes", "domácí"], weight)
    ranked = sorted(words.items(), key=lambda x: -x[1])
    rank = {w: i for i, (w, _h) in enumerate(ranked)}
    sh = words.get("šelma", 0.0)
    content = sum(words.get(w, 0.0) for w in CONTENT)
    leak = sum(words.get(w, 0.0) for w in LEAK)
    bib = sum(words.get(w, 0.0) for w in BIBLE)
    ratio = content / (leak + 1e-9)
    # nejlépe umístěné obsahové slovo
    best_c = min((rank.get(w, 9999), w) for w in CONTENT)
    print(f"{name:26}{sh:7.3f}{content:8.3f}{leak:9.3f}{bib:7.3f}{ratio:12.2f}  #{best_c[0]} {best_c[1]}")

print("\n(šelma>0 = světlo doteklo; obsah/únik vysoký = obsah bije funkční slova; BIBLE~0 = doména neteče do cizí)")
