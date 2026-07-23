#!/usr/bin/env python3
"""ActivationField — SPREADING aktivace nad váženým grafem vazeb.

Světlo nad stíny: slova otázky rozsvítí uzly (slova) a soubory (brána ②), teplo se pak
ROZTEČE po vážených hranách (adjacence slov z mountovaného korpusu), a matcher tím váží
VZOR-kandidáty — mezi překrývajícími se stíny vybere ten nejvíc osvětlený. Vyřešená odpověď
zpětně přihřeje (řetěz drží téma).

Graf hran v1 = adjacence content-lemmat z MOUNTOVANÉHO korpusu (horké soubory); mesh λ^d,
doc_links a jas uzlů (run.py) se zapojí později — je to hotová logika, jen přenést.
"""
import os
import json
import math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class ActivationField:
    """Decayující aktivační pole nad slovy/soubory s šířením po hranách.

    `words`/`files` drží teplo; `adj` je vážený graf sousednosti slov. Pole živí rozhovor
    (`feed`), teplo se šíří (`spread`), mezi tahy utlumí (`decay`); `weight` dá aktivaci
    šablony, `reinforce` zpětný tok.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte parametry (decay, lambda, hops, window, tau) + idf tabulku; prázdné pole."""
        cfg = json.load(open(config_path, encoding="utf-8"))["activation"]
        self.decay_f = cfg["decay"]
        self.lam = cfg["spread_lambda"]
        self.hops = cfg["hops"]
        self.window = cfg.get("window", 5)
        self.tau = cfg.get("tau", 2.0)
        from dataloader import Dataloader
        self.idf = Dataloader(config_path).load_idf()["idf"]        # váha uzlu / vodivost hrany
        self.default_idf = max(self.idf.values()) if self.idf else 1.0
        self.words = {}                       # lemma → teplo
        self.files = {}                       # doc → teplo
        self.adj = defaultdict(lambda: defaultdict(float))    # lemma → {soused: vodivost}

    def build_graph(self, corpus, grammar):
        """Postaví graf VODIVOSTÍ: okenní ko-okurence, hrana = idf(a)×idf(b)×exp(−d/tau).

        Ne jen sousedé (bigram), ale slova v OKNĚ jedné věty — tak se `pes` spojí se `šelma`
        (5 slov od sebe, ale v okně). Vodivost = idf×idf (funkční slova světlo nevedou) ×
        proximity (bližší silněji). Symetrická. Vstup: mountovaný `corpus`, `grammar`.
        """
        for rec in corpus.values():
            for sent in rec["sentences"]:
                content = [grammar.canon_lemma(t).lower() for t in sent if t["upos"] != "PUNCT"]
                n = len(content)
                for i in range(n):
                    wa = self.idf.get(content[i], self.default_idf)
                    for j in range(i + 1, min(i + 1 + self.window, n)):
                        cond = wa * self.idf.get(content[j], self.default_idf) \
                            * math.exp(-(j - i - 1) / self.tau)    # vodivost = idf×idf×proximity
                        self.adj[content[i]][content[j]] += cond
                        self.adj[content[j]][content[i]] += cond

    def feed(self, question_lemmas, dataloader):
        """Tah rozhovoru → rozsvícení: uzly slov dostanou VÁHU = idf + soubory (② brána).

        Obsahová slova (vysoké idf) svítí jasně, funkční (nízké idf) slabě — světlo je od
        začátku zacílené. Vstup: lemmata otázky, `dataloader`. Kumuluje se do pole.
        """
        for lem in question_lemmas:
            l = lem.lower()
            self.words[l] = self.words.get(l, 0.0) + self.idf.get(l, self.default_idf)
        for doc, score in dataloader.select_files(question_lemmas):
            self.files[doc] = self.files.get(doc, 0.0) + score

    def spread(self, hops=None):
        """Teplo teče po hranách × VÁHA (normalizovaná), `hops` skoků, útlum `spread_lambda`.

        Hot slovo přihřeje sousedy úměrně váze hrany — light-beam / mesh.
        """
        for _ in range(hops if hops is not None else self.hops):
            add = defaultdict(float)
            for lem, heat in list(self.words.items()):
                nbrs = self.adj.get(lem)
                if not nbrs:
                    continue
                tot = sum(nbrs.values())
                for nb, w in nbrs.items():
                    add[nb] += heat * self.lam * (w / tot)
            for nb, h in add.items():
                self.words[nb] = self.words.get(nb, 0.0) + h

    def decay(self):
        """Útlum mezi tahy — teplo × decay (slova i soubory)."""
        for store in (self.words, self.files):
            for k in list(store):
                store[k] *= self.decay_f

    def weight_answer(self, answer, doc):
        """Aktivace kandidáta = teplo jeho souboru + teplo jeho odpovědní entity (po spreadu).

        Jádro sdílené oběma cestami matcheru (predikát+role i window-VZOR) — kandidát je
        vždy (odpověď, doc). Vstup: lemma odpovědi, doc. Výstup: teplo (float).
        """
        return self.files.get(doc, 0.0) + self.words.get((answer or "").lower(), 0.0)

    def weight(self, tpl):
        """Aktivace šablony (window-VZOR) — DRY nad `weight_answer`."""
        return self.weight_answer(tpl.answer, tpl.fact_ref[0] if tpl.fact_ref else None)

    def reinforce(self, answer, doc):
        """Zpětný tok: odpověď přihřeje svou entitu a soubor (kontext drží řetěz)."""
        if answer:
            self.words[answer.lower()] = self.words.get(answer.lower(), 0.0) + 1.0
        if doc:
            self.files[doc] = self.files.get(doc, 0.0) + 1.0


if __name__ == "__main__":
    import pickle
    from grammar_vzor import GrammarVzor
    from dataloader import Dataloader
    g = GrammarVzor()
    dl = Dataloader()
    corpus = dl.mount(["wiki_karel_čapek", "wiki_pes_domácí"])
    field = ActivationField()
    field.build_graph(corpus, g)
    field.feed(["spisovatel", "drama"], dl)
    field.spread()
    hot = sorted(field.words.items(), key=lambda x: -x[1])[:8]
    print("horká slova po spreadu:", [(w, round(h, 2)) for w, h in hot])
    print("horké soubory:", {d: round(h, 2) for d, h in field.files.items()})
