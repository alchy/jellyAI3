#!/usr/bin/env python3
"""② DATALOADER — fragmentové indexy + nahrávání souborů do grafu podle matche.

Ze shardů vrstvy ① (`data/corpus/<doc>.pkl`) staví pro KAŽDÝ soubor tf·idf index
(slovo → váha vůči ostatním souborům = četnost × unikátnost termínu) a globální idf
tabulku. Za běhu podle slov rozhovoru vybere nejbližší soubory a nahraje (mount) jen
je — koncept #60 fragmentový graf, ne celý korpus naráz.

`build_indexes()` přepočítá VŠECHNY staty z aktuální množiny shardů. Přidání obsahu:
① zanotuje nový raw soubor (nový shard) → zde `build_indexes()` a idf i všechny
per-soubor indexy jsou aktuální.

Uložení (shardovatelné — past #60 F4, unese tisíce souborů):
  data/index/<doc>.pkl   per-soubor index {lemma: váha}, L2-normovaný
  data/index/_idf.pkl    {"idf": {lemma: idf}, "n": počet_dokumentů}
"""
import os
import json
import glob
import math
import pickle
from collections import Counter

from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class Dataloader:
    """Staví a čte fragmentové indexy a nahrává soubory (shardy) do grafu.

    Instance drží cesty (corpus_dir/index_dir) a prahy z configu. Indexy jsou
    per-soubor na disku a čtou se líně, takže mechanismus unese tisíce souborů:
    studený soubor = jen index na disku, horký se namountuje do grafu.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte blok `dataloader` z JSON configu (cesty ke shardům a indexům, prahy)."""
        self.cfg = json.load(open(config_path, encoding="utf-8"))["dataloader"]
        self.corpus_dir = os.path.join(HERE, self.cfg["corpus_dir"])
        self.index_dir = os.path.join(HERE, self.cfg["index_dir"])

    # ---- shardy (výstup ①) --------------------------------------------------

    def document_ids(self):
        """Vrátí seřazená id všech dostupných souborů (shardů) v corpus_dir.

        Čte jen názvy souborů, nenačítá jejich obsah — levné i u tisíců souborů.
        """
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(self.corpus_dir, "*.pkl")))

    def load_shard(self, doc_id):
        """Načte shard jednoho souboru z disku.

        Vrací záznamy toho souboru ve tvaru `{(doc_id, idx): {entities, sentences}}`
        — přesně jak je zapsala vrstva ①.
        """
        return pickle.load(open(os.path.join(self.corpus_dir, f"{doc_id}.pkl"), "rb"))

    @staticmethod
    def _lemma_counts(records):
        """Spočítá výskyty lemmat v jednom shardu (přeskočí interpunkci a 1-znaky).

        Vrací `Counter` lemma→počet; lemma se sjednocuje na malá písmena, aby
        „Čapek" a „čapek" byly týž term.
        """
        c = Counter()
        for rec in records.values():
            for sent in rec["sentences"]:
                for t in sent:
                    lem = t["lemma"].lower()
                    if t["upos"] == "PUNCT" or len(lem) < 2:
                        continue
                    c[lem] += 1
        return c

    # ---- indexy = staty pro dynamické nahrávání -----------------------------

    def build_indexes(self):
        """Přepočítá VŠECHNY staty pro dynamické nahrávání z aktuálních shardů.

        Pro každý soubor spočítá tf, z celého korpusu df→idf a uloží per-soubor
        L2-normovaný tf·idf index plus globální idf tabulku. Je idempotentní nad
        aktuální množinou shardů — po přidání raw dat (① nový shard) stačí zavolat
        znovu a idf i všechny per-soubor indexy se aktualizují (idf závisí na počtu
        souborů, proto se přepočítává vše, ne jen nový soubor).
        """
        os.makedirs(self.index_dir, exist_ok=True)
        docs = self.document_ids()
        n = max(len(docs), 1)
        tf = {d: self._lemma_counts(self.load_shard(d)) for d in docs}
        df = Counter()
        for d in docs:
            for lem in tf[d]:
                df[lem] += 1
        idf = {lem: math.log(n / dfi) + 1.0 for lem, dfi in df.items()}
        min_tf = self.cfg["min_count"]           # práh výskytů V SOUBORU (šum hapaxů)
        for d in docs:
            weights = {lem: cnt * idf[lem] for lem, cnt in tf[d].items() if cnt >= min_tf}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            weights = {lem: w / norm for lem, w in weights.items()}
            pickle.dump(weights, open(os.path.join(self.index_dir, f"{d}.pkl"), "wb"))
        pickle.dump({"idf": idf, "n": n},
                    open(os.path.join(self.index_dir, "_idf.pkl"), "wb"))
        # PODMĚTOVÝ index: kolikrát je entita PODMĚTEM v souboru (× idf, L2-norm). Soubor se
        # má rozsvítit podle entity JAKO PODMĚTU — „Slovo" je subjekt Jana, ne četnost v bibli.
        subj = {}
        for d in docs:
            sc = Counter()
            for rec in self.load_shard(d).values():
                for sent in rec["sentences"]:
                    for t in sent:
                        if t["deprel"] in ("nsubj", "nsubj:pass") and t["upos"] in ("NOUN", "PROPN"):
                            lem = t["lemma"].lower()
                            if len(lem) >= 2:
                                sc[lem] += 1
            subj[d] = dict(sc)                    # RAW počty (idf entity je konstantní → stačí count)
        pickle.dump(subj, open(os.path.join(self.index_dir, "_subjects.pkl"), "wb"))
        logger("i", f"indexy: {n} souborů, {len(idf)} termů, práh výskytů {min_tf}")
        return {"documents": n, "terms": len(idf)}

    def _load_subjects(self):
        """Načte podmětový index `{doc: {lemma: váha}}` (líně, cache)."""
        if not hasattr(self, "_subjects"):
            path = os.path.join(self.index_dir, "_subjects.pkl")
            self._subjects = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
        return self._subjects

    def load_idf(self):
        """Načte globální idf tabulku: `{"idf": {lemma: idf}, "n": počet}`."""
        return pickle.load(open(os.path.join(self.index_dir, "_idf.pkl"), "rb"))

    def _load_index(self, doc_id):
        """Načte per-soubor tf·idf index (líně, jen když je potřeba)."""
        return pickle.load(open(os.path.join(self.index_dir, f"{doc_id}.pkl"), "rb"))

    # ---- výběr + mount podle matche -----------------------------------------

    def select_files(self, query_lemmas, top_files=None):
        """Seřadí soubory podle shody s dotazem (Σ teplo(slovo) × váha_souboru).

        `query_lemmas` je iterable lemmat, nebo dict lemma→teplo (žhavost z těžiště).
        Vrací `[(doc_id, skóre)]` sestupně, jen soubory se skóre > 0, deterministicky
        (remízy dle jména). To je ta brána: distinktivní slova nasměrují správné soubory.
        """
        heat = query_lemmas if isinstance(query_lemmas, dict) \
            else Counter(l.lower() for l in query_lemmas)
        heat = {k.lower(): v for k, v in heat.items()}
        scored = []
        for d in self.document_ids():
            idx = self._load_index(d)
            s = sum(w * idx.get(lem, 0.0) for lem, w in heat.items())       # tf·idf (četnost)
            if s > 0:
                scored.append((d, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        result = scored[:(top_files or self.cfg["top_files"])]
        # ENTITA JAKO PODMĚT: přidej soubor, kde je slovo dotazu NEJSILNĚJŠÍM podmětem
        # („Slovo" → Jan, i když tf·idf bralo Korintským). Retrieval (guard+def_fit) doostří.
        subj = self._load_subjects()
        subj_min = self.cfg.get("subject_min", 3)
        picked = {d for d, _ in result}
        for lem in heat:
            best_cnt, best_d = 0, None
            for d, sc in subj.items():
                if sc.get(lem, 0) > best_cnt:
                    best_cnt, best_d = sc[lem], d
            if best_cnt >= subj_min and best_d not in picked:
                result.append((best_d, 0.0))
                picked.add(best_d)
        return result

    def mount(self, doc_ids):
        """Nahraje shardy zadaných souborů a sloučí je do jednoho korpusu pro graf.

        Vrací `{(doc_id, idx): rec}` přes všechny žádané soubory — vstup pro stavbu
        slovního grafu (jas + hrany; ta logika žije v `run.py` a při rozdělení se sem
        přenese). Unmount = tento slovník nedržet / zahodit fakty se `source=doc`.
        """
        corpus = {}
        for d in doc_ids:
            corpus.update(self.load_shard(d))
        return corpus


if __name__ == "__main__":
    dl = Dataloader()
    dl.build_indexes()
    for q in (["spisovatel", "drama", "hra"], ["faraón", "Hospodin", "Mojžíš"]):
        print(q, "→", dl.select_files(q, top_files=3))
