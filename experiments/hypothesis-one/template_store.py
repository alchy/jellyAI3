#!/usr/bin/env python3
"""Persistentní šablony FAKT-VZOR × QUERY-VZOR × ODPOVĚĎ + VZOR matcher.

`QueryTemplate` je jeden shadow-pár (řádek). `TemplateStore` je sharduje per zdrojový
soubor (`data/templates/<doc>.jsonl`, jako korpus) a DLE OTÁZKY nahrává jen horké soubory
(mount, koncept #60) — matcher nikdy nedrží celý korpus šablon. Match je primárně dle
query-VZORu (stín otázky), s fuzzy fallbackem přes překryv slotů. `stats()` měří pokrytí.

Persistence je append-only (event-log, mergeable), deduplikace dle `id`.
"""
import os
import json
import hashlib
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config", "config.json")


class QueryTemplate:
    """Jeden shadow-pár: VZOR faktu (díra) × VZOR otázky × ODPOVĚĎ + provenience.

    `id` je stabilní (hash z fact_ref+role+query_vzor), takže se týž pár nezdvojí.
    `as_row`/`from_row` serializují do/z JSONL řádku.
    """

    __slots__ = ("fact_vzor", "query_vzor", "role", "answer", "surface_q", "fact_ref", "origin")

    def __init__(self, fact_vzor, query_vzor, role, answer, surface_q, fact_ref, origin):
        self.fact_vzor = fact_vzor
        self.query_vzor = query_vzor
        self.role = role
        self.answer = answer
        self.surface_q = surface_q
        self.fact_ref = tuple(fact_ref) if fact_ref else None
        self.origin = origin

    @property
    def id(self):
        """Stabilní identita (hash) — deduplikuje týž shadow-pár napříč běhy."""
        key = f"{self.fact_ref}|{self.role}|{self.query_vzor}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def as_row(self):
        """→ dict pro JSONL řádek (včetně id)."""
        return {"id": self.id, "fact_vzor": self.fact_vzor, "query_vzor": self.query_vzor,
                "role": self.role, "answer": self.answer, "surface_q": self.surface_q,
                "fact_ref": list(self.fact_ref) if self.fact_ref else None, "origin": self.origin}

    @classmethod
    def from_row(cls, d):
        """← dict z JSONL řádku."""
        return cls(d["fact_vzor"], d["query_vzor"], d["role"], d["answer"],
                   d["surface_q"], d["fact_ref"], d["origin"])


class TemplateStore:
    """Persistuje, mountuje a matchuje `QueryTemplate` šablony (VZOR matcher).

    Šablony jsou per soubor na disku; do paměti (`mounted`) se dostanou jen mountnuté
    (horké) soubory. Indexy `by_query_vzor`/`by_fact_vzor` se staví nad mountnutými.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte cestu šardů a práh fuzzy z configu; prázdný mount."""
        cfg = json.load(open(config_path, encoding="utf-8"))["template_store"]
        self.dir = os.path.join(HERE, cfg["dir"])
        self.fuzzy_min = cfg.get("fuzzy_min_overlap", 0.5)
        self.mounted = {}                 # doc -> [QueryTemplate]
        self.by_query_vzor = defaultdict(list)
        self.by_fact_vzor = defaultdict(list)
        self._seen = set()                # id zapsané v této build-session (dedup)

    # ---- persistence (build-time) -------------------------------------------

    def reset(self):
        """Smaže všechny šardy (čerstvý build)."""
        if os.path.isdir(self.dir):
            for f in os.listdir(self.dir):
                if f.endswith(".jsonl"):
                    os.remove(os.path.join(self.dir, f))
        self._seen.clear()

    def append(self, doc, tpl):
        """Připíše šablonu do `<doc>.jsonl` (append-only), dedup dle id v rámci session.

        Vstup: doc (str), tpl (QueryTemplate). Vrací True když zapsáno (jinak duplicitní).
        """
        if tpl.id in self._seen:
            return False
        self._seen.add(tpl.id)
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, f"{doc}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(tpl.as_row(), ensure_ascii=False) + "\n")
        return True

    def document_ids(self):
        """Seznam dostupných šardů (souborů se šablonami)."""
        if not os.path.isdir(self.dir):
            return []
        return sorted(f[:-6] for f in os.listdir(self.dir) if f.endswith(".jsonl"))

    # ---- mount + index (runtime, jen horké soubory) -------------------------

    def mount(self, doc_ids):
        """Nahraje šardy zadaných souborů do paměti a přebuduje indexy (jen nad horkými).

        Studený soubor zůstává na disku; horký se namountuje. To je #60 dynamické nahrání.
        """
        for doc in doc_ids:
            path = os.path.join(self.dir, f"{doc}.jsonl")
            if doc in self.mounted or not os.path.exists(path):
                continue
            self.mounted[doc] = [QueryTemplate.from_row(json.loads(l))
                                 for l in open(path, encoding="utf-8") if l.strip()]
        self._reindex()

    def unmount(self, doc_ids):
        """Odebere šardy z paměti a přebuduje indexy."""
        for doc in doc_ids:
            self.mounted.pop(doc, None)
        self._reindex()

    def _reindex(self):
        self.by_query_vzor = defaultdict(list)
        self.by_fact_vzor = defaultdict(list)
        for tpls in self.mounted.values():
            for t in tpls:
                if t.query_vzor:
                    self.by_query_vzor[t.query_vzor].append(t)
                if t.fact_vzor:
                    self.by_fact_vzor[t.fact_vzor].append(t)

    # ---- match (VZOR matcher) -----------------------------------------------

    @staticmethod
    def _slots(vzor):
        return set(vzor.split("·")) if vzor else set()

    def match_scored(self, query_vzor):
        """[(QueryTemplate, mq)] — síla shody VZORu: exakt = 1.0, jinak fuzzy překryv slotů.

        `mq` (match quality) je STRUKTURNÍ base pro glow-orders-ties ve vrstvě ⑤: exaktní
        shoda query-VZORu bije fuzzy; mezi stejně silnými rozhodne až aktivace (glow).
        Fuzzy jen nad prahem `fuzzy_min`.
        """
        exact = self.by_query_vzor.get(query_vzor)
        if exact:
            return [(t, 1.0) for t in exact]
        qs = self._slots(query_vzor)
        if not qs:
            return []
        out = []
        for vz, tpls in self.by_query_vzor.items():
            ov = len(qs & self._slots(vz)) / len(qs | self._slots(vz))
            if ov >= self.fuzzy_min:
                out.extend((t, ov) for t in tpls)
        out.sort(key=lambda x: -x[1])
        return out

    def match(self, query_vzor):
        """Kandidáti pro query-VZOR (bez skóre) — DRY nad `match_scored`.

        Vrací list QueryTemplate; strukturně shodné šablony (stín) — mezi nimi rozhodne
        až vrstva ⑤ (glow-orders-ties nad `match_scored`).
        """
        return [t for t, _mq in self.match_scored(query_vzor)]

    def match_fact(self, fact_vzor):
        """Kandidáti dle fakt-VZORu (druhý směr — strukturální stín otázka ≈ fakt)."""
        return list(self.by_fact_vzor.get(fact_vzor, []))

    # ---- statistiky ---------------------------------------------------------

    def stats(self, saturation=None):
        """Spočte statistiky nad mountnutými šablonami (+ zapíše `_meta.json`).

        Měří: počet šablon, distinktní query/fakt VZORy, rozdělení rolí a origin,
        počet faktů, ⌀ šablon/fakt, top query-VZORy. `saturation` (křivka z buildu) se
        jen předá do meta.
        """
        tpls = [t for lst in self.mounted.values() for t in lst]
        qv = Counter(t.query_vzor for t in tpls if t.query_vzor)
        facts = {t.fact_ref for t in tpls}
        meta = {
            "templates": len(tpls),
            "distinct_query_vzor": len(qv),
            "distinct_fact_vzor": len({t.fact_vzor for t in tpls if t.fact_vzor}),
            "roles": dict(Counter(t.role for t in tpls)),
            "origin": dict(Counter(t.origin for t in tpls)),
            "facts": len(facts),
            "avg_per_fact": round(len(tpls) / max(len(facts), 1), 2),
            "top_query_vzor": qv.most_common(10),
        }
        if saturation is not None:
            meta["saturation"] = saturation
        os.makedirs(self.dir, exist_ok=True)
        json.dump(meta, open(os.path.join(self.dir, "_meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return meta


if __name__ == "__main__":
    # round-trip self-test (bez sítě)
    s = TemplateStore()
    s.reset()
    t = QueryTemplate("PROPN:Nom·VERB:Past·PROPN:Acc·.", "^·PRON:Nom·VERB:Past·?",
                      "who", "Čapek", "Kdo napsal R.U.R.?", ["wiki_r.u.r.", 3], "ollama")
    s.append("wiki_r.u.r.", t)
    s.append("wiki_r.u.r.", t)                     # duplicitní → přeskočí
    s.mount(["wiki_r.u.r."])
    print("id:", t.id)
    print("match exakt:", [x.answer for x in s.match("^·PRON:Nom·VERB:Past·?")])
    print("match fuzzy:", [x.answer for x in s.match("^·PRON:Nom·VERB:Pres·?")])
    print("stats:", s.stats())
