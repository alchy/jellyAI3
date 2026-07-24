#!/usr/bin/env python3
"""Fakt-store — fakty (predikát + role-sloty) sharded per soubor, matchované PREDIKÁTEM.

PARENT-MODEL matching. Window-VZOR vidí jen lokální okno, takže otázku a fakt nesouměří:
„Kde se narodil X" (okno kolem fronted „Kde") ≠ „Narodil se v Y" (okno kolem místa uprostřed).
Predikát ale obě strany spojuje — obojí nese `narodit` + roli `where`. Doc-kontext supluje
pro-drop podmět (fakt bez explicitního podmětu žije v souboru O TÉ entitě).

Fakt = agregace answer-slotů registry (④ synth_registry) per (doc, sent, predikát):
`roles = {role: [lemmata]}`. Shardy `data/facts/<doc>.jsonl`, mount jen horkých (#60).
Match: predikát (kanon) + role díry → lemmata té role. Rozhodne až vrstva ⑤ (glow-orders-ties).
"""
import os
import json
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config", "config.json")


class Fact:
    """Jeden fakt: predikát + role-sloty + provenience (doc, sent, text)."""

    __slots__ = ("predicate", "roles", "doc", "sent", "text")

    def __init__(self, predicate, roles, doc, sent, text=None):
        self.predicate = predicate
        self.roles = roles                 # {role: [lemma, …]}
        self.doc = doc
        self.sent = sent
        self.text = text

    def as_row(self):
        """→ dict pro JSONL řádek."""
        return {"predicate": self.predicate, "roles": self.roles,
                "doc": self.doc, "sent": self.sent, "text": self.text}

    @classmethod
    def from_row(cls, d):
        """← dict z JSONL řádku."""
        return cls(d["predicate"], d["roles"], d["doc"], d["sent"], d.get("text"))


class FactStore:
    """Persistuje, mountuje a matchuje `Fact` fakty predikátem + rolí (parent-model).

    Shardy per soubor na disku; do paměti (`mounted`) jen horké (mount, #60). Index
    `by_predicate` se staví nad mountnutými. `match(predikát, role)` vrací kandidáty slotu.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte cestu shardů z configu; prázdný mount."""
        with open(config_path, encoding="utf-8") as config_file:
            cfg = json.load(config_file)
        self.dir = os.path.join(HERE, cfg.get("fact_store", {}).get("dir", "data/facts"))
        self.mounted = {}                       # doc -> [Fact]
        self.by_predicate = defaultdict(list)
        self.by_ref = {}                        # (doc, sent) -> Fact
        # Líný, pouze metadata index pro selektivní routing. Nevzniká jako nový
        # build artefakt a neobsahuje Fact objekty: drží jen odkazy do shardů.
        self._route_index = None

    # ---- persistence (build-time) -------------------------------------------

    def reset(self):
        """Smaže všechny shardy (čerstvý build)."""
        if os.path.isdir(self.dir):
            for f in os.listdir(self.dir):
                if f.endswith(".jsonl"):
                    os.remove(os.path.join(self.dir, f))

    def append(self, fact):
        """Připíše fakt do `<doc>.jsonl` (append-only)."""
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, f"{fact.doc}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(fact.as_row(), ensure_ascii=False) + "\n")

    def document_ids(self):
        """Seznam dostupných shardů (souborů s fakty)."""
        if not os.path.isdir(self.dir):
            return []
        return sorted(f[:-6] for f in os.listdir(self.dir) if f.endswith(".jsonl"))

    # ---- mount + index (runtime, jen horké soubory) -------------------------

    def mount(self, doc_ids):
        """Nahraje shardy zadaných souborů do paměti a přebuduje index (jen nad horkými)."""
        for doc in doc_ids:
            path = os.path.join(self.dir, f"{doc}.jsonl")
            if doc in self.mounted or not os.path.exists(path):
                continue
            self.mounted[doc] = [Fact.from_row(json.loads(l))
                                 for l in open(path, encoding="utf-8") if l.strip()]
        self._reindex()

    def unmount(self, doc_ids):
        """Odebere shardy z paměti a přebuduje index."""
        for doc in doc_ids:
            self.mounted.pop(doc, None)
        self._reindex()

    def _reindex(self):
        self.by_predicate = defaultdict(list)
        self.by_ref = {}
        for facts in self.mounted.values():
            for f in facts:
                self.by_predicate[f.predicate].append(f)
                self.by_ref[(f.doc, f.sent)] = f

    # ---- match (parent-model) -----------------------------------------------

    def match(self, predicate, hole_role):
        """[(lemma, Fact)] — fakty s daným predikátem nesoucí slot role díry.

        Predikát je JOIN klíč přeživší slovosled; role díry vybere slot. Mezi kandidáty
        (i napříč fakty) rozhodne až aktivace ve vrstvě ⑤ (glow-orders-ties). Bez predikátu
        nebo role → [].
        """
        if not predicate or not hole_role:
            return []
        out = []
        for f in self.by_predicate.get(predicate, []):
            for lem in f.roles.get(hole_role, []):
                out.append((lem, f))
        return out

    # ---- runtime entity routing (bez preprocessing artefaktu) --------------

    @staticmethod
    def _norm(value):
        return (value or "").strip().lower()

    def _build_route_index(self):
        """Postaví malý index `(predicate, answer_role, known_lemma) -> refs`.

        Index se staví až při prvním dotazu, čte řádky faktových shardů, ale
        nemountuje je ani nevytváří ``Fact`` objekty. ``known_lemma`` musí být
        v jiné roli než hledaná odpověď; tím odpovídá stejnému guardu jako
        parent-model v ``Answering._candidates``. Hodnotou je jen ``(doc,sent)``.
        """
        index = defaultdict(list)
        for doc in self.document_ids():
            path = os.path.join(self.dir, f"{doc}.jsonl")
            with open(path, encoding="utf-8") as fact_file:
                for line in fact_file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    predicate = self._norm(row.get("predicate"))
                    roles = row.get("roles") or {}
                    ref = (doc, row.get("sent"))
                    if not predicate:
                        continue
                    for answer_role in roles:
                        links = set()
                        for role, lemmata in roles.items():
                            if role != answer_role:
                                links.update(self._norm(lemma) for lemma in lemmata)
                        for lemma in links:
                            if lemma:
                                index[(predicate, answer_role, lemma)].append(ref)
        self._route_index = index

    def route_docs(self, predicate, hole_role, known, max_docs=4, max_fact_refs=24):
        """Vrátí omezené dokumenty s přímým strukturálním důkazem.

        Je to jen routing layer: z celého fact-store se načtou metadata, pak se
        vrátí nejvýše ``max_docs`` document shardů. Samotné fakta načte až
        ``mount`` v běžném runtime tahu. Přesné shody se řadí podle počtu
        referencí, remízy stabilně podle id dokumentu.
        """
        predicate, hole_role = self._norm(predicate), self._norm(hole_role)
        if not predicate or not hole_role:
            return []
        if self._route_index is None:
            self._build_route_index()
        refs = []
        matched_terms = defaultdict(set)
        for lemma in sorted({self._norm(value) for value in known if self._norm(value)}):
            term_refs = self._route_index.get((predicate, hole_role, lemma), ())
            refs.extend(term_refs)
            for doc, _sent in term_refs:
                matched_terms[doc].add(lemma)
        # Tatáž věta může být zasažena více lemmaty otázky; nesmí spotřebovat budget.
        refs = sorted(set(refs))[:max_fact_refs]
        counts = defaultdict(int)
        for doc, _sent in refs:
            counts[doc] += 1
        return [doc for doc, _count in sorted(
                counts.items(), key=lambda item: (-len(matched_terms[item[0]]), -item[1], item[0]))
                [:max_docs]]


if __name__ == "__main__":
    fs = FactStore()
    docs = fs.document_ids()
    print(f"shardy: {len(docs)}")
    if docs:
        fs.mount(["wiki_karel_čapek"])
        print("narodit×where:", fs.match("narodit", "where")[:5])
