#!/usr/bin/env python3
"""④ SYNTHESIS — query šablony (fakt/tvrzení → 1-n otázek → VZORy) + completeness.

Pro každý fakt vygeneruje Ollama OFFLINE 1-n parafrází otázek na jeho odpovědní sloty;
každá otázka se naparsuje UDPipe 2 na svůj VZOR a určí se, NA CO se ptá (role tázacího
slova = role jako u kteréhokoli tokenu, přes `RoleCatalog`). Vazba
{tvrzení → [query(surface, role, VZOR)]} je pak DB pro standalone match za běhu.

COMPLETENESS: helper ověří, že otázkový set pokrývá VŠECHNY dotazovatelné role faktu —
chybí-li role, set je nekompletní (Ollama driftla / chybí varianta). Kvalitní brána.

Ollama i UDPipe 2 jsou BUILD-TIME (offline). Zapisuje `query_templates.jsonl`.
Vyžaduje Ollama :11434 + UDPipe 2 REST.
"""
import os
import json

from grammar_vzor import GrammarVzor
from role_catalog import RoleCatalog
from annotate_corpus import AnnotateCorpus
from template_store import TemplateStore, QueryTemplate
from logger import logger
import ollama_iface as O

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class SynthQueries:
    """Staví query šablony z faktů registru + kontroluje completeness otázkového setu.

    Instance drží primitiva (`GrammarVzor`), rozklad rolí (`RoleCatalog`) a parser otázek
    (`AnnotateCorpus` = UDPipe 2). `query_of` je čistá analýza jedné otázky; `synthesize_fact`
    projde odpovědní sloty faktu a ověří pokrytí rolí.
    """

    def __init__(self, grammar=None, roles=None, config_path=CONFIG_PATH):
        """Vezme (nebo vytvoří) primitiva/role/parser a načte cestu výstupu."""
        self.g = grammar or GrammarVzor(config_path)
        self.rc = roles or RoleCatalog(self.g, config_path)
        self.parser = AnnotateCorpus(config_path)
        cfg = json.load(open(config_path, encoding="utf-8"))["synthesis"]
        self.out = os.path.join(HERE, cfg["queries"])

    def _parse(self, question):
        """Naparsuje jednu otázku UDPipe 2 → věta tokenů (nebo [])."""
        res = self.parser._udpipe2(question)
        if not res:
            return []
        sents, _ = AnnotateCorpus.parse_conllu(res)
        return sents[0] if sents else []

    @staticmethod
    def _interrogative(tokens):
        """Najde tázací token (rys PronType obsahuje „Int") → (index, token) / (None, None)."""
        for i, t in enumerate(tokens):
            if "Int" in (t.get("feats") or {}).get("PronType", ""):
                return i, t
        return None, None

    def query_of(self, question):
        """Analýza jedné otázky → {surface, role (na co se ptá), vzor}.

        Role otázky = role jejího tázacího slova, počítaná TÝMŽ mapovačem jako u faktu
        (`RoleCatalog.standard_role`), takže „Kdo…" = who, „Co…"(Acc) = whom_what. Bez
        tázacího slova (ANO/NE) → role None.
        """
        toks = self._parse(question)
        i, t = self._interrogative(toks)
        if i is None:
            return {"surface": question, "role": None, "vzor": None}
        role = self.rc.standard_role(t, toks, self.rc._byid(toks))
        mod = self.g.sentence_modality(toks)
        return {"surface": question, "role": role, "vzor": self.g.frame_sig(toks, i, mod)}

    @staticmethod
    def completeness(essential, queries):
        """Porovná dotazovatelné role faktu s rolemi, které otázkový set pokrývá.

        Vstup: `essential` (množina rolí faktu), `queries` (list z `query_of`). Výstup:
        {essential, covered, missing, complete}. `complete` = všechny essential role pokryty.
        """
        covered = {q["role"] for q in queries if q["role"]}
        return {"essential": sorted(essential), "covered": sorted(essential & covered),
                "missing": sorted(essential - covered), "complete": essential <= covered}

    def synthesize_fact(self, fact, n=2):
        """Pro fakt vygeneruje query per odpovědní slot a ověří completeness.

        Pro každý slot (role, lemma) požádá Ollamu o `n` parafrází otázek, každou naparsuje
        na VZOR + roli. Vrací vazbu {fact, predicate, queries[], completeness}.
        """
        essential = {a["role"] for a in fact["answers"]
                     if a["role"] and not a["role"].startswith("_")}
        queries = []
        for a in fact["answers"]:
            if not a["role"]:
                continue
            # šablona = FAKT-VZOR (díra, z registru) × QUERY-VZOR × ODPOVĚĎ (hodnota slotu)
            for q in O.gen_questions(fact["text"], a["lemma"], n):
                queries.append({**self.query_of(q), "target": a["role"],
                                "answer": a["lemma"], "fact_vzor": a.get("fact_vzor")})
        return {"fact": fact["text"], "predicate": fact["predicate"],
                "queries": queries, "completeness": self.completeness(essential, queries)}

    def _deterministic_vzor(self, fact_vzor, role):
        """Deterministická query-VZOR = TRANSPOZICE fakt-VZORu: pivot (odpověď) → tázací slot,
        modalita → „?". Bez Ollamy — a přesně to, co dá parse reálné otázky té struktury.

        Je to TÝŽ princip transpozice jako mezivrstva v koreferenci (dosadit do role díry) —
        jedno primitivum sjednocuje generování otázky i rozřešení díry.
        """
        parts = fact_vzor.split("·")
        r = self.g.radius
        islot = self.g.LANG.get("interrogative_slots", {}).get(role)
        if not islot or len(parts) < r + 2:
            return None
        parts[r] = islot          # pivot (odpovědní slot) → tázací slot
        parts[-1] = "?"           # modalita → otázka
        return "·".join(parts)

    def run(self, facts, store=None, n=2):
        """Postaví query šablony a zapíše je přes `TemplateStore` (shardy per soubor).

        Na každou díru VŽDY deterministickou šablonu (transpozice fakt-VZORu → garantuje
        completeness i bez Ollamy) + Ollama parafráze (povrchová variabilita). Vrací
        (store, množina distinktních query-VZORů) pro měření saturace.
        """
        store = store or TemplateStore()
        vzors, det, oll = set(), 0, 0
        for fact in facts:
            doc, sent = fact["doc"], fact["sent"]
            for a in fact["answers"]:          # deterministické (garantují pokrytí rolí)
                if not a.get("role") or not a.get("fact_vzor"):
                    continue
                qvz = self._deterministic_vzor(a["fact_vzor"], a["role"])
                if qvz and store.append(doc, QueryTemplate(
                        a["fact_vzor"], qvz, a["role"], a["lemma"], None,
                        [doc, sent], "deterministic")):
                    vzors.add(qvz); det += 1
            for q in self.synthesize_fact(fact, n)["queries"]:   # Ollama (variabilita povrchů)
                if q["vzor"] and store.append(doc, QueryTemplate(
                        q["fact_vzor"], q["vzor"], q["role"], q["answer"], q["surface"],
                        [doc, sent], "ollama")):
                    vzors.add(q["vzor"]); oll += 1
        logger("i", f"šablony: {len(facts)} faktů → deterministic {det} + ollama {oll}, "
                     f"distinktních query-VZORů {len(vzors)}")
        return store, vzors


if __name__ == "__main__":
    sq = SynthQueries()
    facts = []
    for line in open(os.path.join(HERE, "registry.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        if e["doc"] in ("wiki_karel_čapek", "wiki_pes_domácí", "bible_jan") and len(e["answers"]) >= 2:
            facts.append(e)
        if len(facts) >= 3:
            break
    for b in [sq.synthesize_fact(f, 2) for f in facts]:
        print(f"\nFAKT: {b['fact'][:70]}")
        print(f"  completeness: {b['completeness']}")
        for q in b["queries"]:
            print(f"    [{q['target']}→{q['role']}] {q['surface']}")
