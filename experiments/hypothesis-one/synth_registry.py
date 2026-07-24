#!/usr/bin/env python3
"""④ SYNTHESIS — grafikon FAKT ↔ OTÁZKA (registr).

Pro každou větu korpusu s přísudkem vyrobí syntetickou vazbu na fakt: predikát,
díra (podmět/účastník), odpovědní sloty (lemma + náš rolový klíč + rod) a kontext.
To jsou data, nad kterými běží match a light-beam. Odpovědi/kontext jsou v ZÁKLADNÍM
TVARU (kanonizace přes `GrammarVzor.canon_lemma` + podmíněný epentetický fold PROPN).

Konzumuje NOVÝ korpus (shardy ① přes `Dataloader.mount`), role bere jednotným
mapovačem `GrammarVzor.role_key` (týž klíč jako ③). Zapisuje `registry.jsonl`.
"""
import os
import json

from grammar_vzor import GrammarVzor
from dataloader import Dataloader
from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config", "config.json")


class SynthRegistry:
    """Staví registr syntetických vazeb FAKT↔OTÁZKA z anotovaného korpusu.

    Instance drží primitiva (`GrammarVzor`), zavaděč korpusu (`Dataloader`) a cesty.
    Jádro `_binding` je čistá funkce nad jednou větou; `run` projde celý korpus.
    """

    ANSWER_DEPREL = {"nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:arg"}
    CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "ADV"}

    def __init__(self, grammar=None, loader=None, config_path=CONFIG_PATH):
        """Vezme (nebo vytvoří) `GrammarVzor` a `Dataloader`; načte cestu registru."""
        self.g = grammar or GrammarVzor(config_path)
        self.dl = loader or Dataloader(config_path)
        cfg = json.load(open(config_path, encoding="utf-8"))
        self.registry_path = os.path.join(HERE, cfg["synthesis"]["registry"])

    @staticmethod
    def _skip(lem, upos):
        """Neslovo: interpunkce nebo jednopísmenná iniciála („T." v „T. G. Masaryk")."""
        return upos == "PUNCT" or (len(lem) == 1 and upos in ("NOUN", "PROPN", "SYM", "X"))

    def _propn_fold(self, corpus):
        """Podmíněný epentetický fold PROPN nad korpusem: Čapk→Čapek, ale JEN když
        „Čapek" reálně existuje a je častější (nekorumpuje Egypt→Egypet, Petr→Peter).

        Vstup: korpus {(doc,idx): rec}. Výstup: dict {lemma: složené_lemma}.
        """
        from collections import Counter
        propn = Counter()
        for rec in corpus.values():
            for sent in rec["sentences"]:
                for t in sent:
                    if t["upos"] == "PROPN":
                        propn[self.g.canon_lemma(t)] += 1
        fold = {}
        for lem, c in propn.items():
            cand = self.g._epen_stem(lem)
            if cand != lem and propn.get(cand, 0) > c:
                fold[lem] = cand
        return fold

    def _binding(self, sent, doc, si, fold):
        """Z jedné anotované věty vazba FAKT↔OTÁZKA, nebo None (žádný přísudek/odpověď).

        Predikát je root VERB (jinak spona = identitní fakt); odpovědní sloty jsou
        argumentová jména se svým rolovým klíčem, kontext jsou obsahová slova věty.

        Vstup: věta, doc_id, index věty, fold PROPN. Výstup: dict vazby / None.
        """
        if not self.g.is_factual(sent):          # nefaktická věta (hedging/nejistota) → žádný fakt
            return None
        canon = self.g.canon_lemma
        root = next((t for t in sent if t.get("deprel") == "root"), None)
        root_verb = next((t for t in sent if t.get("deprel") == "root" and t["upos"] == "VERB"), None)
        has_cop = any(t.get("deprel") == "cop" for t in sent)
        root_nom = None
        if root_verb is not None:
            predicate = fold.get(canon(root_verb), canon(root_verb))
            verb = root_verb
        elif root is not None and root["upos"] in ("NOUN", "ADJ", "PROPN") and has_cop:
            # KOPULA hlavní klauzule má PŘEDNOST před slovesem VEDLEJŠÍ věty:
            # „Králík je býložravec, který VYUŽÍVÁ…" → stav=býložravec (ne predikát využívat).
            predicate, root_nom, verb = self.g.LANG["copula_lemma"], root, None
        else:
            verb = next((t for t in sent if t["upos"] == "VERB"), None)   # fallback: jakékoli sloveso
            if verb is None:
                return None
            predicate = fold.get(canon(verb), canon(verb))
        mod = self.g.sentence_modality(sent)
        answers, context = [], []
        for i, t in enumerate(sent):
            lem = fold.get(canon(t), canon(t))
            if self._skip(lem, t["upos"]):
                continue
            fe = t.get("feats") or {}
            is_ans = (t.get("deprel") in self.ANSWER_DEPREL and t["upos"] in ("NOUN", "PROPN")) \
                or (t is root_nom)
            if is_ans:
                role = self.g.role_key(t.get("deprel"), t["upos"], fe,
                                       prep=self.g.prep_of(sent, i + 1),
                                       nominal_pred=(t is root_nom))
                # VZOR faktu v místě odpovědního slotu (díry) = ŠABLONA FAKTU
                answers.append({"lemma": lem, "role": role,
                                "q": self.g.LANG["role_ask"].get(role, "?"),
                                "gender": fe.get("Gender"), "name": bool(fe.get("NameType")),
                                "fact_vzor": self.g.frame_sig(sent, i, mod)})
            if t["upos"] in self.CONTENT_UPOS and t is not verb:
                context.append(lem)
        if not answers:
            return None
        return {"predicate": predicate, "hole": "subj/person",
                "answers": answers, "context": context, "doc": doc, "sent": si,
                "text": " ".join(t["form"] for t in sent)[:120]}

    def run(self):
        """Projde celý korpus a zapíše `registry.jsonl` (jedna vazba na řádek).

        Nejdřív spočte podmíněný fold PROPN nad celým korpusem, pak z každé věty
        vytvoří vazbu. Vrací souhrn (počet vazeb, unikátních predikátů, slotů).
        """
        from collections import Counter
        corpus = self.dl.mount(self.dl.document_ids())
        fold = self._propn_fold(corpus)
        logger("i", f"podmíněný fold PROPN: {len(fold)} dvojic")
        n = 0
        preds = Counter()
        ans_total = 0
        with open(self.registry_path, "w", encoding="utf-8") as out:
            for (doc, si), rec in corpus.items():
                for sent in rec["sentences"]:
                    e = self._binding(sent, doc, si, fold)
                    if e is None:
                        continue
                    out.write(json.dumps(e, ensure_ascii=False) + "\n")
                    n += 1
                    preds[e["predicate"]] += 1
                    ans_total += len(e["answers"])
        logger("i", f"vazeb FAKT↔OTÁZKA: {n}, unikátních predikátů: {len(preds)}, "
                     f"answer-slotů: {ans_total} (⌀ {ans_total/max(n,1):.1f}/fakt)")
        logger("i", f"nejčastější predikáty: "
                     + ", ".join(f"{p}×{c}" for p, c in preds.most_common(6)))
        return {"bindings": n, "predicates": len(preds), "answer_slots": ans_total}


if __name__ == "__main__":
    SynthRegistry().run()
