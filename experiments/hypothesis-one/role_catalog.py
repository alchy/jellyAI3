#!/usr/bin/env python3
"""③ GRAMMAR — ROZKLAD věty do katalogu rolí (surové UDPipe → NÁŠ klíč).

Tady se surová anotace (deprel/upos/feats) NORMALIZUJE na náš jazykově nezávislý
rolový katalog (who/where/action/…). Klíče jsou univerzální; české popisy i mapování
předložek/spojek jsou DATA v `lang/cs.json`. Primitiva (slot/frame_sig/role_key)
bere z `GrammarVzor` (③).

Rozklad je PER KLAUZULE (každý přísudek = jedna klauzule). Vedlejší (relativní)
klauzule dědí místo z řídící věty — meziklauzulová inference (#59). Kurátorská DB
(`curated.json`) přebíjí role u chyb anotátoru podle VZORu okna.

Pracuje nad UŽ ANOTOVANOU větou (token = WORD_W_ATTR z korpusu ①). Živé parsování
otázky a výběr odpovědi sem NEpatří — to je vrstva ⑤.
"""
import os
import json

from grammar_vzor import GrammarVzor

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class RoleCatalog:
    """Rozkládá anotovanou větu na sektory (tokeny) s NAŠÍM rolovým klíčem.

    Instance drží jazyková data (přes `GrammarVzor`) a kurátorskou DB. Metody jsou
    čisté nad jednou větou; sdílejí jediný mapovač `GrammarVzor.role_key`, takže
    stejný význam = stejný klíč napříč systémem.
    """

    PRED_ACL = ("acl", "acl:relcl", "advcl")   # vedlejší klauzule závislé na řídící
    ARG_DEPREL = ("nsubj", "nsubj:pass", "obj", "iobj", "obl", "obl:arg", "obl:agent")

    def __init__(self, grammar=None, config_path=CONFIG_PATH):
        """Vezme (nebo vytvoří) `GrammarVzor` a načte kurátorskou DB z configu.

        Jazyková data i cesta ke `curated.json` jsou z JSON — nic natvrdo.
        """
        self.g = grammar or GrammarVzor(config_path)
        self.LANG = self.g.LANG
        self.REL_PLACE = set(self.LANG["relative_place_adverbs"])
        self.REL_TIME = set(self.LANG["relative_time_adverbs"])
        self.PASS_SUF = tuple(self.LANG["participle_passive_suffixes"])
        cfg = json.load(open(config_path, encoding="utf-8"))
        curated = json.load(open(os.path.join(HERE, cfg["paths"]["curated"]), encoding="utf-8"))
        self._curated = {k: v["role"] for k, v in curated.items() if not k.startswith("_")}

    # ── pomůcky nad jednou větou (1-based id = index+1, head=0 → kořen) ────────
    @staticmethod
    def _byid(sent):
        return {i + 1: t for i, t in enumerate(sent)}

    @staticmethod
    def _cop_heads(sent):
        return {t["head"] for t in sent if t["deprel"] == "cop"}

    @staticmethod
    def _is_pred(t, cop_h, tid):
        return t["upos"] == "VERB" or (t["upos"] in ("NOUN", "ADJ", "PROPN") and tid in cop_h)

    def _participle_key(self, t):
        if t.get("feats", {}).get("VerbForm") != "Part":
            return None
        if t["feats"].get("Voice") == "Pass" or t["form"].lower().endswith(self.PASS_SUF):
            return "passive_participle"
        return "past_participle"

    # ── role jednoho tokenu → klíč katalogu (jádro sdílí GrammarVzor.role_key) ──
    def role_of(self, t, sent, byid):
        """Obsahová role jednoho tokenu (who/where/whom_what/…), nebo None.

        Souřadný konjunkt dědí roli řídícího členu; argumentové deprel jdou přes
        `GrammarVzor.role_key`; příslovce se dělí na místo/čas/způsob.
        """
        dep, up, f = t["deprel"], t["upos"], t.get("feats", {})
        if dep == "conj" and t["head"] in byid:
            return self.role_of(byid[t["head"]], sent, byid)
        if dep in self.ARG_DEPREL:
            prep = self.g.prep_of(sent, sent.index(t) + 1)
            return self.g.role_key("obl" if dep == "obl:agent" else dep, up, f, prep)
        if dep == "advmod":
            if t["lemma"] in self.REL_PLACE:
                return "where"
            if t["lemma"] in self.REL_TIME:
                return "when"
            return "how"
        if dep in ("amod", "nmod", "nmod:poss", "det", "det:poss"):
            return "which_attribute"
        if dep == "xcomp":
            return "as_what_state"
        return None

    def standard_role(self, t, sent, byid, cop_heads=None):
        """Standardizovaný klíč JEDNOHO sektoru — vždy v našem katalogu, ne v deprel.

        VERB → action, jmenný přísudek se sponou → state, strukturální deprel →
        strukturální klíč (preposition/conjunction/punctuation…), jinak obsahová
        role. Výstup 1. fáze je vždy náš klíč.
        """
        if cop_heads is None:
            cop_heads = self._cop_heads(sent)
        up, dep = t["upos"], t["deprel"]
        tid = sent.index(t) + 1
        if up == "VERB":
            return "action"
        if tid in cop_heads:
            return "state"
        struct = self.LANG["deprel_structural"].get(dep)
        if struct:
            return struct
        return self.role_of(t, sent, byid)

    def standardize(self, sent):
        """Rozklad věty na sektory: [(token, náš klíč)] — plně v katalogu, bez deprel.

        Vstup: anotovaná věta (list tokenů). Výstup: seznam dvojic (token, klíč).
        """
        byid = self._byid(sent)
        cop_heads = self._cop_heads(sent)
        return [(t, self.standard_role(t, sent, byid, cop_heads)) for t in sent]

    def curated_standardize(self, sent, r=2):
        """Standardizace S kurátorskou opravou: [(token, klíč, kurátorováno?)].

        Spočte VZOR okna; je-li v kurátorské DB → OPRAVENÁ role (ruční, konzistentní),
        jinak `standard_role`. Tak DB roste a známé bereme z ní.
        """
        byid = self._byid(sent)
        mod = self.g.sentence_modality(sent)
        cop_heads = self._cop_heads(sent)
        out = []
        for i, t in enumerate(sent):
            if t["upos"] == "PUNCT":
                out.append((t, self.standard_role(t, sent, byid, cop_heads), False))
                continue
            vz = self.g.frame_sig(sent, i, mod, r)
            if vz in self._curated:
                out.append((t, self._curated[vz], True))
            else:
                out.append((t, self.standard_role(t, sent, byid, cop_heads), False))
        return out

    def decompose(self, sent):
        """Rozloží větu na KLAUZULE (přísudek + jeho role) s meziklauzulovou inferencí.

        Každý přísudek = jedna klauzule; jeho argumenty se roztřídí do rolí. Relativní
        klauzule dědí místo z antecedentu (#59). Vstup: anotovaná věta. Výstup: seznam
        klauzulí {predicate, form, deprel, head, roles}.
        """
        byid, cop_h = self._byid(sent), self._cop_heads(sent)

        def clause_of(tid):
            seen = set()
            while tid and tid not in seen:
                seen.add(tid)
                if self._is_pred(byid[tid], cop_h, tid):
                    return tid
                tid = byid[tid]["head"]
            return None

        clauses = {}
        for i, t in enumerate(sent):
            tid = i + 1
            if self._is_pred(t, cop_h, tid):
                c = clauses.setdefault(tid, {"predicate": None, "form": t["form"],
                                             "deprel": t["deprel"], "head": t["head"], "roles": {}})
                if t["upos"] == "VERB":
                    c["predicate"] = t["lemma"]
                    c["roles"]["action"] = [t["lemma"]]
                    pk = self._participle_key(t)
                    if pk:
                        c["roles"].setdefault(pk, []).append(t["form"])
                else:
                    c["predicate"] = self.LANG["copula_lemma"]
                    c["roles"]["state"] = [t["lemma"]]
        for i, t in enumerate(sent):
            tid = i + 1
            if self._is_pred(t, cop_h, tid) or \
                    t["upos"] in ("PUNCT", "CCONJ", "SCONJ", "ADP", "AUX", "PART"):
                continue
            key = self.role_of(t, sent, byid)
            c = clause_of(byid[tid]["head"] if t["deprel"] == "root" else tid)
            if key and c in clauses:
                clauses[c]["roles"].setdefault(key, []).append(t["lemma"])
        for cid, c in clauses.items():
            if c["deprel"] in self.PRED_ACL and c["head"] in byid:
                ant = byid[c["head"]]
                where = c["roles"].get("where", [])
                if any(w in self.REL_PLACE for w in where) and ant["upos"] in ("NOUN", "PROPN"):
                    c["roles"]["where"] = [ant["lemma"] if w in self.REL_PLACE else w for w in where]
                    c["roles"]["_where_inherited"] = [ant["lemma"]]
        return list(clauses.values())


if __name__ == "__main__":
    import pickle
    rc = RoleCatalog()
    shard = pickle.load(open(os.path.join(HERE, "../../data/corpus/bible_lukas.pkl"), "rb"))
    # najdi větu s Kafarnaum pro sektorový rozklad
    for rec in shard.values():
        for sent in rec["sentences"]:
            if any(t["form"] == "Kafarnaum" for t in sent):
                print("věta:", " ".join(t["form"] for t in sent)[:80])
                for t, key in rc.standardize(sent):
                    print(f"  {t['form']:14} {t['upos']:6} {t['lemma']:12} "
                          f"{(t.get('feats') or {}).get('Case','—'):4} {key}")
                raise SystemExit
