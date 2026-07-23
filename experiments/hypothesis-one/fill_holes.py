#!/usr/bin/env python3
"""Zaplnění DĚR (koreference) — placeholder se detekuje ÚFAL rysem, plní antecedentem.

Díra je zájmeno nebo elidovaný podmět ukazující na entitu nepojmenovanou v klauzuli.
Detekce jde JEN z rysu (PronType/Poss/Person/Tense), ne ze seznamu tvarů — ÚFAL
lemmatizuje ho/mu/jemu… na `on` a rozliší rysem, takže je to univerzální. Antecedent
se hledá backward cache v okně N vět zpět; shoda = překryv RODU (ÚFAL syncretismus =
množina) + číslo; podmět má přednost, pak recency (těžiště). Vyřešená díra se vrací
do cache (řetěz), takže navazující díry navazují na stejnou entitu.

Cíl: fakt/role pak nesou SKUTEČNOU entitu, ne „on" — předpoklad syntézy tvrzení↔query.
Detekce je hotová věc; resolver antecedenta je vlastní krok (měřený).
"""
import os
import json

from grammar_vzor import GrammarVzor

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class FillHoles:
    """Detekuje díry a plní je antecedentem přes backward cache + shodu rysu.

    Instance drží primitiva (`GrammarVzor` pro kanonizaci) a parametry z configu
    (okno vět zpět, slovní druhy kandidátů). `detect` je čistá nad jednou větou;
    `resolve_document` jede větu po větě s cache a plní díry typu antecedent.
    """

    def __init__(self, grammar=None, config_path=CONFIG_PATH):
        """Vezme (nebo vytvoří) `GrammarVzor` a načte okno + kandidáty z configu."""
        self.g = grammar or GrammarVzor(config_path)
        cfg = json.load(open(config_path, encoding="utf-8"))["fill_holes"]
        self.window = cfg["window_sentences"]
        self.cand_upos = set(cfg["candidate_upos"])
        self.decay = cfg["decay"]
        self.subj_weight = cfg["subject_weight"]
        self.reinforce = cfg["reinforce"]
        self.zone_beta = cfg["zone_beta"]
        self.anchor_bonus = cfg["anchor_bonus"]

    @staticmethod
    def _gset(g):
        """Rys Gender → množina (ÚFAL syncretismus: „Masc,Neut" → {Masc, Neut})."""
        return set(g.split(",")) if g else set()

    def detect(self, sent):
        """Detekuje díry v jedné větě → list {i, cat, target, g(set), num}.

        Kategorie (dle ÚFAL rysu): pron3 / person12 (osobní), poss3 / poss_reflex /
        poss12 (přivlastňovací), dem (ukazovací samostatné), prodrop (elidovaný podmět
        = přísudek bez nsubj). `target` = „antecedent" nebo speciál (mluvčí/podmět).
        """
        subj_heads = {t["head"] for t in sent if t["deprel"] in ("nsubj", "nsubj:pass")}
        out = []
        for i, t in enumerate(sent):
            f = t.get("feats") or {}
            up = t["upos"]
            g, num = self._gset(f.get("Gender")), f.get("Number")
            if up == "PRON" and f.get("PronType") == "Prs":
                if f.get("Person") == "3":
                    out.append({"i": i, "cat": "pron3", "target": "antecedent", "g": g, "num": num})
                elif f.get("Person") in ("1", "2"):
                    out.append({"i": i, "cat": "person12", "target": "mluvčí/adresát", "g": g, "num": num})
            elif f.get("Poss") == "Yes":
                if f.get("Reflex") == "Yes" or t["lemma"] == "svůj":
                    out.append({"i": i, "cat": "poss_reflex", "target": "podmět klauzule", "g": g, "num": num})
                elif f.get("Person") in ("1", "2"):
                    out.append({"i": i, "cat": "poss12", "target": "mluvčí/adresát", "g": g, "num": num})
                else:
                    out.append({"i": i, "cat": "poss3", "target": "antecedent", "g": g, "num": num})
            elif up in ("PRON", "DET") and f.get("PronType") == "Dem" \
                    and t["deprel"] in ("nsubj", "obj", "root"):
                out.append({"i": i, "cat": "dem", "target": "antecedent", "g": g, "num": num})
            elif up == "VERB" and f.get("Tense") == "Past" \
                    and (i + 1) not in subj_heads and t["deprel"] == "root":
                # FIX 1: neosobní reflexivum („A stalo se tak") NENÍ díra — Neut Sing + „se"
                refl = any(t2.get("head") == i + 1 and t2["lemma"] == "se" for t2 in sent)
                if refl and f.get("Gender") == "Neut" and f.get("Number") == "Sing":
                    continue
                out.append({"i": i, "cat": "prodrop", "target": "antecedent", "g": g, "num": num})
        return out

    def _agree(self, cg, cnum, pg, pnum):
        """Shoda kandidáta s dírou: překryv rodu (prázdné = permisivní) + číslo."""
        g_ok = not (pg and cg) or bool(pg & cg)
        n_ok = pnum is None or cnum is None or cnum == pnum
        return g_ok and n_ok

    def resolve_document(self, sentences):
        """Projde věty dokumentu v pořadí a vrátí resoluce děr typu antecedent.

        Backward cache drží zmínky (kandidáty) z okna N vět; pro každou díru vybere
        nejlepší souhlasnou zmínku (podmět > recency > pozice) = těžiště. Vyřešená
        díra se vrací do cache (řetěz). Vrací list resolucí (sent_k, i, cat, lemma|None).

        Vstup: `sentences` = věty dokumentu V POŘADÍ. Výstup: list resolucí.
        """
        cache = []
        results = []
        for k, sent in enumerate(sentences):
            cache = [c for c in cache if c["k"] >= k - self.window]
            ments = []
            for j, t in enumerate(sent):
                if t["upos"] in self.cand_upos:
                    f = t.get("feats") or {}
                    ments.append({"lemma": self.g.canon_lemma(t), "g": self._gset(f.get("Gender")),
                                  "num": f.get("Number"), "k": k,
                                  "subj": t["deprel"] in ("nsubj", "nsubj:pass"), "pos": j})
            for ph in self.detect(sent):
                if ph["target"] != "antecedent":
                    results.append((k, ph["i"], ph["cat"], None))
                    continue
                cands = cache + [m for m in ments if m["pos"] < ph["i"]]
                best = None
                for m in cands:
                    if self._agree(m["g"], m["num"], ph["g"], ph["num"]):
                        score = (1 if m["subj"] else 0, m["k"], m["pos"])
                        if best is None or score > best[0]:
                            best = (score, m)
                if best:
                    results.append((k, ph["i"], ph["cat"], best[1]["lemma"]))
                    cache.append({"lemma": best[1]["lemma"], "g": best[1]["g"],
                                  "num": best[1]["num"], "k": k, "subj": False, "pos": ph["i"]})
                else:
                    results.append((k, ph["i"], ph["cat"], None))
            cache.extend(ments)
        return results

    def resolve_document_activation(self, sentences):
        """Jako `resolve_document`, ale kandidáty váží AKUMULOVANOU AKTIVACÍ.

        Aktivační pole nad entitami: každá zmínka přičte aktivaci (podmět víc,
        `subject_weight`), na začátku věty se vše utlumí (`decay`). Díra se zaplní
        NEJAKTIVNĚJŠÍ souhlasnou entitou (téma diskurzu, ne jen nejbližší podmět);
        vyřešená díra referenta posílí (`reinforce`) → téma drží přes řetěz zájmen.

        Vstup: věty dokumentu v pořadí. Výstup: list resolucí (sent_k, i, cat, lemma|None).
        """
        field = {}
        results = []
        for k, sent in enumerate(sentences):
            for e in field.values():
                e["act"] *= self.decay
            subj_heads = {t["head"] for t in sent if t["deprel"] in ("nsubj", "nsubj:pass")}
            phs = {d["i"]: d for d in self.detect(sent)}
            for j, t in enumerate(sent):
                if j in phs:
                    ph = phs[j]
                    if ph["target"] != "antecedent":
                        results.append((k, j, ph["cat"], None))
                    elif ph["num"] == "Plur":
                        # FIX 2: nejdřív shodná PLURÁLOVÁ entita (mudrci); jinak plurál =
                        # kolekce → 2 nejteplejší singuláry (Zachariáš+Alžběta, „mezi nimi")
                        plur = [(l, e["act"]) for l, e in field.items()
                                if e["num"] == "Plur"
                                and self._agree(e["g"], e["num"], ph["g"], ph["num"])]
                        if plur:
                            resolved = max(plur, key=lambda x: x[1])[0]
                        else:
                            top = sorted(field.items(), key=lambda kv: -kv[1]["act"])[:2]
                            resolved = "+".join(l for l, _ in top) if top else None
                        results.append((k, j, ph["cat"], resolved))
                    else:
                        best = None
                        for lemma, e in field.items():
                            if self._agree(e["g"], e["num"], ph["g"], ph["num"]) \
                                    and (best is None or e["act"] > best[1]):
                                best = (lemma, e["act"])
                        resolved = best[0] if best else None
                        results.append((k, j, ph["cat"], resolved))
                        if resolved:
                            field[resolved]["act"] += self.reinforce
                if t["upos"] in self.cand_upos:
                    lemma = self.g.canon_lemma(t)
                    f = t.get("feats") or {}
                    w = self.subj_weight if t["deprel"] in ("nsubj", "nsubj:pass") else 1.0
                    e = field.setdefault(lemma, {"act": 0.0, "g": self._gset(f.get("Gender")),
                                                 "num": f.get("Number")})
                    e["act"] += w
                    if f.get("Gender"):
                        e["g"] = self._gset(f.get("Gender"))
                    if f.get("Number"):
                        e["num"] = f.get("Number")
        return results

    def build_zone(self, corpus=None):
        """Postaví ZÓNU per entita = rolový profil (lemma → Counter(pád→počet)) z korpusu.

        Zóna je GLOBÁLNÍ (přes celý korpus): říká, v jakých rolích entita běžně žije
        (`slovo` je Acc-předmět 46 %, `svět` genitiv/místo). Bez argumentu načte celý
        korpus přes `Dataloader`. Uloží do `self.zone` a vrátí ji.
        """
        from collections import Counter
        if corpus is None:
            from dataloader import Dataloader
            dl = Dataloader()
            corpus = dl.mount(dl.document_ids())
        zone = {}
        for rec in corpus.values():
            for s in rec["sentences"]:
                for t in s:
                    if t["upos"] in self.cand_upos:
                        c = (t.get("feats") or {}).get("Case")
                        if c:
                            zone.setdefault(self.g.canon_lemma(t), Counter())[c] += 1
        self.zone = zone
        return zone

    def _fit(self, lemma, case):
        """Transpoziční fit: jak přirozeně entita `lemma` žije v roli `case` (0..1).

        Podíl výskytů entity v daném pádu z její zóny; prázdná zóna / neznámý pád →
        1.0 (neutrální, žádný signál). To je ta mezivrstva — dosadí kandidáta do role díry
        a změří, zda tam patří (`slovo` do Acc sedí, `svět` do Acc ne).
        """
        z = getattr(self, "zone", {}).get(lemma)
        if not z or not case:
            return 1.0
        return z.get(case, 0) / sum(z.values())

    def resolve_document_zone(self, sentences):
        """Aktivace × transpoziční fit zóny: rank = aktivace · (1 + zone_beta · fit).

        Jako `resolve_document_activation`, ale mezi souhlasnými kandidáty rozhoduje i to,
        zda entita FITUJE do ROLE díry (pád) podle své globální zóny. Aktivace drží
        recency/topik, zóna drží „sedí do role" (`ho` Acc → `slovo`, ne `svět`).
        Vyžaduje `build_zone` (zavolá se líně).
        """
        if not hasattr(self, "zone"):
            self.build_zone()
        field = {}
        results = []
        for k, sent in enumerate(sentences):
            for e in field.values():
                e["act"] *= self.decay
            subj_heads = {t["head"] for t in sent if t["deprel"] in ("nsubj", "nsubj:pass")}
            phs = {d["i"]: d for d in self.detect(sent)}
            for j, t in enumerate(sent):
                if j in phs:
                    ph = phs[j]
                    if ph["target"] != "antecedent":
                        results.append((k, j, ph["cat"], None, 0.0))
                    else:
                        ph_case = "Nom" if ph["cat"] == "prodrop" else (
                            None if ph["cat"] == "poss3" else (t.get("feats") or {}).get("Case"))
                        scored = []
                        for lemma, e in field.items():
                            if not self._agree(e["g"], e["num"], ph["g"], ph["num"]):
                                continue
                            scored.append((e["act"] * (1 + self.zone_beta * self._fit(lemma, ph_case)),
                                           lemma))
                        scored.sort(reverse=True)
                        if scored:
                            resolved = scored[0][1]
                            top1 = scored[0][0]
                            top2 = scored[1][0] if len(scored) > 1 else 0.0
                            # JISTOTA = relativní odstup vítěze od druhého (0..1)
                            conf = (top1 - top2) / top1 if top1 > 0 else 0.0
                        else:
                            resolved, conf = None, 0.0
                        results.append((k, j, ph["cat"], resolved, conf))
                        if resolved:
                            field[resolved]["act"] += self.reinforce
                if t["upos"] in self.cand_upos:
                    lemma = self.g.canon_lemma(t)
                    f = t.get("feats") or {}
                    w = self.subj_weight if t["deprel"] in ("nsubj", "nsubj:pass") else 1.0
                    e = field.setdefault(lemma, {"act": 0.0, "g": self._gset(f.get("Gender")),
                                                 "num": f.get("Number")})
                    e["act"] += w
                    if f.get("Gender"):
                        e["g"] = self._gset(f.get("Gender"))
                    if f.get("Number"):
                        e["num"] = f.get("Number")
        return results

    def resolve_document_identity(self, sentences):
        """Zone + KOTVA IDENTITY: entita zavedená copulou „X je/byl Y" se stane PROTAGONISTOU
        a drží teplo (`anchor_bonus`) přes pasáž navzdory útlumu.

        „Slovo byl Bůh" / „bylo Slovo" zavádí `Slovo` jako téma; anchor pak brání, aby ho
        lokálně salientní `svět` přebil, když je referent studený. Jinak jako
        `resolve_document_zone`. Vrací (k, i, cat, lemma, conf).
        """
        if not hasattr(self, "zone"):
            self.build_zone()
        from collections import Counter
        copula = self.g.LANG["copula_lemma"]
        field = {}
        anchors = Counter()                          # lemma → síla (kolikrát copula-subjekt)
        results = []
        for k, sent in enumerate(sentences):
            for e in field.values():
                e["act"] *= self.decay
            for a, strength in anchors.items():       # protagonisté drží teplo ÚMĚRNĚ SÍLE
                if a in field:
                    field[a]["act"] += self.anchor_bonus * strength
            cop_heads = {t["head"] for t in sent if t["deprel"] == "cop"}
            byid = {i + 1: t for i, t in enumerate(sent)}
            phs = {d["i"]: d for d in self.detect(sent)}
            for j, t in enumerate(sent):
                if j in phs:
                    ph = phs[j]
                    if ph["target"] != "antecedent":
                        results.append((k, j, ph["cat"], None, 0.0))
                    else:
                        ph_case = "Nom" if ph["cat"] == "prodrop" else (
                            None if ph["cat"] == "poss3" else (t.get("feats") or {}).get("Case"))
                        scored = []
                        for lemma, e in field.items():
                            if not self._agree(e["g"], e["num"], ph["g"], ph["num"]):
                                continue
                            scored.append((e["act"] * (1 + self.zone_beta * self._fit(lemma, ph_case)),
                                           lemma))
                        scored.sort(reverse=True)
                        if scored:
                            resolved = scored[0][1]
                            top1, top2 = scored[0][0], (scored[1][0] if len(scored) > 1 else 0.0)
                            conf = (top1 - top2) / top1 if top1 > 0 else 0.0
                        else:
                            resolved, conf = None, 0.0
                        results.append((k, j, ph["cat"], resolved, conf))
                        if resolved:
                            field[resolved]["act"] += self.reinforce
                if t["upos"] in self.cand_upos:
                    lemma = self.g.canon_lemma(t)
                    f = t.get("feats") or {}
                    w = self.subj_weight if t["deprel"] in ("nsubj", "nsubj:pass") else 1.0
                    e = field.setdefault(lemma, {"act": 0.0, "g": self._gset(f.get("Gender")),
                                                 "num": f.get("Number")})
                    e["act"] += w
                    if f.get("Gender"):
                        e["g"] = self._gset(f.get("Gender"))
                    if f.get("Number"):
                        e["num"] = f.get("Number")
                    # identity-subjekt (nsubj klauzule s copulou / slovesem „být") → anchor
                    if t["deprel"] in ("nsubj", "nsubj:pass"):
                        head = byid.get(t["head"])
                        if t["head"] in cop_heads or (head and head["lemma"] == copula):
                            anchors[lemma] += 1
        return results


    def resolved_subjects(self, sentences):
        """{(k, predikát-lemma): (podmět-lemma, conf)} pro PRO-DROP podměty (identity resolver).

        Klíč join na fakty (④ registry `hole=subj/person`): pro každou prodrop díru v přísudku
        `root` vrátí resolvovaný podmět. Tím se do faktu doplní chybějící who-slot — otázka
        „Kde se narodil ČAPEK" pak najde fakt „Narodil se…" přes sdílenou entitu (viz
        docs/koreference-do-faktu.html). Vstup: věty dokumentu v pořadí. Výstup: mapa.
        """
        out = {}
        for (k, i, cat, lemma, conf) in self.resolve_document_identity(sentences):
            if cat == "prodrop" and lemma:
                out[(k, self.g.canon_lemma(sentences[k][i]))] = (lemma, conf)
        return out


def _measure():
    """Porovná baseline (nejbližší podmět) vs activation (aktivační pole) — fill rate
    na novém korpusu + spot-check Janova prologu vedle sebe."""
    import pickle
    fh = FillHoles()
    docs = ["bible_jan", "bible_lukas", "bible_matous", "bible_genesis", "wiki_karel_čapek"]
    ANT = ("pron3", "poss3", "dem", "prodrop")
    strategies = (("baseline", fh.resolve_document),
                  ("activation", fh.resolve_document_activation))

    def load(doc):
        shard = pickle.load(open(os.path.join(HERE, f"../../data/corpus/{doc}.pkl"), "rb"))
        return [s for (_d, _i), rec in sorted(shard.items(), key=lambda x: x[0][1])
                for s in rec["sentences"]]

    print("=== fill rate antecedent-typ (5 dokumentů) ===")
    for name, fn in strategies:
        tot = fil = 0
        for doc in docs:
            for (_k, _i, c, lemma) in fn(load(doc)):
                if c in ANT:
                    tot += 1
                    fil += bool(lemma)
        print(f"  {name:11} {fil}/{tot} ({100*fil/max(tot,1):.0f}%)")

    print("\n=== spot-check Janův prolog (baseline → activation) ===")
    sents = load("bible_jan")[:12]
    rb = {(k, i): lemma for (k, i, c, lemma) in fh.resolve_document(sents)}
    ra = {(k, i): lemma for (k, i, c, lemma) in fh.resolve_document_activation(sents)}
    dets = {(k, d["i"]): d["cat"] for k, s in enumerate(sents) for d in fh.detect(s)}
    for k, s in enumerate(sents):
        for i, t in enumerate(s):
            if (k, i) in dets and dets[(k, i)] in ANT:
                b, a = rb.get((k, i)) or "—", ra.get((k, i)) or "—"
                mark = "" if b == a else "   ← ZMĚNA"
                print(f"  „{t['form']}\" [{dets[(k,i)]:7}]  {b:10} → {a}{mark}")


if __name__ == "__main__":
    _measure()
