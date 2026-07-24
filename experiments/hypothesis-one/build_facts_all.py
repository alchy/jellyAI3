#!/usr/bin/env python3
"""Postaví FAKT-store pro celý korpus z registry.jsonl (predikát + role-sloty, sharded per doc).

Agreguje answer-sloty registry (④ synth_registry) per (doc, sent, predikát) → jeden `Fact`
s `roles = {role: [lemmata]}`. Navíc OBOHACUJE fakty o KOREFERENCÍ resolvovaný podmět: má-li
fakt pro-drop díru podmětu (`hole=subj/person`) a `fill_holes` ji vyplní, přidá se
`who = [podmět]` — tím se odemkne predikátová cesta (viz docs/koreference-do-faktu.html).
Bez sítě, deterministické.

Vstup: registry.jsonl + korpus (pro koreferenci). Výstup: data/facts/<doc>.jsonl + počty.
"""
import os
import json
import pickle

from fact_store import FactStore, Fact
from fill_holes import FillHoles
from extract_bio import bio_facts
from extract_relations import relation_facts
from chronos import Chronos
from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "registry.jsonl")
CORP = os.path.join(HERE, "../../data/corpus")


def _load_sentences(doc):
    """Věty dokumentu V POŘADÍ (stejná indexace jako registry `sent`)."""
    shard = pickle.load(open(os.path.join(CORP, f"{doc}.pkl"), "rb"))
    return [s for (_d, _i), rec in sorted(shard.items(), key=lambda x: x[0][1])
            for s in rec["sentences"]]


def main():
    """Přestaví fakt-store z registry + obohatí pro-drop podměty koreferencí."""
    fh = FillHoles()
    # 1) agregace registry per (doc, sent, predikát): role-sloty + zda je podmět díra
    agg = {}
    lines = 0
    for line in open(REG, encoding="utf-8"):
        e = json.loads(line)
        lines += 1
        key = (e["doc"], e["sent"], e["predicate"])
        entry = agg.setdefault(key, {"roles": {}, "subj_hole": False})
        if "subj" in (e.get("hole") or ""):
            entry["subj_hole"] = True
        for a in e["answers"]:
            role, lemma = a.get("role"), a.get("lemma")
            if not role or not lemma:
                continue
            bucket = entry["roles"].setdefault(role, [])
            if lemma not in bucket:
                bucket.append(lemma)
    # 2) per-dokument: koreference (pro-drop podměty) + bio závorka + ČAS (Chronos)
    ch = Chronos()
    subjects = {}                                   # (doc, sent, predikát) -> (podmět, conf)
    bio = []                                        # [Fact] z biografických závorek
    rels = []                                       # [Fact] vztahů (bratr/manžel/otec…) jako hrany
    temporal = []                                   # (doc, sent, predikát, rok) z běžného textu
    from collections import Counter
    for doc in sorted({d for (d, _s, _p) in agg}):
        try:
            sents = _load_sentences(doc)
        except FileNotFoundError:
            continue
        for (k, verb_lemma), (lemma, conf) in fh.resolved_subjects(sents).items():
            subjects[(doc, k, verb_lemma)] = (lemma, conf)
        if sents:                                   # JEN úvodní věta = definice narození/úmrtí
            for pred, roles in bio_facts(sents[0], fh.g):
                bio.append(Fact(pred, roles, doc, 0))
        prot = Counter()                            # protagonista = nejčastější podmět-osoba (holder)
        for s in sents:
            for t in s:
                fe = t.get("feats") or {}
                if t.get("deprel") in ("nsubj", "nsubj:pass") and t["upos"] == "PROPN" \
                        and fe.get("NameType") in ("Giv", "Sur"):
                    prot[fh.g.canon_lemma(t)] += 1
        holder = prot.most_common(1)[0][0] if prot else None
        for si, s in enumerate(sents):              # Chronos: rok → when; + vztahové hrany
            for pred, year in ch.temporal_facts(s, fh.g):
                temporal.append((doc, si, pred, year))
            for pred, roles in relation_facts(s, fh.g, holder):
                rels.append(Fact(pred, roles, doc, si))
    for (doc, si, pred, year) in temporal:          # doplň when do existujícího faktu (nebo založ)
        e = agg.setdefault((doc, si, pred), {"roles": {}, "subj_hole": False})
        b = e["roles"].setdefault("when", [])
        if year not in b:
            b.append(year)
    # 3) obohať fakty o who-slot a postav store
    store = FactStore()
    store.reset()
    docs = set()
    enriched = 0
    for (doc, sent, predicate), entry in agg.items():
        roles = entry["roles"]
        if entry["subj_hole"] and "who" not in roles:          # doplň chybějící podmět
            sub = subjects.get((doc, sent, predicate))
            if sub:
                roles = dict(roles)
                roles["who"] = [sub[0]]
                enriched += 1
        if not roles:
            continue
        store.append(Fact(predicate, roles, doc, sent))
        docs.add(doc)
    for f in bio + rels:                            # biografické závorky + vztahové hrany
        store.append(f)
        docs.add(f.doc)
    logger("i", f"fakty: {len(agg)}+{len(bio)} bio +{len(rels)} vztahy nad {len(docs)} soubory "
                f"({lines} bindings); koreferencí doplněno who: {enriched}")
    print(f"faktů: {len(agg)}  bio: {len(bio)}  vztahy: {len(rels)}  soubory: {len(docs)}  "
          f"koref-who: {enriched}")


if __name__ == "__main__":
    main()
