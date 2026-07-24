#!/usr/bin/env python3
"""Extrakce VZTAHŮ jako fakt/hrana z KOPULOVÉ konstrukce „podmět BYL <vztah-slovo> <osoba>".

Stejnovětný lever (měřeno: 87 % odpovědí je ve stejné větě, jen nevytěžené). Běžná extrakce
z „Byl mladším bratrem Josefa Čapka" vytěží jen `state=bratr` a ztratí OBĚ osoby. Tady se
vztah stane FAKTEM (hranou mezi entitami): predikát = vztah-slovo, role who + whose_of_what.
Symetrické vztahy (bratr/manžel) obousměrně → dotaz „bratr Karla" i „bratr Josefa" fungují.

Vztahová slova + symetrie = jazyková data (`grammar.LANG`). Vstup: věta, grammar, holder
(protagonista dokumentu / coref podmět pro pro-drop). Výstup: [(vztah_predikát, roles)].
"""


def relation_facts(sent, grammar, holder=None):
    """Věta → vztahové fakty. Holder = explicitní nsubj-osoba, jinak `holder` (protagonista)."""
    reln = set(grammar.LANG.get("relation_nouns", []))
    symm = set(grammar.LANG.get("symmetric_relations", []))
    if not reln or not any(t.get("deprel") == "cop" for t in sent):
        return []
    h = holder
    for t in sent:                                             # explicitní podmět-osoba přebije holder
        fe = t.get("feats") or {}
        if t.get("deprel") == "nsubj" and t["upos"] == "PROPN" and fe.get("NameType") in ("Giv", "Sur"):
            h = grammar.canon_lemma(t)
            break
    if not h:
        return []
    out = []
    for i, t in enumerate(sent):
        rel = grammar.canon_lemma(t).lower()
        if t["upos"] == "NOUN" and rel in reln and t.get("deprel") in ("root", "conj", "appos"):
            target = None                                      # nejbližší PROPN-osoba PO vztah-slově
            for u in sent[i + 1:]:
                fe = u.get("feats") or {}
                if u["upos"] == "PROPN" and fe.get("NameType") in ("Giv", "Sur"):
                    cl = grammar.canon_lemma(u)
                    if cl.lower() != h.lower():
                        target = cl
                        break
            if not target:
                continue
            out.append((rel, {"who": [h], "whose_of_what": [target]}))
            if rel in symm:                                    # symetrický → i opačný směr
                out.append((rel, {"who": [target], "whose_of_what": [h]}))
    return out
