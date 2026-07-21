#!/usr/bin/env python3
"""FÁZE 1 — třídní API. Každá třída řeší JEDEN krok: jasný vstup → jasný výstup,
žádné monolity. Kompozici dělá `Phase1`. Logika je v run.py/roles.py; tady je
čistá, dokumentovatelná, testovatelná fasáda po třídách.

KONFIG (nic natvrdo v kódu):
  config.json  — radius, modality_marks, punct_keep, služby (UDPipe url), cesty
  lang/cs.json — role_catalog, role_ask, deprel_to_role, role_prepositions,
                 deprel_structural, structural_roles, clause_markers …
  curated.json — kurátorské opravy VZOR → role

Testy: test_phase1.py (co kontrolují viz tam).
"""
import json, sys
sys.path.insert(0, "/Users/j/Projects/jellyAI3/experiments/hypothesis-one")
import roles as _roles
_run = _roles._run

_HERE = "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/"
CONFIG = json.load(open(_HERE + "config.json", encoding="utf-8"))


class Annotator:
    """Krok 1a — text věty → SEKTORY (anotace UDPipe).

    Vstup:  text (str) — jedna věta z blobu.
    Výstup: list[Token] — sektory (dict s form/lemma/upos/feats/head/deprel).
    Konfig: config.json['services']['udpipe_url'].
    Příklad:
        Annotator().annotate("Odešel do galilejského města Kafarnaum a učil je v sobotu.")[4]
        → {'form': 'Kafarnaum', 'upos': 'PROPN', 'feats': {'Case': 'Nom', …}, 'deprel': 'nsubj'}
    """
    def __init__(self, config=CONFIG):
        self.url = config["services"]["udpipe_url"]

    def annotate(self, text):
        return _roles.udpipe(text)[0]     # první věta (list tokenů)


class Standardizer:
    """Krok 1b — SEKTOR → role (náš klíč); deprel se ven nedostane.

    Vstup:  token (dict), sent (list[Token]).
    Výstup: role (str | None) — klíč katalogu (who/where/action/preposition…).
    Konfig: lang/cs.json (deprel_to_role, role_prepositions, deprel_structural).
    Příklad:
        Standardizer().standardize(sent)
        → [('Odešel','action'), ('města','where'), ('Kafarnaum','who'), …]
    """
    def role(self, token, sent):
        byid = _roles._byid(sent)
        cop = {x["head"] for x in sent if x["deprel"] == "cop"}
        return _roles.standard_role(token, sent, byid, cop)

    def standardize(self, sent):
        return _roles.standardize(sent)


class VzorBuilder:
    """Krok 1c — sektor + okno → VZOR (přesná gramatická šablona).

    Vstup:  sent (list[Token]), i (int index pivotu), r (int poloměr, jinak z konfigu).
    Výstup: VZOR (str) — 'slot·…·PIVOT·…·slot·modalita'; pivot nese pád = roli.
    Konfig: config.json['radius'] (výchozí r).
    Příklad:
        VzorBuilder().vzor(sent, 4, r=2)   # pivot = Kafarnaum
        → 'ADJ:Gen·NOUN:Gen·PROPN:Nom·CCONJ·VERB:Past·.'
    """
    def __init__(self, config=CONFIG):
        self.r = config["radius"]

    def vzor(self, sent, i, r=None):
        mod = _run.sentence_modality(sent)
        return _run.frame_sig(sent, i, mod, self.r if r is None else r)


class Curator:
    """Krok 1d — VZOR → kurátorská oprava (přebíjí chyby ÚFALu).

    Vstup:  sent (list[Token]), r (int poloměr VZORu).
    Výstup: list[(Token, role: str, kurátorováno: bool)].
    Konfig: curated.json (VZOR → {role, note}).
    Příklad:
        Curator().standardize(sent, r=2)
        → [('Odešel','action',False), …, ('jim','to_whom',True), …]
    """
    def __init__(self, config=CONFIG):
        path = _HERE + config["paths"]["curated"]
        self.db = {k: v["role"] for k, v in
                   json.load(open(path, encoding="utf-8")).items()
                   if not k.startswith("_")}

    def standardize(self, sent, r=2):
        byid = _roles._byid(sent)
        mod = _run.sentence_modality(sent)
        cop = {x["head"] for x in sent if x["deprel"] == "cop"}
        out = []
        for i, t in enumerate(sent):
            if t["upos"] == "PUNCT":
                out.append((t, _roles.standard_role(t, sent, byid, cop), False)); continue
            vz = _run.frame_sig(sent, i, mod, r)
            if vz in self.db:
                out.append((t, self.db[vz], True))
            else:
                out.append((t, _roles.standard_role(t, sent, byid, cop), False))
        return out


class Phase1:
    """Kompozice 1a–1d.

    Vstup:  text (str) — věta.
    Výstup: list[(Token, role: str, kurátorováno: bool)] — vstup pro fázi 2.
    Konfig: config.json (skládá podřízené třídy).
    Příklad:
        Phase1().run("Odešel do galilejského města Kafarnaum a učil je v sobotu.")
    """
    def __init__(self, config=CONFIG):
        self.annotator = Annotator(config)
        self.standardizer = Standardizer()
        self.vzor = VzorBuilder(config)
        self.curator = Curator(config)

    def run(self, text, r=2):
        sent = self.annotator.annotate(text)
        return self.curator.standardize(sent, r)


if __name__ == "__main__":
    for tok, role, cur in Phase1().run(
            "Odešel do galilejského města Kafarnaum a učil je v sobotu."):
        print(f"  {tok['form']:13} {role:16} {'[KURÁTOR]' if cur else ''}")
