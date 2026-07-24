#!/usr/bin/env python3
"""FÁZE 1 — třídní API. Každá třída řeší JEDEN krok: jasný vstup → jasný výstup,
žádné monolity. Kompozici dělá `Phase1`. Logika je v run.py/roles.py; tady je
čistá, dokumentovatelná, testovatelná fasáda po třídách.

SLOVNÍČEK ATOMŮ (odráží se v typových aliasech níže; „X_ARRAY = pole X"):
  WORD_PLAIN         str          holé slovo (bez atributů)
  WORD_PLAIN_ARRAY   list[str]    pole holých slov (syrová tokenizace)
  WORD_W_ATTR        dict         slovo + atributy (lemma/upos/feats/head/deprel)
  WORD_W_ATTR_ARRAY  list[dict]   věta / okno  ← workhorse (v kódu proměnná `sent`)
  SLOT               str          match-obálka slova = run.slot()
  SLOT_ARRAY         str          VZOR = run.frame_sig()  (≡ VZOR, „mluvnický vzor pán")
  MOST (abstrakce):  WORD_W_ATTR → SLOT ·  WORD_W_ATTR_ARRAY → SLOT_ARRAY

KONFIG (nic natvrdo v kódu):
  config.json  — radius, modality_marks, punct_keep, služby (UDPipe url), cesty
  lang/cs.json — role_catalog, role_ask, deprel_to_role, role_prepositions,
                 deprel_structural, structural_roles, clause_markers …
  curated.json — kurátorské opravy SLOT_ARRAY → role

Testy: test_phase1.py (co kontrolují viz tam).
"""
import json, sys
sys.path.insert(0, "/Users/j/Projects/jellyAI3/experiments/hypothesis-one")
import roles as _roles
_run = _roles._run

_HERE = "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/"
CONFIG = json.load(open(_HERE + "config.json", encoding="utf-8"))

# ── slovníček atomů jako typové aliasy (odráží glossary v kódu) ──────────────
WORD_PLAIN = str                 # holé slovo
WORD_PLAIN_ARRAY = list          # list[WORD_PLAIN]
WORD_W_ATTR = dict               # slovo + atributy
WORD_W_ATTR_ARRAY = list         # list[WORD_W_ATTR] = věta / okno
SLOT = str                       # match-obálka = slot()
SLOT_ARRAY = str                 # VZOR = frame_sig()


class Annotator:
    """Krok 1a — text věty → WORD_W_ATTR_ARRAY (anotace UDPipe).

    Vstup:  text (str) — jedna věta z blobu.
    Výstup: WORD_W_ATTR_ARRAY — pole WORD_W_ATTR (dict s form/lemma/upos/feats/head/deprel).
    Konfig: config.json['services']['udpipe_url'].
    Příklad:
        Annotator().annotate("Odešel do galilejského města Kafarnaum a učil je v sobotu.")[4]
        → {'form': 'Kafarnaum', 'upos': 'PROPN', 'feats': {'Case': 'Nom', …}, 'deprel': 'nsubj'}
    """
    def __init__(self, config=CONFIG):
        self.url = config["services"]["udpipe_url"]

    def annotate(self, text: str) -> WORD_W_ATTR_ARRAY:
        return _roles.udpipe(text)[0]     # první věta


class Standardizer:
    """Krok 1b — WORD_W_ATTR → role (náš klíč); deprel se ven nedostane.

    Vstup:  word_w_attr (WORD_W_ATTR), sent (WORD_W_ATTR_ARRAY).
    Výstup: role (str | None) — klíč katalogu (who/where/action/preposition…).
    Konfig: lang/cs.json (deprel_to_role, role_prepositions, deprel_structural).
    Příklad:
        Standardizer().standardize(sent)
        → [('Odešel','action'), ('města','where'), ('Kafarnaum','who'), …]
    """
    def role(self, word_w_attr: WORD_W_ATTR, sent: WORD_W_ATTR_ARRAY):
        byid = _roles._byid(sent)
        cop = {x["head"] for x in sent if x["deprel"] == "cop"}
        return _roles.standard_role(word_w_attr, sent, byid, cop)

    def standardize(self, sent: WORD_W_ATTR_ARRAY):
        return _roles.standardize(sent)


class VzorBuilder:
    """Krok 1c — WORD_W_ATTR_ARRAY + pivot → SLOT_ARRAY (VZOR, přesná šablona).

    Vstup:  sent (WORD_W_ATTR_ARRAY), i (int index pivotu), r (int poloměr; jinak z konfigu).
    Výstup: SLOT_ARRAY (str) — 'slot·…·PIVOT·…·slot·modalita'; pivot nese pád = roli.
    Konfig: config.json['radius'].
    Příklad:
        VzorBuilder().vzor(sent, 4, r=2)   # pivot = Kafarnaum
        → 'ADJ:Gen·NOUN:Gen·PROPN:Nom·CCONJ·VERB:Past·.'
    """
    def __init__(self, config=CONFIG):
        self.r = config["radius"]

    def vzor(self, sent: WORD_W_ATTR_ARRAY, i: int, r=None) -> SLOT_ARRAY:
        mod = _run.sentence_modality(sent)
        return _run.frame_sig(sent, i, mod, self.r if r is None else r)


class Curator:
    """Krok 1d — SLOT_ARRAY → kurátorská oprava (přebíjí chyby ÚFALu).

    Vstup:  sent (WORD_W_ATTR_ARRAY), r (int poloměr VZORu).
    Výstup: list[(WORD_W_ATTR, role: str, kurátorováno: bool)].
    Konfig: curated.json (SLOT_ARRAY → {role, note}).
    Příklad:
        Curator().standardize(sent, r=2)
        → [('Odešel','action',False), …, ('jim','to_whom',True), …]
    """
    def __init__(self, config=CONFIG):
        path = _HERE + config["paths"]["curated"]
        self.db = {k: v["role"] for k, v in
                   json.load(open(path, encoding="utf-8")).items()
                   if not k.startswith("_")}

    def standardize(self, sent: WORD_W_ATTR_ARRAY, r=2):
        byid = _roles._byid(sent)
        mod = _run.sentence_modality(sent)
        cop = {x["head"] for x in sent if x["deprel"] == "cop"}
        out = []
        for i, w in enumerate(sent):
            if w["upos"] == "PUNCT":
                out.append((w, _roles.standard_role(w, sent, byid, cop), False)); continue
            slot_array = _run.frame_sig(sent, i, mod, r)    # SLOT_ARRAY (VZOR)
            if slot_array in self.db:
                out.append((w, self.db[slot_array], True))
            else:
                out.append((w, _roles.standard_role(w, sent, byid, cop), False))
        return out


# ── pomůcky pro persistentní slovník ────────────────────────────────────────
def _resolve(path):
    """Absolutní cesta beze změny; relativní vůči adresáři experimentu."""
    return path if path.startswith("/") else _HERE + path

def _load_role_map(path):
    """Načte {VZOR: {"role": …}} → {VZOR: role}. Chybějící soubor = prázdno."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {k: v["role"] for k, v in data.items()
            if not k.startswith("_") and isinstance(v, dict) and "role" in v}


class PersistentDeterminer:
    """Krok 1d (persistentní) — SLOT_ARRAY → role přes STATICKÝ SLOVNÍK, vrstveno:
        CURATED (ruční) → CONFIRMED (ověřený slovník) → CANDIDATE (živý návrh k revizi).

    NAČÍTÁ vždy (curated + determination). UKLÁDÁ jen explicitně (save_candidates /
    promote) — `determine()` na disk NEŠAHÁ (determinismus + revizní brána).

    determine():        vstup sent (WORD_W_ATTR_ARRAY) → [(WORD_W_ATTR, role, provenance)].
    build(sents):       offline dávka — naplní self.pending (bez zápisu).
    save_candidates():  explicitně zapíše self.pending do candidates.json (k revizi).
    promote(vzory):     po revizi povýší VZORy na CONFIRMED (zapíše determination.json).
    Konfig: config.json['paths'] (curated/determination/candidates), ['radius'].
    """
    CURATED, CONFIRMED, CANDIDATE = "CURATED", "CONFIRMED", "CANDIDATE"

    def __init__(self, config=CONFIG):
        self.r = config.get("radius", 2)
        p = config["paths"]
        self._curated_path = _resolve(p["curated"])
        self._det_path = _resolve(p["determination"])
        self._cand_path = _resolve(p["candidates"])
        self.curated = _load_role_map(self._curated_path)     # VZOR → role (ruční)
        self.confirmed = _load_role_map(self._det_path)       # VZOR → role (ověřené)
        self.pending = {}                                     # VZOR → role (CANDIDATE, PAMĚŤ)

    def determine(self, sent, r=2):
        """Lookup CURATED → CONFIRMED → miss → standard_role (CANDIDATE do self.pending).
        BEZ zápisu na disk. Vrací trojice (WORD_W_ATTR, role, provenance)."""
        byid = _roles._byid(sent)
        mod = _run.sentence_modality(sent)
        cop = {x["head"] for x in sent if x["deprel"] == "cop"}
        out = []
        for i, w in enumerate(sent):
            if w["upos"] == "PUNCT":                          # strukturní, triviálně jisté
                out.append((w, _roles.standard_role(w, sent, byid, cop), self.CONFIRMED)); continue
            vz = _run.frame_sig(sent, i, mod, r)              # SLOT_ARRAY (VZOR)
            if vz in self.curated:
                out.append((w, self.curated[vz], self.CURATED))
            elif vz in self.confirmed:
                out.append((w, self.confirmed[vz], self.CONFIRMED))
            else:
                role = _roles.standard_role(w, sent, byid, cop)   # živý fallback = návrh
                self.pending[vz] = role                           # poznamenej (jen v PAMĚTI!)
                out.append((w, role, self.CANDIDATE))
        return out

    def build(self, sents, r=2):
        """Offline dávka nad WORD_W_ATTR_ARRAY-y: naplní self.pending (bez zápisu)."""
        for sent in sents:
            self.determine(sent, r)
        return len(self.pending)

    def save_candidates(self):
        """Explicitně zapíše self.pending do candidates.json (podklad k revizi)."""
        try:
            data = json.load(open(self._cand_path, encoding="utf-8"))
        except FileNotFoundError:
            data = {"_comment": "CANDIDATE VZOR→role (živě odvozené, ČEKAJÍ na revizi)."}
        for vz, role in self.pending.items():
            data[vz] = {"role": role, "status": "pending"}
        json.dump(data, open(self._cand_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return len(self.pending)

    def promote(self, vzory, source=None):
        """Po revizi: povýší dané VZORy na CONFIRMED (zapíše determination.json).
        Role bere z `source` (dict VZOR→role), jinak z self.pending."""
        src = source or self.pending
        try:
            data = json.load(open(self._det_path, encoding="utf-8"))
        except FileNotFoundError:
            data = {"_comment": "CONFIRMED VZOR→role (ověřeno revizí)."}
        n = 0
        for vz in vzory:
            if vz in src:
                data[vz] = {"role": src[vz], "status": "confirmed"}
                self.confirmed[vz] = src[vz]; n += 1
        json.dump(data, open(self._det_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return n


class Phase1:
    """Kompozice 1a–1d.

    Vstup:  text (str) — věta.
    Výstup: list[(WORD_W_ATTR, role: str, kurátorováno: bool)] — vstup pro fázi 2.
    Konfig: config.json (skládá podřízené třídy).
    Příklad:
        Phase1().run("Odešel do galilejského města Kafarnaum a učil je v sobotu.")
    """
    def __init__(self, config=CONFIG):
        self.annotator = Annotator(config)
        self.standardizer = Standardizer()
        self.vzor = VzorBuilder(config)
        self.curator = Curator(config)

    def run(self, text: str, r=2):
        sent = self.annotator.annotate(text)      # WORD_W_ATTR_ARRAY
        return self.curator.standardize(sent, r)


if __name__ == "__main__":
    for w, role, cur in Phase1().run(
            "Odešel do galilejského města Kafarnaum a učil je v sobotu."):
        print(f"  {w['form']:13} {role:16} {'[KURÁTOR]' if cur else ''}")
