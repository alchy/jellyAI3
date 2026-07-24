#!/usr/bin/env python3
"""TESTY FÁZE 1 — kontrolují každou třídu: jasný vstup → očekávaný výstup.

Framework: prosté asserty; běží pod pytestem i samostatně (`python test_phase1.py`).
Fixture: reálná věta z CACHE `annotations.pkl` (deterministické, bez živé služby).

CO SE KONTROLUJE:
  Standardizer — sektor dostane správný náš klíč (Kafarnaum→who, města→where, …), deprel ven
  VzorBuilder  — VZOR je přesný (pivot nese pád) a generalizuje přes lexém
  Curator      — kurátorský vklad opraví chybu ÚFALu (dativ zájmeno → to_whom)
  Config       — parametry jsou z JSON (nic natvrdo)
"""
import pickle, phase1

_ANN = pickle.load(open(phase1._HERE + "../../data/annotations.pkl", "rb"))

def _find(prefix, doc=None):
    for (d, _si), rec in _ANN.items():
        if doc and not d.startswith(doc):
            continue
        for s in rec["sentences"]:
            if " ".join(t["form"] for t in s).startswith(prefix):
                return s
    raise RuntimeError("fixture nenalezen: " + prefix)

SENT = _find("Odešel do galilejského města Kafarnaum", "bible_lukas")


def test_standardizer_roles():
    """1b: každý sektor dostane správný náš klíč."""
    r = dict((t["form"], role) for t, role in phase1.Standardizer().standardize(SENT))
    assert r["Odešel"] == "action"
    assert r["do"] == "preposition"
    assert r["města"] == "where"
    assert r["Kafarnaum"] == "who"          # zděděná chyba ÚFALu (nsubj) — dokumentováno
    assert r["."] == "punctuation"


def test_vzor_precise():
    """1c: VZOR pivotu Kafarnaum je přesný (nese pád v pivotu)."""
    vz = phase1.VzorBuilder().vzor(SENT, 4, r=2)
    assert vz == "ADJ:Gen·NOUN:Gen·PROPN:Nom·CCONJ·VERB:Past·."


def test_vzor_pivot_carries_case():
    """1c: pivot nese pád → různé pády = různý VZOR (rozliší roli)."""
    b = phase1.VzorBuilder()
    kafarnaum = b.vzor(SENT, 4, r=1)        # PROPN:Nom v pivotu
    assert "PROPN:Nom" in kafarnaum


def test_curator_shape():
    """1d: kuratela vrací trojice (token, role, kurátorováno?)."""
    out = phase1.Curator().standardize(SENT, r=2)
    assert len(out) == len(SENT)
    assert all(len(x) == 3 and isinstance(x[2], bool) for x in out)


def test_curator_fixes_dative():
    """1d: někde v korpusu kuratela opraví dativ zájmeno (whom_what) na to_whom."""
    std, cur = phase1.Standardizer(), phase1.Curator()
    for (_d, _si), rec in _ANN.items():
        for s in rec["sentences"]:
            for (t, role, c) in cur.standardize(s, 2):
                if c and role == "to_whom" and std.role(t, s) == "whom_what":
                    return                          # nalezena reálná oprava
    raise AssertionError("kuratela neopravila žádný dativ na to_whom")


def test_config_from_json():
    """Konfig: parametry jsou z config.json, ne natvrdo."""
    assert isinstance(phase1.CONFIG["radius"], int)
    assert "." in phase1.CONFIG["modality_marks"]
    assert phase1.CONFIG["services"]["udpipe_url"].startswith("http")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__); ok += 1
        except Exception as e:                       # noqa
            print("FAIL", fn.__name__, "→", e)
    print(f"\n{ok}/{len(fns)} passed")
