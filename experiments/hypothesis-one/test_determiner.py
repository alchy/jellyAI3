#!/usr/bin/env python3
"""TESTY PersistentDeterminer — vrstvení CURATED → CONFIRMED → CANDIDATE + persistence.

Framework: prosté asserty; běží pod pytestem i samostatně. Izolace: determination/
candidates jdou do tempfile adresáře (repo se nemutuje). Fixture z cache annotations.pkl.

CO SE KONTROLUJE:
  provenance  — determine() vrací (WORD_W_ATTR, role, CURATED/CONFIRMED/CANDIDATE)
  miss        — neznámý VZOR → CANDIDATE a je v self.pending (v paměti)
  curated     — ruční VZOR přebíjí re-derivaci (provenance CURATED)
  no-write    — determine() NEZAPISUJE na disk (determinismus + revizní brána)
  persist     — save_candidates() a promote() zapíší; nový determiner načte CONFIRMED
"""
import json, pickle, tempfile, os, copy, phase1

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

def _tmp_config():
    d = tempfile.mkdtemp()
    cfg = copy.deepcopy(phase1.CONFIG)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["determination"] = os.path.join(d, "determination.json")
    cfg["paths"]["candidates"] = os.path.join(d, "candidates.json")
    return cfg                       # curated zůstává reálný (curated.json)


def test_provenance_values():
    out = phase1.PersistentDeterminer(_tmp_config()).determine(SENT)
    D = phase1.PersistentDeterminer
    assert all(len(x) == 3 for x in out)
    assert all(p in (D.CURATED, D.CONFIRMED, D.CANDIDATE) for _w, _r, p in out)


def test_miss_is_candidate_and_pending():
    det = phase1.PersistentDeterminer(_tmp_config())
    out = det.determine(SENT)
    kaf = [p for (w, _r, p) in out if w["form"] == "Kafarnaum"]
    assert kaf and kaf[0] == det.CANDIDATE           # novel VZOR → CANDIDATE
    assert len(det.pending) > 0                      # poznamenáno v paměti


def test_curated_overrides():
    det = phase1.PersistentDeterminer(_tmp_config())
    for (_d, _si), rec in _ANN.items():
        for s in rec["sentences"]:
            if any(p == det.CURATED for _w, _r, p in det.determine(s)):
                return                               # ruční VZOR přebil re-derivaci
    raise AssertionError("žádný CURATED nenalezen")


def test_determine_no_disk_write():
    cfg = _tmp_config()
    det = phase1.PersistentDeterminer(cfg)
    det.determine(SENT)
    assert not os.path.exists(cfg["paths"]["candidates"])    # determine() nic nezapsal
    assert not os.path.exists(cfg["paths"]["determination"])


def test_save_candidates_and_promote():
    cfg = _tmp_config()
    det = phase1.PersistentDeterminer(cfg)
    det.determine(SENT)
    n = det.save_candidates()
    assert n > 0 and os.path.exists(cfg["paths"]["candidates"])
    vzory = list(det.pending)[:2]
    assert det.promote(vzory) == len(vzory)
    # nový determiner načte povýšené jako CONFIRMED
    det2 = phase1.PersistentDeterminer(cfg)
    assert len(det2.confirmed) == len(vzory)
    assert set(det2.confirmed) == set(vzory)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__); ok += 1
        except Exception as e:                       # noqa
            print("FAIL", fn.__name__, "→", e)
    print(f"\n{ok}/{len(fns)} passed")
