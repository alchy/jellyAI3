#!/usr/bin/env python3
"""hypothesis-two — SPRÁVNÝ match: živá otázka ↔ SYNTETICKÁ otázka (ne fakt).

Match NEpatří mezi otázku a fakt (různý slovosled → 0), ale mezi živou otázku
a syntetickou otázku, kterou si SAMI generujeme z faktu (stejný slovosled).
Protože šablonu definujeme sami při daném r, shodná otázka se trefí při JAKÉMKOLI r.
r je jen dial jistoty: parafráze se při r=2 shodne méně (přísnější) než při r=1.

Měří překryv rámů (podíl rámů živé otázky nalezených v šabloně) při r=1 a r=2.
Vyžaduje UDPipe :8092.
"""
import sys, json, importlib.util, urllib.request
sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb
vb.serve = lambda *a, **k: None
_spec = importlib.util.spec_from_file_location(
    "h1run", "/Users/j/Projects/jellyAI3/experiments/hypothesis-one/run.py")
_run = importlib.util.module_from_spec(_spec); sys.modules["h1run"] = _run
_spec.loader.exec_module(_run)
frame_sig = _run.frame_sig
from jellyai.normalize import merge_abbreviations
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "PRON", "ADP"}   # i tázací PRON + předložka (nesou slovosled)

def udpipe(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:8092/parse", data=data,
                                 headers={"Content-Type": "application/json"})
    return [t for s in merge_abbreviations(json.loads(
        urllib.request.urlopen(req, timeout=10).read())["sentences"]) for t in s]

def frames(toks, r):
    return {frame_sig(toks, i, "", r) for i, t in enumerate(toks) if t["upos"] in CONTENT}

# (syntetická šablona = generujeme z faktu, živá otázka, popis)
PAIRS = [
    ("Kdo patřil mezi pátečníky?", "Kdo patřil mezi pátečníky?",  "shodná"),
    ("Kdo patřil mezi pátečníky?", "Kdo mezi pátečníky patřil?",  "parafráze (slovosled)"),
    ("Kdo napsal R.U.R.?",         "Kdo napsal R.U.R.?",          "shodná"),
    ("Kdo napsal R.U.R.?",         "Kdo je autorem R.U.R.?",      "parafráze (jiná vazba)"),
    ("Co kázal Ježíš?",            "Co kázal Ježíš?",             "shodná"),
    ("Kdo objevil mloky?",         "Kdo objevil mloky?",          "shodná"),
    ("Kdo byl Karel Čapek?",       "Kdo byl Karel Čapek?",        "shodná"),
]

def main():
    print(f"{'syntetická šablona':30} {'živá otázka':30} {'popis':22} {'r1':>5} {'r2':>5}")
    print("─" * 96)
    for tmpl, live, desc in PAIRS:
        tt, lt = udpipe(tmpl), udpipe(live)
        def ov(r):
            lf, tf = frames(lt, r), frames(tt, r)
            return len(lf & tf) / max(len(lf), 1)
        print(f"{tmpl:30} {live:30} {desc:22} {ov(1):>5.0%} {ov(2):>5.0%}")

if __name__ == "__main__":
    main()
