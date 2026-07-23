#!/usr/bin/env python3
"""Auto-generuje TRÉNINK pro naučenou assurance bránu z extrahovaných faktů (ne z 77 ručních).

Fáze 1: z faktů (predikát + role) instancuje šablonové otázky se ZNÁMOU odpovědí (slot faktu).
Fáze 2: pro každou spustí answer(return_features) a olabeluje SPRÁVNĚ/ŠPATNĚ (shoda s faktem).
Confident-wrongy vzniknou přirozeně (glow mispickne). Výstup: gate_train.json (rysy + label),
DISJUNKTNÍ od ručního etalonu (145) → poctivé měření generalizace. Spuštění: python3 gen_train.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from answering import Answering

OUT = os.path.join(HERE, "gate_train.json")

# (predikát, slot odpovědi, šablona {s}=podmět) — gramatické otázky, odpověď = slot faktu
TEMPLATES = [
    ("narodit", "when", "Kdy se narodil {s}?"),
    ("narodit", "where", "Kde se narodil {s}?"),
    ("zemřít", "when", "Kdy zemřel {s}?"),
    ("zemřít", "where", "Kde zemřel {s}?"),
    ("být", "state", "Kdo je {s}?"),
    ("žít", "where", "Kde žil {s}?"),
]


def norm(s): return (s or "").strip().lower()
def accept(got, ans):
    a = norm(got)
    return bool(a) and any(g and (a == norm(g) or norm(g) in a or a in norm(g)) for g in ans)


def clear(a):
    a.store.mounted.clear(); a.facts.mounted.clear()
    a.field.words.clear(); a.field.files.clear(); a.field.adj.clear()


def _name(s, idf):
    """Podmět je JMÉNO (PROPN-like): velké písmeno, ≥3 znaky, distinktivní (idf)."""
    return bool(s) and s[:1].isupper() and len(s) >= 3 and idf.get(s.lower(), 0.0) >= 1.5


def collect_examples(a):
    """Fáze 1 — z faktů instancuj (otázka, očekávaná odpověď). FILTR na spolehlivé fakty:
    jen JMÉNA (ne obecná slova bůh/člověk), NE bible (teologický šum), sanity slotu
    (rok pro when, gazetteer pro where, definiční profese pro state). Bez parsování."""
    a.facts.mount(a.facts.document_ids())
    idf = a.field.idf
    ex, seen = [], set()
    for f in a.facts.by_ref.values():
        if f.doc.startswith("bible"):                        # teologie = nespolehlivý ground-truth
            continue
        who = f.roles.get("who", [])
        s = who[0] if who else None
        if not _name(s, idf):
            continue
        for pred, slot, tmpl in TEMPLATES:
            if f.predicate != pred or not f.roles.get(slot):
                continue
            val = [x.lower() for x in f.roles[slot]]
            if slot == "when" and not any(v.isdigit() and len(v) == 4 for v in val):
                continue
            if slot == "where" and not any(a.topos.is_place(v) for v in val):
                continue
            if slot == "state" and (f.sent != 0 or not all(len(v) >= 4 and v.isalpha() for v in val)):
                continue
            q = tmpl.format(s=s)
            if q not in seen:
                seen.add(q); ex.append((q, val))
        if f.predicate == "napsat" and f.roles.get("whom_what"):   # „Kdo napsal {dílo}?" → who
            work = f.roles["whom_what"][0]
            if work[:1].isupper() and len(work) >= 3:
                q = f"Kdo napsal {work}?"
                if q not in seen:
                    seen.add(q); ex.append((q, [x.lower() for x in who]))
    return ex


def main():
    a = Answering()
    ex = collect_examples(a)
    print(f"fáze 1: {len(ex)} šablonových otázek z faktů", flush=True)
    rows = []
    for i, (q, exp) in enumerate(ex):
        clear(a)                                             # answer() si mountne dle dotazu sám
        r = a.answer(q, return_features=True)
        if r and r["mode"] == "answer" and "features" in r:
            rows.append({"feat": r["features"], "correct": int(accept(r["answer"], exp)),
                         "q": q, "got": r["answer"]})
        if i % 50 == 0:
            print(f"  fáze 2: {i}/{len(ex)}", flush=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    pos = sum(r["correct"] for r in rows)
    print(f"\nhotovo: {len(rows)} answer-mode případů → správně {pos}, "
          f"confident-wrong {len(rows) - pos}\n→ {OUT}")


if __name__ == "__main__":
    main()
