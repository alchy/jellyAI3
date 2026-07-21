#!/usr/bin/env python3
"""hypothesis-one — graf slov v základním tvaru, jas = tf·idf, viewBase 2D.

Iterace 3 — váha slova není holá četnost, ale tf·idf:
  jak často je slovo ZDE (segment = článek)  ×  jak je VZÁCNÉ ve větším korpusu
  (pozadí = celý paměťový text = všechny dokumenty).

  tf(w)   = f_S(w)                         # četnost základního tvaru v segmentu
  idf(w)  = ln( (N_C + 1) / (f_C(w) + 1) ) # vzácnost v korpusu (N_C = tokenů korpusu)
  váha(w) = tf(w) · idf(w)                  # emise ∝ váha (jas uzlu)

Důsledek: „v/a/být" mají f_C obrovské → idf ≈ 0 → skoro nesvítí (chudinky).
„Karel/Čapek" jsou zde časté, jinde vzácné → vysoké idf → svítí.

Ostatní jako dřív: kanonizace do 1. pádu (lemma), uzel = základní tvar,
všechny atributy v meta (detail okno), hrany = sousednost + mesh λ^d,
per-hranový jas přes edge.meta.brightness (dodělané ve viewBase), obyčejné čáry.
"""
import sys, pickle, math, json
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/j/Projects/jellyAI3")
import viewbase as vb

ROOT = "/Users/j/Projects/jellyAI3"
# konfig NENÍ natvrdo — čte se z config.json (zákon 3 i pro parametry VZORu)
_CONFIG = json.load(open(ROOT + "/experiments/hypothesis-one/config.json", encoding="utf-8"))
CFG = {"lang": _CONFIG.get("lang", "cs"), "radius": _CONFIG["radius"],
       "modality_marks": _CONFIG["modality_marks"],
       "lambda": _CONFIG["lambda"], "mesh_eps": _CONFIG["mesh_eps"]}
DOC = "wiki_karel_čapek"
N_SENTS = 25
PUNCT_KEEP = set(_CONFIG["punct_keep"])

def slot(tok):
    """Sektor VZORu: UPOS + nosné syntaktické rysy (pád, slovesný čas). Přesný na
    gramatiku, abstraktní na lexém — jako mluvnický vzor „pán". Pád v pivotu JE role,
    proto musí ve vzoru vždy být (rozliší Kdo=Nom/Komu=Dat/Co=Acc). Přesnost není
    kompromis — je to definice vzoru; pokrytí povrchů dělá KVANTITA vzorů, ne rozmazání."""
    up = tok["upos"]
    if up == "PUNCT":
        return tok["form"] if tok["form"] in PUNCT_KEEP else "PUNCT"
    feats = tok.get("feats") or {}
    vals = [feats[k] for k in ("Case", "Tense") if k in feats]
    return up + (":" + ":".join(vals) if vals else "")

def sentence_modality(toks):
    for t in reversed(toks):
        if t["upos"] == "PUNCT" and t["form"] in CFG["modality_marks"]:
            return t["form"]
    return "."

def frame_sig(toks, i, modality, r=None):
    """VZOR při poloměru r (konfig CFG['radius']): r sektorů vlevo/vpravo + pivot + modalita.
    Sektor = `slot()` = UPOS + pád/čas (přesný na gramatiku, jako vzor „pán"). r řídí rozsah
    okna; pokrytí povrchů = mnoho přesných vzorů (Ollama), ne rozostření jednoho."""
    r = CFG["radius"] if r is None else r
    left = [slot(toks[i - k]) if i - k >= 0 else "^" for k in range(r, 0, -1)]
    right = [slot(toks[i + k]) if i + k < len(toks) else "$" for k in range(1, r + 1)]
    return "·".join(left + [slot(toks[i])] + right + [modality])   # pivot nese pád = roli

import json as _json
# jazyková data (zákon 3): NIKDY české řetězce v kódu; nový jazyk = nový JSON
LANG = _json.load(open(f"{ROOT}/experiments/hypothesis-one/lang/cs.json", encoding="utf-8"))
_VOW = LANG["vowels"]

def _epen_stem(stem):
    """Vlož epentetické -e- do koncového shluku souhlásek (Karl→Karel, Čapk→Čapek).
    Jen mezi písmeny (ne u zkratek typu R.U.R.)."""
    if len(stem) >= 3 and stem[-1].isalpha() and stem[-2].isalpha() \
            and stem[-1].lower() not in _VOW and stem[-2].lower() not in _VOW:
        return stem[:-1] + "e" + stem[-1]
    return stem

def canon_lemma(tok):
    """Základní tvar; přivlastňovací ADJ (Poss=Yes) → base jméno, uniformně:
    Karlova→Karel, Čapkův→Čapek, Bechtěrevovou→Bechtěrev. „Univerzita Karlova"
    se tak rozloží na Karel + univerzita — stejné pravidlo pro všechno, bez výjimek."""
    lemma = tok["lemma"]
    if tok["upos"] == "ADJ" and (tok.get("feats") or {}).get("Poss") == "Yes":
        for suf in LANG["possessive_adj_suffixes"]:
            if lemma.endswith(suf) and len(lemma) > len(suf) + 1:
                return _epen_stem(lemma[:-len(suf)])
    # PROPN epentetický fold (Čapk→Čapek) NENÍ bezpodmínečný — korumpoval by
    # Egypt→Egypet, Petr→Peter…; dělá se PODMÍNĚNĚ nad korpusem (build remap,
    # gen_registry) jen když -e- verze reálně existuje jako častější jméno.
    return lemma

def prep_of(sent, tid):
    """Předložka (case-marker) závislá na tokenu s 1-based id tid, jinak ''."""
    for t in sent:
        if t.get("head") == tid and t.get("deprel") == "case":
            return t["lemma"]
    return ""

def role_key(deprel, upos, feats, prep="", nominal_pred=False):
    """UD deprel (+ pád/životnost/předložka) → JEDNOTNÝ klíč katalogu rolí
    (who/where/whom_what/…). Jediný zdroj mapování — sdílejí gen_registry i roles;
    stejný význam = stejný klíč. Mapa i předložky jsou data v LANG (zákon 3)."""
    if nominal_pred:                         # jmenný přísudek se sponou → stav
        return "state"
    if prep:                                 # předložka může roli přebít (o kom, s kým)
        for key, preps in LANG.get("role_prepositions", {}).items():
            if prep in preps:
                return key
    table = LANG["deprel_to_role"].get(deprel)
    if not table:
        return None
    if "anim" in table:                      # subjekt rozlišuje životnost
        anim = feats.get("Animacy") == "Anim" or bool(feats.get("NameType")) or upos == "PROPN"
        return table["anim"] if anim else table["inanim"]
    return table.get(feats.get("Case", ""), table["default"])

# ---- build -----------------------------------------------------------------
def build():
    with open(f"{ROOT}/data/annotations.pkl", "rb") as f:
        ann = pickle.load(f)

    # POZADÍ (větší korpus = celý paměťový text): tokenová četnost + DOKUMENTOVÁ
    # frekvence (df = v kolika dokumentech slovo je). idf df-based: „v/a/být"
    # jsou ve VŠECH dokumentech → idf≈0 → nesvítí (jsou i v ostatních korpusech).
    corpus = Counter()
    N_C = 0
    docfreq = defaultdict(set)
    doc_ids = set()
    for (doc, _si), rec in ann.items():
        doc_ids.add(doc)
        for sent in rec["sentences"]:
            for tok in sent:
                if tok["upos"] != "PUNCT":
                    corpus[tok["lemma"]] += 1
                    N_C += 1
                    docfreq[tok["lemma"]].add(doc)
    N_docs = len(doc_ids)

    # SEGMENT: rendered článek
    W = defaultdict(lambda: {"freq": 0, "forms": Counter(), "upos": Counter(),
                             "case": set(), "gender": set(), "tense": set(),
                             "frames": Counter(), "is_name": False})
    adj = Counter()
    stream = []
    persons = []
    for si in range(N_SENTS):
        rec = ann.get((DOC, si))
        if rec is None:
            continue
        rec_tokens = []
        for sent in rec["sentences"]:
            modality = sentence_modality(sent)
            prev_word = None
            for i, tok in enumerate(sent):
                gidx = len(stream)
                lemma = canon_lemma(tok)
                # neslovo: interpunkce + jednopísmenné iniciály („T. G. Masaryk")
                skip = tok["upos"] == "PUNCT" or (
                    len(lemma) == 1 and tok["upos"] in ("NOUN", "PROPN", "SYM", "X"))
                stream.append({"lemma": None if skip else lemma, "gidx": gidx})
                rec_tokens.append({"tok": tok, "gidx": gidx, "lemma": lemma})
                if skip:
                    continue
                w = W[lemma]
                w["freq"] += 1
                w["forms"][tok["form"]] += 1
                w["upos"][tok["upos"]] += 1
                fe = tok["feats"]
                if fe.get("Case"):   w["case"].add(fe["Case"])
                if fe.get("Gender"): w["gender"].add(fe["Gender"])
                if fe.get("Tense"):  w["tense"].add(fe["Tense"])
                w["frames"][frame_sig(sent, i, modality)] += 1
                if prev_word is not None:
                    adj[(prev_word, lemma)] += 1
                prev_word = lemma
        for ent in rec.get("entities", []):
            if ent["type"] != "P":
                continue
            hit = [rt for rt in rec_tokens
                   if rt["tok"]["start"] >= ent["start"]
                   and rt["tok"]["end"] <= ent["end"] and rt["tok"]["upos"] != "PUNCT"]
            if not hit:
                continue
            for rt in hit:
                W[rt["lemma"]]["is_name"] = True
            persons.append({"lemmas": [rt["lemma"] for rt in hit],
                            "gidx_end": hit[-1]["gidx"]})

    # FOLD artefaktů lemmat příjmení: epentetické -e- (Čapků→„Čapk" → Čapek).
    # UDPipe u oblique/plurálu utne vloženou -e- ; vlož ji zpět a je-li výsledek
    # známé (a četnější) PROPN, slouč. Jen vlastní jména, jen pokud kandidát žije.
    def _epen(lem):
        vow = "aeiouyáéíóúůýěäö"
        if len(lem) >= 3 and lem[-1].lower() not in vow and lem[-2].lower() not in vow:
            return lem[:-1] + "e" + lem[-1]
        return None
    remap = {}
    for lem, w in W.items():
        cand = _epen(lem) if "PROPN" in w["upos"] else None
        if cand and cand in W and "PROPN" in W[cand]["upos"] \
                and W[cand]["freq"] >= w["freq"]:
            remap[lem] = cand
    if remap:
        merged = defaultdict(lambda: {"freq": 0, "forms": Counter(), "upos": Counter(),
                                      "case": set(), "gender": set(), "tense": set(),
                                      "frames": Counter(), "is_name": False})
        for lem, w in W.items():
            m = merged[remap.get(lem, lem)]
            m["freq"] += w["freq"]
            m["forms"] += w["forms"]; m["upos"] += w["upos"]; m["frames"] += w["frames"]
            m["case"] |= w["case"]; m["gender"] |= w["gender"]; m["tense"] |= w["tense"]
            m["is_name"] = m["is_name"] or w["is_name"]
        W = merged
        adj = Counter({(remap.get(a, a), remap.get(b, b)): c
                       for (a, b), c in adj.items()
                       if remap.get(a, a) != remap.get(b, b)})
        for nd in stream:
            if nd["lemma"] in remap:
                nd["lemma"] = remap[nd["lemma"]]
        for p in persons:
            p["lemmas"] = [remap.get(x, x) for x in p["lemmas"]]

    # tf·idf: emise ∝ (četnost ZDE) × (vzácnost v korpusu, df-based)
    for lem, w in W.items():
        w["corpus_freq"] = corpus.get(lem, 0)
        w["df"] = len(docfreq.get(lem, ()))
        w["idf"] = math.log((N_docs + 1) / (w["df"] + 1))
        w["tfidf"] = w["freq"] * w["idf"]
    tfmax = max((w["tfidf"] for w in W.values()), default=1.0) or 1.0

    # mesh λ^d z jmen
    lam, eps = CFG["lambda"], CFG["mesh_eps"]
    kmax = 1
    while lam ** kmax >= eps:
        kmax += 1
    mesh = {}
    for p in persons:
        src = p["lemmas"][-1]
        for d in range(1, kmax + 1):
            q = p["gidx_end"] + d
            if q >= len(stream):
                break
            tgt = stream[q]["lemma"]
            if tgt is None:
                continue
            wv = round(lam ** d, 4)
            if wv > mesh.get((src, tgt), 0):
                mesh[(src, tgt)] = wv
    return {"W": W, "adj": adj, "mesh": mesh, "tfmax": tfmax, "N_C": N_C,
            "N_docs": N_docs, "corpus_uniq": len(corpus), "kmax": kmax,
            "n_tokens": len(stream), "remap": remap}

# ---- barvy -----------------------------------------------------------------
def lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))

# ---- viewBase --------------------------------------------------------------
def to_canvas(g):
    W, tfmax = g["W"], g["tfmax"]
    emis = {lem: w["tfidf"] / tfmax for lem, w in W.items()}      # emise ∝ tf·idf

    canvas = vb.Canvas(title="hypothesis-one · slova · jas = tf·idf",
                       dimensions=3, theme="cyber", highlight_neighbors=1)
    canvas.detail_window()
    canvas.set_edge_style("line", 0.0)
    canvas.define_type("slovo", color="#2b3350", size=0.7)
    canvas.define_type("jmeno", color="#3a2f14", size=0.9)

    for lem, w in W.items():
        e = emis[lem]
        is_name = w["is_name"]
        base, hi = ("#3a2f14", "#ffcf7a") if is_name else ("#2b3350", "#dfe9ff")
        canvas.add_node(
            lem, type="jmeno" if is_name else "slovo", label=lem,
            color=lerp(base, hi, e ** 0.7), size=round(0.6 + 2.0 * e, 3),
            mass=round(e, 4),                                # gravitace ∝ tf·idf
            **{
                "lemma (základ)": lem,
                "tf·idf (jas)": f"{w['tfidf']:.2f}  →  emise {e:.2f}",
                "tf (zde)": w["freq"],
                "idf (df-based)": f"{w['idf']:.2f}",
                "dokumentů (df)": f"{w['df']} / {g['N_docs']}",
                "korpus. četnost": w["corpus_freq"],
                "upos": ", ".join(f"{u}×{n}" for u, n in w["upos"].most_common()),
                "tvary": ", ".join(f"{f}×{n}" for f, n in w["forms"].most_common(6)),
                "pády": ", ".join(sorted(w["case"])) or "—",
                "rod": ", ".join(sorted(w["gender"])) or "—",
                "čas": ", ".join(sorted(w["tense"])) or "—",
                "jméno (NER)": "ano" if is_name else "ne",
                "nejčastější rám": w["frames"].most_common(1)[0][0] if w["frames"] else "—",
            })

    amax = max(g["adj"].values(), default=1)
    raw = []                              # (a, b, kind, weight, syrový jas)
    for (a, b), c in g["adj"].items():
        if a == b or a not in W or b not in W:
            continue
        raw.append((a, b, "sousednost", c,
                    0.5 * (emis[a] + emis[b]) * (0.4 + 0.6 * c / amax)))
    for (src, tgt), wv in g["mesh"].items():
        if src == tgt or src not in W or tgt not in W:
            continue
        raw.append((src, tgt, "mesh", wv, emis.get(src, 0) * wv))
    # NORMALIZACE jasu hran na viditelný rozsah: /max, odmocnina (zvedne slabé),
    # práh 0.12 (i nejslabší hrana je vidět), špička = 1.0 → hrany opravdu škálují
    # dedup (bigram > mesh), pak jas dle POŘADÍ — rovnoměrné rozložení [0.06,0.86]
    # (ne dle skew hodnot: půlka hran je tmavá, půlka jasná, žádné slití doběla)
    pair = {}
    for a, b, kind, w, rv in raw:
        key = (a, b) if a <= b else (b, a)
        prev = pair.get(key)
        if prev is None or (kind == "sousednost" and prev[2] != "sousednost"):
            pair[key] = (a, b, kind, w, rv)
    items = list(pair.values())
    order = sorted(range(len(items)), key=lambda i: items[i][4])
    n = max(1, len(items) - 1)
    bright = [0.0] * len(items)
    for rank, i in enumerate(order):
        bright[i] = round(0.06 + 0.80 * rank / n, 4)
    for i, (a, b, kind, w, rv) in enumerate(items):
        canvas.ensure_edge(a, b, kind=kind, weight=w, brightness=bright[i])
    return canvas, emis

# ---- info okno -------------------------------------------------------------
def info_window(canvas, g, emis):
    w = vb.TerminalWindow("info", title="⚡ hypothesis-one · tf·idf", prompt="",
                          closable=False, input=False)
    canvas.open_terminal(w, on_input=lambda e: None)
    top = sorted(g["W"].items(), key=lambda kv: -kv[1]["tfidf"])[:16]
    lines = [f"segment: {DOC} ({N_SENTS} vět) · korpus: {g['N_docs']} dokumentů / "
             f"{g['corpus_uniq']} slov",
             "jas = tf·idf = (četnost ZDE) × ln((D+1)/(df+1))  [df = v kolika dok.]", "",
             f"{'slovo':16} {'tf':>3} {'df':>3} {'idf':>5} {'tf·idf':>7}  emise"]
    for lem, ww in top:
        e = emis[lem]
        tag = " «jméno»" if ww["is_name"] else ""
        lines.append(f"{lem:16} {ww['freq']:>3} {ww['df']:>3} {ww['idf']:>5.2f} "
                     f"{ww['tfidf']:>7.2f}  {'█'*round(e*10) or '·'}{tag}")
    lines += ["", "v/a/být: velké korpus.četnost → idf malé → skoro nesvítí.",
              "hrany: sousednost + mesh λ^d, jas přes edge.brightness."]
    canvas.terminal_write(w.window_id, "\n".join(lines))

def _to_red(hexc):
    """Přesune jas do ČERVENÉHO kanálu (modrá→červená), jas/luminanci nechá.
    Zvládne obě palety: modrá slova (B→R) i teplá jména (zůstanou) → jednotně
    červená věta. R'=max(R,B), B'=min(R,B), G beze změny."""
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return "#%02x%02x%02x" % (max(r, b), g, min(r, b))

def show_sentence(canvas, g, match):
    """Zvýrazní VĚTU (uzly + hrany) v červeném odstínu — zamění modrý kanál za
    červený, jas (glow) nechává beze změny. Větu najde dle podřetězce forem.

    Uzly: přebarví aktuální barvu (swap R↔B). Hrany: nastaví explicitní color =
    swap R↔B display barvy, kterou by frontend nakreslil z jasu (renderer color ctí).
    """
    with open(f"{ROOT}/data/annotations.pkl", "rb") as f:
        ann = pickle.load(f)
    remap = g.get("remap", {})
    target = None
    for si in range(N_SENTS):
        rec = ann.get((DOC, si))
        if not rec:
            continue
        for sent in rec["sentences"]:
            if match in " ".join(t["form"] for t in sent):
                target = sent
                break
        if target:
            break
    if target is None:
        return 0
    lemmas = []
    for tok in target:
        lem = canon_lemma(tok)
        lem = remap.get(lem, lem)
        skip = tok["upos"] == "PUNCT" or (
            len(lem) == 1 and tok["upos"] in ("NOUN", "PROPN", "SYM", "X"))
        if not skip:
            lemmas.append(lem)
    for lem in dict.fromkeys(lemmas):                    # uzly červeně
        if not canvas.has_node(lem):
            continue
        col = canvas.node(lem)["meta"].get("color")
        if col:
            canvas.update_node(lem, color=_to_red(col))
    for a, b in zip(lemmas, lemmas[1:]):                 # hrany červeně
        if a == b or not canvas.has_edge(a, b):
            continue
        br = canvas.edge(a, b)["meta"].get("brightness", 0.5)
        disp = lerp("#1f4f6e", "#6fb8e8", br)            # jako frontend (base→glow)
        canvas.ensure_edge(a, b, color=_to_red(disp))
    return len(lemmas)

def top5_window(canvas, g):
    w = vb.TerminalWindow("top5", title="🏆 TOP 5 slov", prompt="",
                          closable=False, input=False)
    canvas.open_terminal(w, on_input=lambda e: None)
    top = sorted(g["W"].items(), key=lambda kv: -kv[1]["tfidf"])[:5]
    lines = [f"{i}. {lem:14} tf·idf {ww['tfidf']:.2f}  (tf {ww['freq']}, df {ww['df']}/{g['N_docs']})"
             for i, (lem, ww) in enumerate(top, 1)]
    canvas.terminal_write(w.window_id, "\n".join(lines))


if __name__ == "__main__":
    g = build()
    canvas, emis = to_canvas(g)
    info_window(canvas, g, emis)
    top5_window(canvas, g)
    n = show_sentence(canvas, g, "Mezi pátečníky patřili")   # ta věta červeně
    print(f"věta zvýrazněna červeně: {n} uzlů")
    top = sorted(g["W"].items(), key=lambda kv: -kv[1]["tfidf"])[:12]
    print(f"slov(unik) {len(g['W'])} · korpus {g['corpus_uniq']}/{g['N_C']} · "
          f"bigramů {len(g['adj'])} · mesh {len(g['mesh'])}")
    print("TOP tf·idf:", ", ".join(f"{l}({w['tfidf']:.1f})" for l, w in top))
    print("→ viewBase 2D na http://127.0.0.1:8080")
    vb.serve(canvas, port=8080, open_browser=False)
