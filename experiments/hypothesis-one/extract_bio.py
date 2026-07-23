#!/usr/bin/env python3
"""Extrakce biografické ZÁVORKY: „Osoba ( datum místo – datum místo )" → narodit/zemřít fakty.

Univerzální wikipediální konstrukce, kterou běžná extrakce zahazuje (podměty se rozpadnou,
datumy nedostanou roli). „Karel Čapek ( 9. ledna 1890 Malé Svatoňovice – 25. prosince 1938
Praha ) byl…" nese narození i úmrtí (datum + místo), ale registry z toho vytěží jen
`být/who=Karel/state=spisovatel`. Tady se závorka rozparsuje: levá půle (před en-pomlčkou)
= narození, pravá = úmrtí; rok = 4místné NUM, místo = PROPN NameType=Geo. Deterministické.

Vstup: anotovaná věta + grammar. Výstup: [(predikát, roles{who/when/where})].
"""


def _year(toks):
    """První 4místný rok (1000–2100) z tokenů, jinak None."""
    for t in toks:
        f = t["form"]
        if f.isdigit() and 1000 <= int(f) <= 2100:
            return f
    return None


def _place(toks, grammar):
    """První GEO entita (NameType=Geo) z tokenů → lemma, jinak None."""
    for t in toks:
        if t["upos"] == "PROPN" and (t.get("feats") or {}).get("NameType") == "Geo":
            return grammar.canon_lemma(t)
    return None


def bio_facts(sent, grammar):
    """Věta → biografické fakty narodit/zemřít ze závorky (who + when + where).

    Osoba = první PROPN se jménem (Giv/Sur) v úvodu věty. Závorka se rozdělí en-pomlčkou
    „–" na narození (vlevo) a úmrtí (vpravo). Bez osoby/závorky/data → []. Vstup: anotovaná
    věta, grammar (kanonizace). Výstup: [(predikát, roles)].
    """
    person = None
    for t in sent[:6]:
        if t["upos"] == "PROPN" and (t.get("feats") or {}).get("NameType") in ("Giv", "Sur"):
            person = grammar.canon_lemma(t)
            break
    if not person:
        return []
    op = next((i for i, t in enumerate(sent) if t["form"] == "("), None)
    cl = next((i for i, t in enumerate(sent) if t["form"] == ")" and (op is None or i > op)), None)
    if op is None or cl is None:
        return []
    inner = sent[op + 1:cl]
    dash = next((i for i, t in enumerate(inner) if t["form"] in ("–", "—")), None)
    halves = [inner] if dash is None else [inner[:dash], inner[dash + 1:]]
    preds = ["narodit", "zemřít"]
    out = []
    for k, half in enumerate(halves[:2]):
        roles = {"who": [person]}
        y = _year(half)
        p = _place(half, grammar)
        if y:
            roles["when"] = [y]
        if p:
            roles["where"] = [p]
        if len(roles) > 1:                       # jen když je aspoň datum nebo místo
            out.append((preds[k], roles))
    return out
