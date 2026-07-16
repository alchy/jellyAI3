"""Normalizace výstupu jazykových služeb — oprava tokenizace patternem.

Tokenizace trhá tečkované zkratky („R.U.R." → R/./U/./R/.), lemmatizace pak
nad fragmenty halucinuje (U→„United") a NER vrací useknuté paskvily („R.U").
Oprava se děje na **jediném chokepointu** (`UfalClient.parse`/`entities`),
takže korpus i otázky projdou týmž kódem. Žádný výčet zkratek — univerzální
pattern: běh ≥2 těsně navazujících párů ⟨jednopísmenný token⟩⟨tečka⟩.
"""

import re

# ≥2 páry ⟨písmeno⟩⟨.⟩ přímo v textu (Unicode písmeno, ne číslice/podtržítko)
_ABBREV = re.compile(r"(?:[^\W\d_]\.){2,}")


def _letter(tok):
    """Jednopísmenný alfabetický token."""
    form = tok.get("form", "")
    return len(form) == 1 and form.isalpha()


def _adjacent(left, right):
    """Tokeny na sebe těsně navazují (bez mezery)."""
    return left.get("end") is not None and left.get("end") == right.get("start")


def _run_length(sent, start):
    """Délka zkratkového běhu ⟨písmeno⟩⟨.⟩⟨písmeno⟩⟨.⟩… od indexu (0 = žádný).

    Vyžaduje ≥2 páry a těsné navazování offsetů — jediná iniciála („K.")
    ani písmena oddělená mezerami běh netvoří.
    """
    j = start
    prev_dot = None
    while (j + 1 < len(sent) and _letter(sent[j])
           and sent[j + 1].get("form") == "." and _adjacent(sent[j], sent[j + 1])
           and (prev_dot is None or _adjacent(prev_dot, sent[j]))):
        prev_dot = sent[j + 1]
        j += 2
    length = j - start
    return length if length >= 4 else 0


def merge_abbreviations(sentences):
    """Sloučí rozsekané tečkované zkratky do jednoho tokenu (po větách).

    Sloučený token dědí syntax (head/deprel) prvního písmene, form=lemma je
    celá zkratka („R.U.R."), upos PROPN; 1-based `head` odkazy celé věty se
    přemapují.

    Args:
        sentences (list[list[dict]]): Věty s tokeny (form, head, start/end…).

    Returns:
        list[list[dict]]: Věty s normalizovanými tokeny (kopie).
    """
    return [_merge_sentence(sent) for sent in sentences]


def _merge_sentence(sent):
    """Normalizuje jednu větu (viz `merge_abbreviations`)."""
    out, id_map = [], {}
    i = 0
    while i < len(sent):
        run = _run_length(sent, i)
        if run:
            tokens = sent[i:i + run]
            merged = dict(tokens[0])
            merged["form"] = merged["lemma"] = "".join(
                t.get("form", "") for t in tokens)
            merged["upos"] = "PROPN"
            merged["end"] = tokens[-1].get("end")
            out.append(merged)
            for old in range(i + 1, i + run + 1):
                id_map[old] = len(out)
            i += run
        else:
            out.append(dict(sent[i]))
            id_map[i + 1] = len(out)
            i += 1
    for tok in out:
        head = tok.get("head", 0)
        if head:
            tok["head"] = id_map.get(head, head)
    return out


def expand_abbreviation_entities(text, entities):
    """Roztáhne NER entity useknuté uvnitř tečkované zkratky na celou zkratku.

    NameTag nad rozsekanou zkratkou vrací fragmenty („R.U"); entita překrývající
    zkratkový match v textu se nastaví na celý match. Duplicitní spany se srazí
    (drží se první — kontejnerová entita předchází fragmentům).

    Args:
        text (str): Původní text věty.
        entities (list[dict]): Entity s text/type/start/end.

    Returns:
        list[dict]: Normalizované entity (kopie).
    """
    spans = [(m.start(), m.end(), m.group()) for m in _ABBREV.finditer(text)]
    out, seen = [], set()
    for entity in entities:
        item = dict(entity)
        start, end = item.get("start"), item.get("end")
        for s, e, group in spans:
            if start is not None and end is not None \
                    and start < e and end > s and (start, end) != (s, e):
                item.update(text=group, start=s, end=e)
                break
        key = (item.get("start"), item.get("end"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
