"""Základní textové utility sdílené napříč knihovnou.

Tokenizace a dělení na věty jsou první krok každého retrieval/QA systému:
než můžeme text porovnávat nebo indexovat, musíme ho převést na jednotky
(slova, věty). Tyto funkce záměrně nedělají nic „chytrého" (žádný stemming
ani neuronové modely) — jsou to jednoduché, čitelné a předvídatelné stavební
kameny, na kterých stojí retriever i answerer.
"""

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_BOUNDARY_RE = re.compile(r"[.!?…]\s+")

# České zkratky, po nichž se věta nemá dělit (tečka k nim patří, ne konec věty).
_ABBREV = {
    "mudr", "judr", "phdr", "rndr", "ing", "prof", "doc", "csc", "mgr", "bc",
    "tzv", "atd", "apod", "např", "tj", "mj", "tzn", "aj", "resp", "popř", "cca",
    "č", "r", "st", "sv", "kap", "obr", "tab", "roč", "s", "str", "stol",
}

# Malá sada českých stopslov (funkční slova, tázací zájmena, spojky, předložky).
# Odstranění stopslov zlepšuje retrieval: slova jako „kdo", „je", „na" se
# vyskytují skoro všude, nenesou význam dotazu a jen zašumují skóre podobnosti.
CZECH_STOPWORDS = {
    "a", "aby", "aj", "ale", "ani", "ano", "az", "až", "bez", "bude", "budou",
    "by", "byl", "byla", "byli", "bylo", "být", "co", "což", "či", "do", "i",
    "je", "jeho", "jej", "její", "jejich", "jen", "ještě", "jak", "jako", "jsem",
    "jsi", "jsme", "jsou", "jste", "k", "kam", "kde", "kdo", "když", "ke", "která",
    "které", "který", "kteří", "ku", "ma", "má", "mají", "mé", "mezi", "mi", "mít",
    "mne", "mně", "můj", "my", "na", "nad", "nade", "nam", "nám", "napiš", "naš",
    "náš", "ne", "nebo", "něco", "něj", "není", "než", "ní", "nic", "o", "od",
    "on", "ona", "oni", "ono", "pak", "po", "pod", "podle", "pokud", "pouze",
    "proč", "pro", "před", "přes", "při", "s", "se", "si", "ta", "tak", "také",
    "tam", "te", "tě", "ten", "tedy", "to", "tom", "tomu", "tuto", "ty", "u", "už",
    "v", "vám", "vas", "vás", "ve", "více", "však", "všech", "vy", "z", "za",
    "ze", "že",
}


def tokenize(text, stopwords=None):
    """Rozdělí text na slovní tokeny pro porovnávání a indexaci.

    Cílem je převést libovolný český text na seznam normalizovaných slov, aby
    se dva texty daly porovnat na úrovni slov (retriever i answerer pak počítají
    překryv/podobnost nad těmito tokeny). Slova se převedou na malá písmena kvůli
    porovnání bez ohledu na velikost, ale diakritika se **zachovává** — „kůň"
    a „kun" jsou v češtině různá slova. Volitelně se odfiltrují stopslova.

    Args:
        text (str): Vstupní text (věta, dotaz, pasáž).
        stopwords (set[str] | None): Množina slov k odstranění. Když None,
            neodstraní se nic.

    Returns:
        list[str]: Tokeny v malých písmenech, v pořadí výskytu (i s duplicitami).
    """
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks


def split_sentences(text):
    """Rozdělí text na věty a nenaletí na zkratky ani na data/ordinály.

    Věta je v tomto projektu základní jednotka odpovědi (chunker z nich skládá
    pasáže, extraktivní answerer vrací jednu konkrétní), takže na jejím správném
    vyseknutí hodně záleží. Dělíme na koncové interpunkci s mezerou, ale hranici
    zahodíme, když tečka patří ke zkratce („MUDr.", „tzv.") nebo k pořadovému
    číslu/datu pokračujícímu malým písmenem („9. ledna"). Číslo následované velkým
    písmenem („…číslo 0. Věta…") je naopak normální konec věty.

    Args:
        text (str): Vstupní text (odstavec, pasáž, dokument).

    Returns:
        list[str]: Věty bez okolních bílých znaků; prázdné vynechány. Pro
            prázdný/bílý vstup vrací prázdný seznam.
    """
    text = text.strip()
    if not text:
        return []
    sentences = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        punct_pos = match.start()
        # slovo těsně před interpunkcí (souvislý běh písmen/číslic)
        j = punct_pos
        while j > start and text[j - 1].isalnum():
            j -= 1
        preceding = text[j:punct_pos]
        next_char = text[match.end()] if match.end() < len(text) else ""
        if preceding.lower() in _ABBREV:
            continue                                    # zkratka → nesekat
        if preceding.isdigit() and next_char.islower():
            continue                                    # ordinál/datum uprostřed
        sentence = text[start:punct_pos + 1].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
