"""Základní textové utility sdílené napříč knihovnou.

Tokenizace a dělení na věty jsou první krok každého retrieval/QA systému:
než můžeme text porovnávat nebo indexovat, musíme ho převést na jednotky
(slova, věty). Tyto funkce záměrně nedělají nic „chytrého" (žádný stemming
ani neuronové modely) — jsou to jednoduché, čitelné a předvídatelné stavební
kameny, na kterých stojí retriever i answerer.
"""

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")

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
    """Rozdělí text na jednotlivé věty podle koncové interpunkce.

    Věta je v tomto projektu základní jednotka odpovědi — chunker z vět skládá
    pasáže a extraktivní answerer vrací jednu konkrétní větu. Dělení je záměrně
    jednoduché (podle `.`, `!`, `?`, `…` následovaných mezerou); nezpracovává
    zkratky ap., ale pro účely retrievalu je to dostatečné a předvídatelné.

    Args:
        text (str): Vstupní text (odstavec, pasáž, dokument).

    Returns:
        list[str]: Věty bez okolních bílých znaků; prázdné věty jsou vynechány.
            Pro prázdný/bílý vstup vrací prázdný seznam.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]
