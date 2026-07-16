"""Jazyková pravidla jako data — jazyk je zásuvný modul (JSON), ne kód.

Kmenování (pádové koncovky, epenteze, minimální kmen) je jazykově specifické;
core je jazykově agnostický a pravidla načítá z `jellyai/lang/<jazyk>.json`,
případně z libovolné cesty. Nový jazyk = nový JSON soubor, přepnutí = config
(`graph.language`) — bez zásahu do kódu. Stejný vzor je připraven i pro další
jazykové tabulky (relační jména, „work" slovesa, měsíce), až se sem přestěhují.
"""

import json
import os

_DIR = os.path.dirname(__file__)
_cache = {}


def load_rules(language="cs"):
    """Načte jazyková pravidla kmenování z JSON (s cache).

    Args:
        language (str): Kód jazyka (soubor `jellyai/lang/<kód>.json`),
            nebo přímo cesta k vlastnímu `.json` souboru.

    Returns:
        dict: {"min_stem" (int), "suffixes" (tuple, seřazené nejdelší první —
            pořadí v JSON je libovolné), "vowels" (str), "epenthesis_vowel"
            (str; prázdné = jazyk epentezi nemá)}.
    """
    path = language if language.endswith(".json") \
        else os.path.join(_DIR, f"{language}.json")
    if path not in _cache:
        with open(path, encoding="utf-8") as fh:
            rules = json.load(fh)
        rules["suffixes"] = tuple(sorted(rules.get("suffixes", ()),
                                         key=len, reverse=True))
        _cache[path] = rules
    return _cache[path]
