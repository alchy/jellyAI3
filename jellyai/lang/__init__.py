"""Jazyková pravidla jako data — jazyk je zásuvný modul (JSON), ne kód.

Kmenování (pádové koncovky, epenteze, minimální kmen) i lexikální tabulky
(autorská podstatná jména/slovesa) jsou jazykově specifické; core je jazykově
agnostický a čte je z `jellyai/lang/<jazyk>.json`, případně z libovolné cesty.
Nový jazyk = nový JSON soubor, přepnutí = config (`graph.language`) — bez
zásahu do kódu. **Aktivní jazyk je stav tohoto modulu** (`set_language`/
`current`) — jediný zdroj pro canon, extract i spread, aby se strany nerozešly.
"""

import json
import os

_DIR = os.path.dirname(__file__)
_cache = {}
_active = {}


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
        rules["work_nouns"] = frozenset(rules.get("work_nouns", ()))
        rules["work_verbs"] = frozenset(rules.get("work_verbs", ()))
        rules["feminine_name_suffixes"] = tuple(
            rules.get("feminine_name_suffixes", ()))
        rules["metalanguage_nouns"] = frozenset(
            rules.get("metalanguage_nouns", ()))
        rules["generic_event_verbs"] = frozenset(
            rules.get("generic_event_verbs", ()))
        rules["interrogative_adverbs"] = frozenset(
            rules.get("interrogative_adverbs", ()))
        rules["predicate_synonyms"] = {
            lemma: tuple(group)
            for group in rules.get("predicate_synonyms", ())
            for lemma in group}
        _cache[path] = rules
    return _cache[path]


def set_language(language="cs"):
    """Aktivuje jazyk procesu (kód jazyka nebo cesta k JSON). Vrátí pravidla."""
    _active["rules"] = load_rules(language)
    return _active["rules"]


def current():
    """Aktivní jazyková pravidla (výchozí „cs", když nikdo nepřepnul)."""
    if "rules" not in _active:
        set_language("cs")
    return _active["rules"]
