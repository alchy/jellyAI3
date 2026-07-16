"""Jazyk jako zásuvný modul — pravidla kmenování žijí v JSON, ne v kódu.

Core (canon._stem) je jazykově agnostický algoritmus; koncovky, epenteze a
minimální kmen jsou data z `jellyai/lang/<jazyk>.json`. Nový jazyk = nový JSON,
přepnutí = config — bez zásahu do kódu.
"""

import json

from jellyai.lang import load_rules
from jellyai.graph.canon import cluster_key, set_language


def test_czech_rules_load_from_json():
    rules = load_rules("cs")
    assert "ovi" in rules["suffixes"]                  # dativ (fix stemmeru)
    # loader řadí nejdelší první — v JSON může být pořadí libovolné (doplňování
    # koncovek nesmí vyžadovat znalost matching pořadí)
    assert list(rules["suffixes"]) == sorted(rules["suffixes"], key=len, reverse=True)


def test_custom_language_json_changes_stemming(tmp_path):
    """Nový jazyk = nový JSON soubor, žádná změna kódu."""
    path = tmp_path / "xx.json"
    path.write_text(json.dumps({"min_stem": 2, "suffixes": ["os", "o"],
                                "vowels": "aeiou", "epenthesis_vowel": ""}),
                    encoding="utf-8")
    set_language(str(path))
    try:
        assert cluster_key("Markos Vindo") == cluster_key("Marko Vindos")
    finally:
        set_language("cs")
    assert cluster_key("Markos") != cluster_key("Marko")   # česká pravidla zpět


def test_language_is_config_driven():
    """Jazyk se volí v configu — přepnutí nevyžaduje zásah do kódu."""
    from config import Config
    assert Config().graph.language == "cs"
