"""Vykonavatel vzorů — regulární sekvence tříd nad lexerem (#46 fáze 2).

Vzor je PŘÍSNĚ regulární (spec 2026-07-19-vzorova-gramatika §7): prvky jsou
třídy, literály a volitelnost — ŽÁDNÉ podmínky, proměnné ani vnořování.
Match je ukotvený na CELOU sekvenci tokenů, aby si vzor nekradl podobné
věty jiného smyslu.

Syntaxe prvku (řetězec):
    "otaz"           token má třídu otaz
    "otaz:kdo|koho"  třídu otaz A norm ∈ {kdo, koho}
    ":v|ve|na"       jen norm (literál, deakcentovaně)
    "?…"             volitelný prvek (kterákoli z podob výše)

Vazby: 1-based index PRVKU vzoru → TaggedToken (volitelný nenaplněný → None).
"""


def _element_matches(element, token):
    """Sedí prvek vzoru (bez prefixu ?) na token?"""
    cls, _, norms = element.partition(":")
    if cls and cls not in token.classes:
        return False
    if norms and token.norm not in norms.split("|"):
        return False
    return True


def match_sequence(pattern, tagged):
    """Ukotvený match vzoru na celou sekvenci tokenů.

    Args:
        pattern (list[str]): Prvky vzoru (syntaxe v docstringu modulu).
        tagged (list[TaggedToken]): Výstup lexeru.

    Returns:
        dict[int, TaggedToken | None] | None: Vazby 1-based indexů prvků,
        nebo None, když vzor nesedí.
    """
    def walk(p, t, binding):
        if p == len(pattern):
            return binding if t == len(tagged) else None
        element = pattern[p]
        optional = element.startswith("?")
        body = element[1:] if optional else element
        if t < len(tagged) and _element_matches(body, tagged[t]):
            found = walk(p + 1, t + 1, {**binding, p + 1: tagged[t]})
            if found is not None:
                return found
        if optional:
            return walk(p + 1, t, {**binding, p + 1: None})
        return None

    return walk(0, 0, {})
