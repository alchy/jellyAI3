"""Vykonavatel vzorů — regulární sekvence tříd nad lexerem (#46 fáze 2).

Vzor je PŘÍSNĚ regulární (spec 2026-07-19-vzorova-gramatika §7): prvky jsou
třídy, literály a volitelnost — ŽÁDNÉ podmínky, proměnné ani vnořování.
Match je ukotvený na CELOU sekvenci tokenů, aby si vzor nekradl podobné
věty jiného smyslu.

Syntaxe prvku (řetězec):
    "otaz"           token má třídu otaz
    "otaz:kdo|koho"  třídu otaz A norm ∈ {kdo, koho}
    "l_tvar!spona"   třídu l_tvar a ŽÁDNOU z vyloučených („byl" je
                     hypotézově obojí — sponové otázky nechává jiným)
    ":v|ve|na"       jen norm (literál, deakcentovaně)
    "uzel+"          SPAN 1..n tokenů, který orákulum `is_span` potvrdí
                     jako entitu grafu („Karel Čapek", „Válka s mloky");
                     hladově nejdelší, ustupuje kvůli zbytku vzoru
    "?…"             volitelný prvek (kterákoli z podob výše)

Vazby: 1-based index PRVKU vzoru → TaggedToken; spanový prvek → list
TaggedToken; volitelný nenaplněný → None.
"""


def _element_matches(element, token):
    """Sedí prvek vzoru (bez prefixu ?) na token?"""
    cls_part, _, norms = element.partition(":")
    cls, *excluded = cls_part.split("!")
    if cls and cls not in token.classes:
        return False
    if any(ex in token.classes for ex in excluded):
        return False
    if norms and token.norm not in norms.split("|"):
        return False
    return True


def match_sequence(pattern, tagged, is_span=None):
    """Ukotvený match vzoru na celou sekvenci tokenů.

    Args:
        pattern (list[str]): Prvky vzoru (syntaxe v docstringu modulu).
        tagged (list[TaggedToken]): Výstup lexeru.
        is_span (callable | None): Orákulum `text → bool` pro prvek
            `uzel+` (typicky `_span_is_node` answereru — slovník entit
            je graf). None = spanové prvky nematchnou.

    Returns:
        dict | None: Vazby 1-based indexů prvků (token | list | None),
        nebo None, když vzor nesedí.
    """
    def walk(p, t, binding):
        if p == len(pattern):
            return binding if t == len(tagged) else None
        element = pattern[p]
        optional = element.startswith("?")
        body = element[1:] if optional else element
        if body == "uzel+":
            if is_span is not None:
                # hladově nejdelší potvrzený span; ustupuje, aby zbytek
                # vzoru vyšel (druhou entitu nesmí spolknout první)
                for end in range(len(tagged), t, -1):
                    span = tagged[t:end]
                    if "funkcni" in span[0].classes \
                            or "funkcni" in span[-1].classes:
                        # entita nezačíná/nekončí funkčním slovem — volné
                        # orákulum by pustilo i paskvil („Pavla v"); uvnitř
                        # smí („Válka s mloky")
                        continue
                    if not is_span(" ".join(tok.form for tok in span)):
                        continue
                    found = walk(p + 1, end, {**binding, p + 1: span})
                    if found is not None:
                        return found
        elif t < len(tagged) and _element_matches(body, tagged[t]):
            found = walk(p + 1, t + 1, {**binding, p + 1: tagged[t]})
            if found is not None:
                return found
        if optional:
            return walk(p + 1, t, {**binding, p + 1: None})
        return None

    return walk(0, 0, {})
