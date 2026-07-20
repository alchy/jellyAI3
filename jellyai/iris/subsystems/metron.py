"""Metron — expert na MÍRY A POČTY; tady jeho brána Q nad tokeny řádku.

Aritmetika z výroku uživatele (BACKLOG #56): „Kolik je 1 plus 1?",
„3 * 4", součty pro účtenku („sto, osmdesátdva, třicet, 50, 32 …
dohromady?"). Metron si NÁROKUJE číselné tokeny a operátory (týž
princip jako Chronos claim_words, #26) a počítá VLASTNÍ bezpečnou
aritmetikou — nikdy eval. Čísla nesou tabulky jazyka: cifry, slovní
číslovky včetně SLOŽENIN (osmdesátdva = osmdesát + dva — hladový
rozklad po prefixech, stovky/tisíce násobí), operátory slovem
(operator_words) i znakem. Druhá brána Metronu — počítání FAKTŮ
grafu („Kolikrát pršelo?") — je BACKLOG #11.
"""

import re

from jellyai.graph.canon import deaccent
from jellyai.lang import current

_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[+*/×÷-]|\w+", re.UNICODE)


def _compound_numeral(word, table):
    """Složená slovní číslovka hladovým rozkladem po prefixech
    („osmdesatdva" → 80+2; „dveste" → 2×100). None = není číslovka."""
    parts, rest = [], word
    while rest:
        prefix = next((p for p in sorted(table, key=len, reverse=True)
                       if rest.startswith(p)), None)
        if prefix is None:
            return None
        parts.append(table[prefix])
        rest = rest[len(prefix):]
    total, current_group = 0, 0
    for value in parts:
        if value >= 100:
            current_group = max(current_group, 1) * value
            if value >= 1000:
                total, current_group = total + current_group, 0
        else:
            current_group += value
    return total + current_group


def _number(token, table):
    """Token → číslo (cifry vč. desetinné čárky, slovní číslovka), či None."""
    if re.fullmatch(r"\d+(?:[.,]\d+)?", token):
        return float(token.replace(",", "."))
    return _compound_numeral(deaccent(token.lower()), table)


def _format(value):
    """Číslo česky: celé bez desetin, jinak desetinná ČÁRKA."""
    if float(value).is_integer():
        return str(int(value))
    return f"{round(value, 6):g}".replace(".", ",")


def _evaluate(numbers, operators):
    """Bezpečné vyhodnocení zleva s prioritou * a / (žádný eval)."""
    values, pending = [numbers[0]], []
    for op, number in zip(operators, numbers[1:]):
        if op == "*":
            values[-1] *= number
        elif op == "/":
            if number == 0:
                return None                  # dělení nulou — poctivě mlčet
            values[-1] /= number
        else:
            pending.append(op)
            values.append(number)
    total = values[0]
    for op, value in zip(pending, values[1:]):
        total = total + value if op == "+" else total - value
    return total


def compute(text):
    """Brána Q Metronu: najde v řádku výraz NEBO součtový výčet.

    Výraz = čísla střídaná operátory („1 plus 1", „2 + 3 * 4").
    Součet = ≥2 čísla + fráze z `sum_phrases` („dohromady", „celkem" —
    účtenka). Jinak None — otázky grafu i časy („v 18:30") se nenárokují.

    Returns:
        tuple[str, str] | None: (přepis výrazu, výsledek), nebo None.
    """
    lang = current()
    table = lang.get("word_numerals", {})
    op_words = lang.get("operator_words", {})
    low = deaccent(text.lower())
    if ":" in text:
        return None                          # čas („v 18:30") patří Chronosu
    stream = []                              # čísla a operátory v pořadí textu
    for token in _TOKEN.findall(text):
        number = _number(token, table)
        if number is not None:
            stream.append(("num", number))
            continue
        symbol = (token if token in "+-*/×÷"
                  else op_words.get(deaccent(token.lower())))
        if symbol is not None:
            stream.append(("op", symbol.replace("×", "*").replace("÷", "/")))
    numbers = [v for kind, v in stream if kind == "num"]
    if len(numbers) < 2:
        return None
    alternating = all(kind == ("num" if i % 2 == 0 else "op")
                      for i, (kind, _) in enumerate(stream))
    if alternating and len(stream) >= 3:
        operators = [v for kind, v in stream if kind == "op"]
        result = _evaluate(numbers, operators)
        if result is None:
            return None
        transcript = " ".join(
            _format(v) if kind == "num" else v for kind, v in stream)
        return transcript, _format(result)
    if any(phrase in low for phrase in lang.get("sum_phrases", ())):
        transcript = " + ".join(_format(n) for n in numbers)
        return transcript, _format(sum(numbers))
    return None
