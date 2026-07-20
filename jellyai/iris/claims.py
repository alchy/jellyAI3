"""Formální claim() přímých expertů (#57 E2 — zárodek #26 na úrovni grafu).

Expert deklaruje NÁROK na celý tah (přímá brána: výraz Metronu,
hodinová otázka Chronosu, meta-otázka na těžiště) jako záznam registru:
jméno worker uzlu + rozpoznávač. Kompilace otázkového grafu z registru
staví worker uzly a osvětlení volá rozpoznávače jednotně — ruční výčet
expertů v kódu grafu mizí; nový expert = nový claim, žádný zásah do
compile/illuminate/turn.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertClaim:
    """Nárok přímého experta na tah.

    Atributy:
        name (str): Jméno worker uzlu grafu (např. „metron-vypocet“).
        worker (str): Subsystém, který tah odbaví.
        priority (int): Přednost mezi přímými experty (vyšší dřív) —
            zrcadlí dnešní pořadí bran v turn().
        recognize: callable(text, now) -> výsledek | None — nárok
            NESE i výsledek (postřeh 1.3: rozpoznání = výpočet, jednou)
    """
    name: str
    worker: str
    priority: int
    recognize: object


def _metron(text, now):
    from jellyai.iris.subsystems.metron import compute
    return compute(text)


def _chronos(text, now):
    from jellyai.iris.subsystems.chronos import clock_answer
    return clock_answer(text, now)


def _meta_focus(text, now):
    from jellyai.graph.canon import deaccent
    from jellyai.lang import current
    low = deaccent(text.lower())
    return any(p in low for p in current().get("focus_query_phrases", ()))


def default_claims():
    """Nároky dnešních tří přímých expertů (pořadí = přednost bran)."""
    return (
        ExpertClaim("metron-vypocet", "metron", 2, _metron),
        ExpertClaim("chronos-hodiny", "chronos", 1, _chronos),
        ExpertClaim("meta-focus", "iris", 0, _meta_focus),
    )
