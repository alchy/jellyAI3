"""Subsystémy Iris — doménoví experti nad osami reality (spec 2026-07-18).

Každý subsystém (Chronos čas, Mnemos paměť, budoucí Topos prostor, Metron
množství) sdílí týž půdorys — čtyři schopnosti, vše jako JSON data:

* **POZNÁNÍ**    — vzory, jak poznat doménový výraz (karty/tabulky);
* **KANONIZACE** — překlad jazyk ↔ kanonický záznam („za čtvrt hodiny"
  ↔ timestamp, „ráno" ↔ 07:00, „v Praze" ↔ uzel Praha);
* **AKTIVACE**   — reflektor: doménový údaj → jas uzlů v ActivationField
  (jediný způsob, jak subsystém ovlivňuje odpověď);
* **ZÁZNAMY**    — JSONL persistence (deník, připomínky, definice).

Iris je dirigent: vede dialog a rozhoduje KARTAMI; subsystémy nikdy
nemluví s uživatelem přímo. Tři brány zapojení: E (extrakce), Q (nárok
na tokeny otázky → omezení), A (reflektor). Kód subsystému je výhradně
MECHANISMUS — chování nesou karty a jazyková data (ZÁKON projektu).
"""

from jellyai.iris.subsystems.chronos import (ChronosTicker, TimeInterval,
                                             clock_answer, format_due,
                                             resolve_due, resolve_temporal)
from jellyai.iris.subsystems.mnemos import (parse_statement, persist,
                                            remember, replay)

__all__ = ["ChronosTicker", "TimeInterval", "clock_answer", "format_due",
           "resolve_due", "resolve_temporal",
           "parse_statement", "persist", "remember", "replay"]
