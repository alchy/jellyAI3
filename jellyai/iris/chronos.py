"""Chronos — orientace Iris v ČASE: primitiva → absolutní intervaly.

Iris zná „teď" svého běhu a umí na časovou osu ukotvit relativní výrazy:
„dnes", „včera", „zítra", „za hodinu", „před dvěma hodinami", „před týdnem",
„tento týden", „tento měsíc", „v 19. století"… Výsledkem je vždy **interval**
⟨start, end) s granularitou — i bod v čase je interval své jednotky („za
hodinu" = celá hodina 13:00–14:00).

Zásady:

* **„Teď" je VŽDY parametr** (`now`) — testy a benchmarky ho fixují
  (determinismus!), živé API dodá systémové hodiny.
* **Slovník je jazykové datum** (`cs.json` klíč `temporal`: slova dnů,
  jednotky, směrovky, číslovky, měsíce) — kód nese jen aritmetiku.
* **Napojení na graf** přes `contains_date` nad výstupem `parse_date`
  (rok/měsíc/den časových uzlů): částečné datum („1818") má vlastní interval
  své granularity a testuje se PŘEKRYV — chronos datum interpretuje, nikam
  ho nepřevádí (rok 1900 tak správně patří do 19. století).
"""

import calendar
import re

from dataclasses import dataclass
from datetime import datetime, timedelta

from jellyai.graph.canon import deaccent
from jellyai.lang import current


def _floor(moment, unit):
    """Zarovná okamžik na začátek své jednotky (hodina/den/týden/měsíc/rok)."""
    if unit == "hour":
        return moment.replace(minute=0, second=0, microsecond=0)
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "day":
        return day
    if unit == "week":
        return day - timedelta(days=day.weekday())      # pondělí
    if unit == "month":
        return day.replace(day=1)
    return day.replace(month=1, day=1)                  # year


def _shift(moment, unit, n):
    """Posune okamžik o n jednotek (kalendářně u měsíců/roků, jinak deltou)."""
    if unit == "hour":
        return moment + timedelta(hours=n)
    if unit == "day":
        return moment + timedelta(days=n)
    if unit == "week":
        return moment + timedelta(weeks=n)
    if unit == "month":
        total = moment.year * 12 + (moment.month - 1) + n
        year, month = total // 12, total % 12 + 1
        day = min(moment.day, calendar.monthrange(year, month)[1])
        return moment.replace(year=year, month=month, day=day)
    return moment.replace(year=moment.year + n)         # year


@dataclass(frozen=True)
class TimeInterval:
    """Půlotevřený časový interval ⟨start, end) s granularitou.

    Atributy:
        start (datetime): Začátek (včetně).
        end (datetime): Konec (vyjma).
        granularity (str): hour | day | week | month | year | century.
    """
    start: datetime
    end: datetime
    granularity: str

    def contains(self, moment):
        """True, když okamžik leží v intervalu (start ≤ t < end)."""
        return self.start <= moment < self.end

    def contains_date(self, parsed):
        """Příslušnost data z grafu (výstup `parse_date`) — PŘEKRYVEM.

        Částečné datum má vlastní interval své granularity („1818" = celý
        rok), takže i hrubě datovaný fakt se do jemného intervalu trefí,
        když se překrývají. Neukotvené datum (bez roku) nesvítí.

        Args:
            parsed (dict): Podmnožina {"rok", "měsíc" (nominativ), "den"}.

        Returns:
            bool: True při překryvu intervalů.
        """
        if not parsed or "rok" not in parsed:
            return False
        months = current()["temporal"].get("months", {})
        year = int(parsed["rok"])
        month = months.get(deaccent(str(parsed.get("měsíc", "")).lower()))
        day = int(parsed["den"]) if parsed.get("den") else None
        if month and day:
            lo = datetime(year, month, day)
            hi = lo + timedelta(days=1)
        elif month:
            lo = datetime(year, month, 1)
            hi = _shift(lo, "month", 1)
        else:
            lo = datetime(year, 1, 1)
            hi = datetime(year + 1, 1, 1)
        return lo < self.end and self.start < hi


def clock_answer(text, now):
    """Přímá odpověď z hodin: „Co je za den?", „Kolik je hodin?".

    Chronos je tu sám odpovídací komponentou — odpověď nejde z grafu, ale
    z okamžiku `now`. Fráze otázek i jména dnů/měsíců jsou jazyková data
    (`temporal.clock_questions`, `weekday_names`, `month_genitives`).

    Args:
        text (str): Dotaz uživatele.
        now (datetime): Okamžik „teď" (zvenku — determinismus).

    Returns:
        str | None: Věta s odpovědí, nebo None (není hodinová otázka).
    """
    lang = current()["temporal"]
    questions = lang.get("clock_questions", {})
    low = deaccent(text.lower())
    if any(phrase in low for phrase in questions.get("day", ())):
        weekday = lang["weekday_names"][now.weekday()]
        month = lang["month_genitives"][now.month - 1]
        return f"Dnes je {weekday} {now.day}. {month} {now.year}."
    if any(phrase in low for phrase in questions.get("time", ())):
        return f"Je {now.hour}:{now.minute:02d}."
    return None


def resolve_temporal(text, now):  # pylint: disable=too-many-branches,too-many-return-statements
    """Najde v textu časové primitivum a ukotví ho na absolutní interval.

    Args:
        text (str): Dotaz uživatele (nebo jeho část).
        now (datetime): Okamžik „teď" — VŽDY zvenku (determinismus).

    Returns:
        TimeInterval | None: Interval primitiva, nebo None (text čas nenese).
    """
    lang = current()["temporal"]
    if not lang:
        return None
    tokens = [deaccent(t.lower()) for t in re.findall(r"[\w.]+", text)]
    units = lang.get("units", {})
    numerals = lang.get("numerals", {})

    # 1) století — absolutní interval; víc zmínek („18. nebo 19.") se sjednotí
    if any(units.get(t) == "century" for t in tokens):
        cents = [int(t.rstrip(".")) for t in tokens
                 if t.rstrip(".").isdigit() and 1 <= int(t.rstrip(".")) <= 21]
        if cents:
            return TimeInterval(datetime((min(cents) - 1) * 100 + 1, 1, 1),
                                datetime(max(cents) * 100 + 1, 1, 1),
                                "century")

    # 2) „teď/nyní" — bod v čase s oknem (default 15 min, symetricky kolem
    #    now): korelace na okamžik potřebuje toleranci, ne celý den
    if any(tok in frozenset(lang.get("now_words", ())) for tok in tokens):
        half = timedelta(minutes=lang.get("now_window_minutes", 15)) / 2
        return TimeInterval(now - half, now + half, "moment")

    # 3) slova dne: dnes/včera/zítra/předevčírem/pozítří
    for tok in tokens:
        if tok in lang.get("day_words", {}):
            day = _floor(now, "day") + timedelta(days=lang["day_words"][tok])
            return TimeInterval(day, day + timedelta(days=1), "day")

    # 3) směrovka ± N jednotka: „před dvěma hodinami", „za hodinu",
    #    „před týdnem" (= DEN před sedmi dny, ne celý týden)
    back = frozenset(lang.get("back_words", ()))
    forward = frozenset(lang.get("forward_words", ()))
    for i, tok in enumerate(tokens):
        if tok not in back and tok not in forward:
            continue
        count, unit = 1, None
        for follow in tokens[i + 1:i + 4]:
            if follow in numerals:
                count = numerals[follow]
            elif follow.rstrip(".").isdigit():
                count = int(follow.rstrip("."))
            elif follow in units and units[follow] != "century":
                unit = units[follow]
                break
        if unit is None:
            continue
        sign = -1 if tok in back else 1
        shifted = _shift(now, unit, sign * count)
        gran = "hour" if unit == "hour" else "day"
        start = _floor(shifted, gran)
        return TimeInterval(start, _shift(start, gran, 1), gran)

    # 4) „tento týden/měsíc/rok", „letos" — aktuální interval jednotky
    currents = frozenset(lang.get("current_words", ()))
    for i, tok in enumerate(tokens):
        if tok in ("letos", "letosni"):
            start = _floor(now, "year")
            return TimeInterval(start, _shift(start, "year", 1), "year")
        if tok in currents:
            unit = next((units[f] for f in tokens[i + 1:i + 3]
                         if f in units and units[f] != "century"), None)
            if unit:
                start = _floor(now, unit)
                return TimeInterval(start, _shift(start, unit, 1), unit)
    return None
