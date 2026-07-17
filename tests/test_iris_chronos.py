"""Chronos — orientace Iris v čase: primitiva → absolutní intervaly.

„Teď" je VŽDY parametr (determinismus: testy ho fixují, živé API dodá
hodiny). Interval je půlotevřený ⟨start, end) a nese granularitu; graf se
napojuje přes `contains_date` nad výstupem `parse_date` (překryv intervalů —
částečné datum „1818" má vlastní roční interval).
"""

from datetime import datetime

from jellyai.iris.chronos import TimeInterval, resolve_temporal

NOW = datetime(2026, 7, 17, 12, 0)      # pátek; týden začíná po 2026-07-13


def test_today_yesterday_tomorrow():
    dnes = resolve_temporal("Co se stalo dnes?", NOW)
    assert dnes.start == datetime(2026, 7, 17)
    assert dnes.end == datetime(2026, 7, 18)
    assert dnes.granularity == "day"
    assert resolve_temporal("vcera", NOW).start == datetime(2026, 7, 16)
    assert resolve_temporal("Zítra?", NOW).start == datetime(2026, 7, 18)


def test_in_an_hour_and_hours_ago():
    za = resolve_temporal("Co bude za hodinu?", NOW)
    assert (za.start, za.end) == (datetime(2026, 7, 17, 13),
                                  datetime(2026, 7, 17, 14))
    assert za.granularity == "hour"
    pred = resolve_temporal("Co se stalo před dvěma hodinami?", NOW)
    assert (pred.start, pred.end) == (datetime(2026, 7, 17, 10),
                                      datetime(2026, 7, 17, 11))


def test_week_ago_is_a_day():
    """„Před týdnem" míní DEN před sedmi dny, ne celý týden."""
    pred = resolve_temporal("před týdnem", NOW)
    assert (pred.start, pred.end) == (datetime(2026, 7, 10),
                                      datetime(2026, 7, 11))
    assert pred.granularity == "day"


def test_this_week_and_month_intervals():
    tyden = resolve_temporal("Co se stalo tento týden?", NOW)
    assert (tyden.start, tyden.end) == (datetime(2026, 7, 13),
                                        datetime(2026, 7, 20))
    assert tyden.granularity == "week"
    mesic = resolve_temporal("tento měsíc", NOW)
    assert (mesic.start, mesic.end) == (datetime(2026, 7, 1),
                                        datetime(2026, 8, 1))


def test_now_word_gets_moment_window():
    """„Teď/nyní" = bod v čase s oknem (15 min symetricky kolem now)."""
    from datetime import timedelta
    ted = resolve_temporal("Co se děje teď?", NOW)
    assert ted.granularity == "moment"
    assert (ted.start, ted.end) == (NOW - timedelta(minutes=7.5),
                                    NOW + timedelta(minutes=7.5))
    assert ted.contains(NOW)
    assert resolve_temporal("nyní", NOW) == ted


def test_no_temporal_primitive_returns_none():
    assert resolve_temporal("Kdo napsal Babičku?", NOW) is None


def test_clock_answers_day_and_time():
    """Hodinové otázky odpovídá Chronos přímo z „teď" (2026-07-17 je pátek)."""
    from jellyai.iris.chronos import clock_answer
    den = clock_answer("Co je za den?", NOW)
    assert den == "Dnes je pátek 17. července 2026."
    assert clock_answer("Kolik je hodin?", NOW) == "Je 12:00."
    assert clock_answer("Kdo napsal Babičku?", NOW) is None


def test_automaton_routes_clock_question_to_chronos():
    """Automat s injektovaným clock: hodinová otázka jde Chronosu, ne grafu."""
    from tests.test_iris_automaton import _brothers_graph, _iris
    iris = _iris(_brothers_graph())
    iris.clock = lambda: NOW
    out = iris.turn("Kolik je hodin?")
    assert out.kind == "answer" and out.text == "Je 12:00."
    assert out.used["components"] == ["chronos"]


def test_contains_date_over_parse_date_output():
    """Napojení na graf: časový uzel (rok/měsíc/den z parse_date) spadá do
    intervalu překryvem — částečné datum má vlastní interval své granularity."""
    dnes = resolve_temporal("dnes", NOW)
    assert dnes.contains_date({"rok": "2026", "měsíc": "červenec", "den": "17"})
    assert not dnes.contains_date({"rok": "2026", "měsíc": "červenec", "den": "16"})
    mesic = resolve_temporal("tento měsíc", NOW)
    assert mesic.contains_date({"rok": "2026", "měsíc": "červenec"})
    assert not mesic.contains_date({"rok": "1818"})
    assert mesic.contains_date({"rok": "2026"})   # roční interval se překrývá
    assert not mesic.contains_date({})            # neukotvené datum nesvítí


def test_deterministic():
    a = resolve_temporal("před třemi dny", NOW)
    b = resolve_temporal("před třemi dny", NOW)
    assert a == b and a.start == datetime(2026, 7, 14)


def test_century_interval():
    """„v 19. století" → ⟨1801, 1901); rok 1900 do 19. století PATŘÍ —
    chronos datum interpretuje, nikam ho nepřevádí."""
    stol = resolve_temporal("Chodila kovářova kobyla bosa v 19. století?", NOW)
    assert (stol.start, stol.end) == (datetime(1801, 1, 1), datetime(1901, 1, 1))
    assert stol.contains_date({"rok": "1900"})
    assert not stol.contains_date({"rok": "1901"})


def test_century_alternatives_span():
    """„v 18. nebo 19. století" → sjednocený interval obou století."""
    stol = resolve_temporal("v 18. nebo 19. století", NOW)
    assert (stol.start, stol.end) == (datetime(1701, 1, 1), datetime(1901, 1, 1))
    assert stol.contains_date({"rok": "1900"})
    assert stol.contains_date({"rok": "1750"})


def test_numeric_date_from_graph_matches_interval():
    """Uzel „21.1.1900" (numerické datum korpusu) se přes parse_date trefí
    do intervalu 19. století — scénář kovářovy kobyly."""
    from jellyai.graph.graph import parse_date
    parsed = parse_date("21.1.1900")
    assert parsed.get("rok") == "1900" and parsed.get("den") == "21"
    stol = resolve_temporal("v 19. století", NOW)
    assert stol.contains_date(parsed)
