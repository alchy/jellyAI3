"""Testy pulzující trasy — vizualizace zrcadlí aktivační pole (context window).

Per-dotaz model: hustota provozu ∝ aktivaci trasy, mezi dotazy ustálená; další
dotaz pole posune a trasy s vyhaslými uzly zmizí. Logika decay je v answereru —
sem chodí hotové `scores`.
"""

from collections import Counter

from jellyai.viz.pulse import TracePulse


def test_ignite_reflects_field():
    p = TracePulse()
    state = p.ignite({"Božena Němcová": 1.1, "Babička": 0.55},
                     ["Babička", "Božena Němcová"])
    assert state["sizes"] == {"Božena Němcová": 1.1, "Babička": 0.55}
    assert state["extinguish"] == []


def test_reignite_extinguishes_nodes_gone_from_field():
    p = TracePulse()
    p.ignite({"a": 1.0, "b": 0.5}, ["a", "b"])
    state = p.ignite({"c": 1.0}, ["c", "d"])       # a, b vypadly z pole
    assert set(state["sizes"]) == {"c"}
    assert "a" in state["extinguish"] and "b" in state["extinguish"]


def test_hotter_trace_has_denser_traffic():
    p = TracePulse()
    p.ignite({"a": 0.4, "b": 0.4}, ["a", "b"])                 # slabá trasa
    p.ignite({"a": 0.4, "b": 0.4, "c": 2.0, "d": 2.0}, ["c", "d"])   # silná; slabá žije dál
    counts = Counter()
    for _ in range(30):
        for pkt in p.tick()["packets"]:
            counts[tuple(pkt)] += 1
    assert counts[("c", "d")] > counts[("a", "b")]            # teplejší = hustší


def test_trace_drops_when_its_nodes_leave_field():
    p = TracePulse()
    p.ignite({"a": 1.0, "b": 1.0}, ["a", "b"])
    p.ignite({"c": 1.0, "d": 1.0}, ["c", "d"])     # a, b mimo pole → trasa umlkne
    seen = set()
    for _ in range(30):
        for pkt in p.tick()["packets"]:
            seen.add(tuple(pkt))
    assert ("a", "b") not in seen
    assert ("c", "d") in seen


def test_traffic_is_steady_between_queries():
    """Bez nového dotazu se hustota nemění (žádný wall-clock časovač)."""
    p = TracePulse()
    p.ignite({"a": 1.0, "b": 1.0}, ["a", "b"])
    first = sum(len(p.tick()["packets"]) for _ in range(30))
    second = sum(len(p.tick()["packets"]) for _ in range(30))
    assert first == second and first > 0


def test_no_trace_no_packets():
    p = TracePulse()
    p.ignite({"a": 2.0}, [])                        # bez trasy
    assert all(p.tick()["packets"] == [] for _ in range(20))
