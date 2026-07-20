"""Shadow měření otázkového grafu (#57, experimentální větev).

`.venv/bin/python benchmark/run_qgraph.py [--variant weights]`

Nic nepřepíná — MĚŘÍ, jak by dispatch osvětlením (qgraph.illuminate)
souhlasil se skutečnou cestou automatu:

- DIALOG benchmark: každý tah se přehraje živým automatem; skutečná
  cesta se čte z metadat odpovědi (komponenty + karty). Shadow verdikt
  = vítěz osvětlení; u clarify tahů platí shoda, když vítězná OTÁZKA
  má zpřesňovací hranu na skutečně vystřelený clarify uzel (graf by
  stál v uzlu otázky a sám si ostřil focus — zadání user).
- ETALON: shoda vítězného uzlu otázky s kartou, kterou reálně vybral
  build_query (answerer.turn.query_card); „oba bez karty" je shoda
  (poziční šablona je dnešní fallback, graf ji nezná záměrně).

Varianta `--variant weights` láme remízy vahami z telemetrie
(data/telemetry.jsonl) — test otázky „kolik provozu je potřeba".
Tahy mimo rozsah (výroky, příkazy, volby kandidátů) se počítají
jako pokrytí, ne shoda.
"""
import argparse
import json
import os

from datetime import datetime

from config import Config
from jellyai.iris import IrisAutomaton
from jellyai.iris.qgraph import (DialogPosition, compile_qgraph, decorate,
                                 illuminate)
from jellyai.lang import current
from jellyai.iris.triage import load_rows
from jellyai.tasks import make_graph_answerer

DIALOG = os.path.join(os.path.dirname(__file__), "dialog.jsonl")
ETALON = os.path.join(os.path.dirname(__file__), "etalon.jsonl")


def _now():
    return datetime(2026, 7, 17, 12, 0)


from jellyai.iris.claims import default_claims

_WORKER_NODES = {c.worker: c.name for c in default_claims()}


def _actual_route(response, qgraph):
    """Skutečná cesta tahu z metadat odpovědi → jméno uzlu / None.
    Jména worker uzlů z registru claimů (postřeh 4.5), ne literály."""
    components = response.used.get("components", ())
    patterns = response.used.get("patterns", ())
    if "metron" in components:
        return _WORKER_NODES["metron"]
    if "iris" in components:
        return _WORKER_NODES["iris"]
    for name in patterns:
        node = qgraph.nodes.get(name)
        if node is not None and node.kind in ("otazka", "clarify", "vyrok"):
            return name
    if list(components) == ["chronos"] and not patterns:
        return _WORKER_NODES["chronos"]
    return None                                  # mimo rozsah experimentu


def _actual_decorations(answerer):
    """Dekorace, které tah SKUTEČNĚ aplikoval (stav answereru po tahu)."""
    found = set()
    if answerer.time_filter is not None:
        found.add("chronos:interval")
    if answerer.place_filter is not None:
        found.add("topos:oblast")
    if answerer.turn.theme_bound:
        found.add("role:adresat")
    pattern = answerer.turn.pattern
    if pattern is not None:
        if any(term == current()["user_entity"]
               for _, term in getattr(pattern, "known", ())):
            found.add("mnemos:prvni-osoba")
        if pattern.hole_role == "relation":
            found.add("vztah:operator")
    return found


def _base_key(node):
    """Klíč BEZ vah — remíza dvou uzlů = místo, kde by váhy rozhodly."""
    return (node.kind, node.priority,
            len(node.pattern) if node.pattern else 0)


def _agrees(shadow, actual, qgraph):
    """Shoda shadow verdiktu se skutečnou cestou (vč. clarify hran)."""
    if not shadow:
        return actual is None
    winner = shadow[0]
    if winner.name == actual:
        return True
    actual_node = qgraph.nodes.get(actual) if actual else None
    if actual_node is not None and actual_node.kind == "clarify":
        # graf by stál ve vítězné otázce; zpřesnění je JEJÍ hrana
        return any(e.kind == "zpresneni" and e.target == actual
                   for e in winner.edges)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("tiers", "weights"),
                        default="tiers")
    args = parser.parse_args()
    config = Config()
    answerer = make_graph_answerer(config)
    telemetry = (load_rows(config.graph.telemetry_path)
                 if args.variant == "weights" else ())
    deck_iris = IrisAutomaton(answerer, clock=_now)
    qgraph = compile_qgraph(deck_iris.deck, answerer._predicates,  # pylint: disable=protected-access
                            telemetry_rows=telemetry)
    use_weights = args.variant == "weights"

    agree = miss = skipped = ties = 0
    state_agree = state_miss = 0
    vyrok_agree = vyrok_miss = 0
    deco_agree = deco_miss = 0
    disagreements = []
    with open(DIALOG, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    for scenario in scenarios:
        iris = IrisAutomaton(make_graph_answerer(config), clock=_now)
        position = DialogPosition(qgraph)        # stav dialogu = pozice
        for turn in scenario["turns"]:
            text = turn["u"]
            shadow = illuminate(text, qgraph, now=_now(),
                                is_node=iris.answerer._span_is_node,  # pylint: disable=protected-access
                                use_weights=use_weights)
            standing_in_clarify = (position.node is not None
                                   and position.node.kind == "clarify")
            response = iris.turn(text)
            actual = _actual_route(response, qgraph)
            actual_node = qgraph.nodes.get(actual) if actual else None
            if standing_in_clarify and "?" not in text:
                # STAVOVÝ tah: stojíme ve zpřesnění, tah je krok po hraně
                # `navrat` (volba kandidáta / doplnění identity) — graf by
                # přehrál otázku, ne směroval text
                resumed = position.resume()
                if resumed == "*" and actual is not None:
                    state_agree += 1
                else:
                    state_miss += 1
                    disagreements.append((text, actual or "—", "navrat"))
                continue
            if actual_node is not None and actual_node.kind == "vyrok":
                # ROVINA VÝROKY (#51 fáze 1): shoda vítěze osvětlení se
                # skutečně vybranou výrokovou kartou (brána E)
                if shadow and shadow[0].name == actual:
                    vyrok_agree += 1
                else:
                    vyrok_miss += 1
                    disagreements.append(
                        (text, actual, shadow[0].name if shadow else "—"))
                continue
            if actual is None:
                skipped += 1                 # mimo rozsah (příkazy aj.)
                continue
            if "?" not in text and actual_node.kind == "clarify":
                # VÝROKOVÁ clarify (identita podmětu #43): graf stojí ve
                # výroku a ostří — pozice, ne dispatch (#51 spec §3)
                if shadow:
                    position.enter(shadow[0].name)
                position.sharpen(actual)
                skipped += 1
                continue
            if _agrees(shadow, actual, qgraph):
                agree += 1
            else:
                miss += 1
                shadow_name = shadow[0].name if shadow else "—"
                disagreements.append((text, actual, shadow_name))
            if len(shadow) > 1 and _base_key(shadow[0]) == _base_key(shadow[1]):
                # REMÍZA základního klíče — jediné místo, kde by váhy
                # z telemetrie mohly rozhodnout (test „kolik provozu")
                ties += 1
            # posun pozice v grafu podle skutečné cesty tahu
            if actual_node is not None and actual_node.kind == "clarify":
                if shadow:
                    position.enter(shadow[0].name)
                position.sharpen(actual)
            elif actual_node is not None:
                position.enter(actual)

    et_agree = et_miss = 0
    with open(ETALON, encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    for item in items:
        questions = item.get("dialog") or [item["q"]]
        answerer.reset()
        for question in questions:
            shadow = illuminate(question, qgraph, now=_now(),
                                is_node=answerer._span_is_node,  # pylint: disable=protected-access
                                use_weights=use_weights)
            answerer.answer(question, [])
            actual = answerer.turn.query_card
            # DEKORACE (T3): nároky se měří tam, kde answerer opravdu
            # běžel — shadow předpověď vs. skutečně aplikované filtry
            want = decorate(question, now=_now())
            got = _actual_decorations(answerer)
            if want == got:
                deco_agree += 1
            else:
                deco_miss += 1
                disagreements.append(
                    (question, f"deco {sorted(got) or '—'}",
                     f"{sorted(want) or '—'}"))
            shadow_card = next((n.name for n in shadow
                                if n.kind == "otazka"), None)
            if shadow_card == actual:
                et_agree += 1
            else:
                et_miss += 1
                disagreements.append((question, actual or "šablona",
                                      shadow_card or "—"))

    for text, actual, shadow_name in disagreements:
        print(f"[NESHODA] {text[:44]:44} skutečně: {actual:24} "
              f"shadow: {shadow_name}")
    total = agree + miss
    et_total = et_agree + et_miss
    pct = 100 * agree // total if total else 0
    et_pct = 100 * et_agree // et_total if et_total else 0
    state_total = state_agree + state_miss
    deco_total = deco_agree + deco_miss
    state_pct = 100 * state_agree // state_total if state_total else 0
    deco_pct = 100 * deco_agree // deco_total if deco_total else 0
    vyrok_total = vyrok_agree + vyrok_miss
    vyrok_pct = 100 * vyrok_agree // vyrok_total if vyrok_total else 0
    print(f"\nQGRAPH SHADOW [{args.variant}]: dialog {agree}/{total} "
          f"({pct} %), výroky {vyrok_agree}/{vyrok_total} ({vyrok_pct} %), "
          f"stav {state_agree}/{state_total} ({state_pct} %), "
          f"dekorace {deco_agree}/{deco_total} ({deco_pct} %), "
          f"mimo rozsah {skipped}, remíz {ties}   "
          f"etalon {et_agree}/{et_total} ({et_pct} %)")
    return agree, miss, et_agree, et_miss


if __name__ == "__main__":
    main()
