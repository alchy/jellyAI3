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
  build_query (answerer.last_query_card); „oba bez karty" je shoda
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
from jellyai.iris.qgraph import compile_qgraph, illuminate
from jellyai.iris.triage import load_rows
from jellyai.tasks import make_graph_answerer

DIALOG = os.path.join(os.path.dirname(__file__), "dialog.jsonl")
ETALON = os.path.join(os.path.dirname(__file__), "etalon.jsonl")


def _now():
    return datetime(2026, 7, 17, 12, 0)


def _actual_route(response, qgraph):
    """Skutečná cesta tahu z metadat odpovědi → jméno uzlu / None."""
    components = response.used.get("components", ())
    patterns = response.used.get("patterns", ())
    if "metron" in components:
        return "metron-vypocet"
    if "iris" in components:
        return "meta-focus"
    for name in patterns:
        node = qgraph.nodes.get(name)
        if node is not None and node.kind in ("otazka", "clarify"):
            return name
    if list(components) == ["chronos"] and not patterns:
        return "chronos-hodiny"
    return None                                  # mimo rozsah experimentu


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

    agree = miss = skipped = 0
    disagreements = []
    with open(DIALOG, encoding="utf-8") as fh:
        scenarios = [json.loads(line) for line in fh if line.strip()]
    for scenario in scenarios:
        iris = IrisAutomaton(make_graph_answerer(config), clock=_now)
        for turn in scenario["turns"]:
            text = turn["u"]
            shadow = illuminate(text, qgraph, now=_now(),
                                is_node=iris.answerer._span_is_node,  # pylint: disable=protected-access
                                use_weights=use_weights)
            response = iris.turn(text)
            actual = _actual_route(response, qgraph)
            if actual is None:
                skipped += 1
                continue
            actual_node = qgraph.nodes.get(actual)
            if actual_node is not None \
                    and actual_node.kind in ("otazka", "clarify") \
                    and "?" not in text:
                # STAVOVÝ tah (volba kandidáta, výrok s clarify-identity):
                # krok po hraně `navrat`, ne textové směrování — pozici
                # v grafu shadow nezná (měří jen dotazovou polovinu)
                skipped += 1
                continue
            if _agrees(shadow, actual, qgraph):
                agree += 1
            else:
                miss += 1
                shadow_name = shadow[0].name if shadow else "—"
                disagreements.append((text, actual, shadow_name))

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
            actual = answerer.last_query_card
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
    print(f"\nQGRAPH SHADOW [{args.variant}]: dialog {agree}/{total} "
          f"({pct} %), mimo rozsah {skipped}   "
          f"etalon {et_agree}/{et_total} ({et_pct} %)")
    return agree, miss, et_agree, et_miss


if __name__ == "__main__":
    main()
