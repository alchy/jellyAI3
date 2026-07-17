"""Iris služba — REST rozhraní automatu zaostření (spec §5).

Dotaz běžným jazykem jde dovnitř (`POST /query`), ven jde odpověď
s **metadaty**: které komponenty a pattern-karty se na tahu podílely,
jistota zaostření (assurance), aktivační okno uzlů a aktivních dokumentů.
`POST /graphql` vykoná pseudo-QL pattern přímo (jazyk dotazu testovatelný
bez parseru), `POST /reset` začne nový rozhovor, `GET /schema` popíše,
na co se lze ptát: predikáty grafu, role, typy uzlů, tázací slova a karty.
"""

import os
import sys

# služba běží jako skript ze `services/` — kořen repa (o úroveň výš) musí
# na sys.path, aby šly importovat `config` a `jellyai`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import serve, parse_args

from jellyai.answerer.pattern import pattern_from_json, pattern_to_json
from jellyai.lang import current

_ROLES = ["subj", "obj", "loc", "time", "num", "pred", "attr", "theme", "val"]


def _fact_json(fact):
    """Faktový uzel → JSON: predikát + dvojice [role, uzel] za každého účastníka."""
    if fact is None:
        return None
    return {"predicate": fact.predicate,
            "participants": [[p.role, p.node] for p in fact.participants]}


def _trace_json(trace):
    """Trasa odpovědi JSON-safe: klíč faktu je tuple s Participanty (pickle
    identita, ne wire formát) — pro API se rozbalí na predikát + účastníky."""
    if trace is None:
        return None
    out = dict(trace)
    fact = out.get("fact")
    if isinstance(fact, tuple) and len(fact) == 2:
        predicate, participants = fact
        out["fact"] = {"predicate": predicate,
                       "participants": [[p.role, p.node] for p in participants]}
    return out


def make_routes(automaton, deck):
    """Vytvoří routy služby navázané na jeden automat (jednu konverzaci).

    Args:
        automaton (IrisAutomaton): Automat zaostření nad answererem.
        deck (PatternDeck): Balíček pattern-karet (pro popis ve `/schema`).

    Returns:
        tuple[dict, dict]: (POST routy: cesta → funkce(payload) → dict,
            GET routy: cesta → funkce() → dict).
    """
    def query(payload):
        """`/query` {question, temperature?} → odpověď/dialog + metadata tahu."""
        r = automaton.turn(payload["question"],
                           temperature=payload.get("temperature", 0.0))
        # VŠECHNA API volání se zrcadlí do konzole služby — i externí
        # (testy, curl) jsou vidět: otázka, odpověď, jistota, komponenty
        top = ", ".join(f"{n}={j:.2f}" for n, j in r.activation_window[:5])
        print(f"\n❓ {payload['question']}"
              f"\n💬 [{r.kind}, assurance {r.assurance:.2f}] {r.text}"
              f"\n   komponenty: {', '.join(r.used.get('components', []))}"
              f"   karty: {', '.join(r.used.get('patterns', [])) or '—'}"
              f"\n   aktivace: {top or '—'}", flush=True)
        return {"answer": r.text, "kind": r.kind, "assurance": r.assurance,
                "clarify": r.clarify, "trace": _trace_json(r.trace),
                "sources": r.sources, "alternatives": r.alternatives,
                "pattern": pattern_to_json(automaton.answerer.last_pattern),
                "activation": {"nodes": r.activation_window,
                               "docs": r.docs_window},
                "used": r.used}

    def graphql(payload):
        """`/graphql` {predicate, known, hole…} → přímé vykonání patternu."""
        pat = pattern_from_json(payload)
        topic, values, fact = automaton.answerer.run_pattern(pat)
        return {"answer": ", ".join(values), "values": values, "topic": topic,
                "fact": _fact_json(fact)}

    def reset(payload):
        """`/reset` {} → nový rozhovor (vymaže dialog i aktivační pole)."""
        automaton.reset()
        return {"status": "ok"}

    def schema():
        """`/schema` → na co se lze ptát: predikáty, role, typy, díry, karty."""
        graph = automaton.answerer.graph
        holes = {form: list(spec[:2]) for form, spec
                 in sorted(current()["interrogatives"].items())}
        return {"predicates": sorted({f.predicate for f in graph.facts.values()}),
                "roles": _ROLES,
                "node_types": sorted({n.type for n in graph.nodes.values()
                                      if n.type}),
                "holes": holes,
                "patterns": [{"name": card.name, "teach": card.teach}
                             for card in deck.cards]}

    posts = {"/query": query, "/graphql": graphql, "/reset": reset}
    gets = {"/schema": schema}
    return posts, gets


def main():
    args = parse_args()
    from config import Config
    from jellyai.iris import IrisAutomaton, PatternDeck
    from jellyai.tasks import make_graph_answerer
    config = Config()
    config.graph.graph_path = args.model     # --model = uložený faktový graf
    answerer = make_graph_answerer(config)
    deck = PatternDeck.for_language(config.graph.language)
    deck.load()
    automaton = IrisAutomaton(answerer, deck)   # prahy nesou karty (ZÁKON)
    posts, gets = make_routes(automaton, deck)
    serve(args.host, args.port, posts, gets)


if __name__ == "__main__":
    main()
