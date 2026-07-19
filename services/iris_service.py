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
                "used": r.used,
                "memorized": r.memorized,    # nová vzpomínka → do vizualizace
                "forgotten": r.forgotten}    # zapomenutá entita → z vizualizace

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

    from datetime import datetime
    from jellyai.buildinfo import git_sha
    started = datetime.now().isoformat(timespec="seconds")
    sha = git_sha()

    def version():
        """`/version` → git SHA + čas startu (verzovací handshake, #40):
        web při připojení pozná, jestli mluví na aktuální build."""
        return {"sha": sha, "started": started}

    posts = {"/query": query, "/graphql": graphql, "/reset": reset}
    gets = {"/schema": schema, "/version": version}
    return posts, gets


def _web_event(host, port, event, message):
    """Push eventu do webu (viewBase REST most `/api/event`).

    Iniciátorem je CHRONOS — web je pasivní displej; když neběží, push
    tiše selže (konzole služby připomínku nese vždy).
    """
    import json as _json
    import urllib.request
    body = _json.dumps({"event": event,
                        "payload": {"window_id": "konzole",
                                    "line": message}}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}/api/event", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=2):
        pass


def _channels(config):
    """REGISTR KANÁLŮ připomínek (spec §4.3 — modularita pro integrace).

    Kanál = jméno → funkce(message). Dnes `console` (konzole služby +
    řádek webové konzole) a `window` (statické okno ⏰ Reminder visící
    do zavření). Budoucí `alarm-audio`/`email`/`whatsapp` se sem jen
    PŘIDAJÍ — jádro se nemění; volbu kanálu ponesou karty.
    """
    host, port = config.services.host, config.services.web_port

    def console(message):
        print(f"\n{message}", flush=True)
        _web_event(host, port, "terminal_write", message)

    def window(message):
        _web_event(host, port, "reminder_window", message)

    channels = {"console": console, "window": window}

    # E-MAILOVÝ KANÁL (rozšiřovací bod dle spec) — pošle text připomínky mailem
    # přes lokální Postfix (127.0.0.1:25). Aktivní jen když je nastaven adresát
    # v env `JELLY_REMINDER_EMAIL`. Odesílatel `JELLY_REMINDER_FROM` MUSÍ být
    # @lordaudio.eu (OpenDKIM podepisuje *@lordaudio.eu, SPF alignment) kvůli
    # doručitelnosti. Selhání SMTP notify obalí try/except → hodiny nespadnou.
    default_recipient = os.environ.get("JELLY_REMINDER_EMAIL")
    sender = os.environ.get("JELLY_REMINDER_FROM", "jelly@lordaudio.eu")

    def email(message):
        # ADRESÁT: z připomínky (ReminderMessage.recipient — „pošli Jindrovi…"),
        # jinak DEFAULT z env. Bez obojího se nic neposílá.
        to = getattr(message, "recipient", None) or default_recipient
        if not to:
            return
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = "Připomínka — jellyAI3 (Iris)"
        msg.set_content(str(message))
        with smtplib.SMTP("127.0.0.1", 25, timeout=15) as smtp:
            smtp.send_message(msg)
        print(f"[email] připomínka odeslána na {to}", flush=True)

    channels["email"] = email
    return channels


def main():
    args = parse_args()
    from config import Config
    from jellyai.iris import IrisAutomaton, PatternDeck
    from jellyai.iris.subsystems.chronos import ChronosTicker
    from jellyai.tasks import make_graph_answerer
    config = Config()
    config.graph.graph_path = args.model     # --model = uložený faktový graf
    from jellyai.buildinfo import git_sha
    print(f"[iris] build {git_sha()} — model {args.model}", flush=True)
    answerer = make_graph_answerer(config)
    deck = PatternDeck.for_language(config.graph.language)
    deck.load()
    automaton = IrisAutomaton(answerer, deck,   # prahy nesou karty (ZÁKON)
                              memory_path=config.graph.memory_path,
                              reminders_path=config.graph.reminders_path)

    channels = _channels(config)

    def notify(message):
        for send in channels.values():   # v1: všechny kanály; volbu
            try:                         # per připomínka ponesou karty
                send(message)
            except Exception:  # noqa: BLE001 — web nemusí běžet
                pass

    ChronosTicker(automaton.fire_due, notify).start()   # vlastní vlákno hodin
    posts, gets = make_routes(automaton, deck)
    serve(args.host, args.port, posts, gets)


if __name__ == "__main__":
    main()
