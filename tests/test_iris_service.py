"""REST služba Iris — testy rout (in-process) + integrace přes HTTP.

Routy se testují bez sítě: `make_routes` vrátí obyčejné funkce a test jim
předá payload dict — přesně to, co by jinak přišlo v těle JSON POSTu.
Jediný integrační test spustí službu jako subprocess a ověří celou cestu:
argumenty → načtení grafu z disku → HTTP dotaz → JSON odpověď s metadaty.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.extract import Participant, make_fact
from jellyai.graph.graph import FactGraph
from jellyai.iris import IrisAutomaton, PatternDeck
from jellyai.ufal_client import FakeUfalClient

sys.path.insert(0, "services")
from iris_service import make_routes


def _brothers_graph():
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Karel Čapek", "person"),
                                 Participant("pred", "spisovatel", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Josef Čapek", "person"),
                                 Participant("pred", "malíř", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    return g


def _routes():
    answerer = GraphAnswerer(_brothers_graph(), FakeUfalClient(),
                             ExtractiveAnswerer(AnswererConfig()))
    deck = PatternDeck.for_language("cs")
    deck.load()
    automaton = IrisAutomaton(answerer, deck)
    return make_routes(automaton, deck)


def test_query_answers_with_metadata():
    """/query s jistou otázkou → odpověď + pattern + seřazené aktivační okno."""
    posts, _ = _routes()
    out = posts["/query"]({"question": "Kdo napsal R.U.R.?"})
    assert "Karel" in out["answer"] and out["kind"] == "answer"
    assert out["pattern"]["predicate"] == "napsat"
    nodes = out["activation"]["nodes"]
    assert nodes
    jas = [score for _, score in nodes]
    assert jas == sorted(jas, reverse=True)      # okno seřazené sestupně


def test_query_dialog_carries_clarify():
    """/query s homonymem → dialog: clarify nese oba kandidáty + jméno karty."""
    posts, _ = _routes()
    out = posts["/query"]({"question": "Kdo je Čapek?"})
    assert out["kind"] == "dialog"
    assert "Karel Čapek" in out["clarify"]["candidates"]
    assert "Josef Čapek" in out["clarify"]["candidates"]
    assert "focus-offer-homonym" in out["used"]["patterns"]


def test_graphql_runs_raw_pattern():
    """/graphql vykoná pseudo-QL pattern přímo — jazyk testovatelný bez parseru."""
    posts, _ = _routes()
    out = posts["/graphql"]({"predicate": "napsat",
                             "known": [["obj", "R.U.R."]], "hole": "subj"})
    assert out["answer"] == "Karel Čapek"
    assert out["values"] == ["Karel Čapek"]
    assert out["fact"]["predicate"] == "napsat"


def test_schema_describes_graph_and_deck():
    """GET /schema popíše predikáty grafu, tázací díry jazyka i pattern-karty."""
    _, gets = _routes()
    schema = gets["/schema"]()
    assert "napsat" in schema["predicates"]
    assert "kdo" in schema["holes"]
    assert any(card["name"] == "focus-offer-homonym"
               for card in schema["patterns"])


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(port, proc, timeout=30.0):
    """Čeká na /health služby; spadlý proces nebo timeout = chyba testu."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert proc.poll() is None, f"služba spadla (exit {proc.returncode})"
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1):
                return
        except Exception:  # noqa: BLE001 - dokud nenaběhne, zkoušíme dál
            time.sleep(0.2)
    raise AssertionError(f"služba na portu {port} nenaběhla do {timeout}s")


def test_service_subprocess_answers_over_http(tmp_path):
    """Integrace: služba jako subprocess odpoví na /query přes skutečné HTTP."""
    model = _brothers_graph().save(str(tmp_path / "graph.pkl"))
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "services/iris_service.py",
                             "--port", str(port), "--model", model,
                             "--telemetry", "off"])
    try:
        _wait_health(port, proc)
        data = json.dumps({"question": "Kdo napsal R.U.R.?"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/query", data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            out = json.load(resp)
        assert "Karel" in out["answer"]
    finally:
        proc.terminate()


def test_version_route_reports_sha_and_start():
    """#40 verzovací handshake: /version nese git SHA a čas startu —
    web při připojení pozná, na jakou instanci mluví."""
    _, gets = _routes()
    out = gets["/version"]()
    assert out["sha"] and out["started"]


def test_version_warning_only_on_real_mismatch():
    """Křičí se jen při skutečné neshodě dvou známých verzí — „unknown"
    (mimo git, např. nasazený tarball) poctivě nesoudí."""
    from jellyai.iris.client import version_warning

    assert version_warning("abc1234", "abc1234") is None
    assert version_warning("unknown", "abc1234") is None
    assert version_warning("abc1234", "unknown") is None
    warning = version_warning("abc1234", "def5678")
    assert warning is not None
    assert "abc1234" in warning and "def5678" in warning
