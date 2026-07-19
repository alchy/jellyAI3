import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jellyai.ufal_client import _post, _ServiceHandle, FakeUfalClient


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_fake_ufal_client():
    client = FakeUfalClient(
        entities={"věta": [{"text": "Rossum", "type": "P", "start": 0, "end": 6}]},
        generate={("Praha", "loc"): ["Praze"], "Rossum": ["Rossum"]})
    assert client.entities("věta")[0]["text"] == "Rossum"
    assert client.entities("neznámá") == []
    assert client.generate("Praha", "loc") == ["Praze"]
    assert client.generate("Rossum", "cokoliv") == ["Rossum"]   # fallback dle lemmatu
    assert client.generate("neznámé", "x") == ["neznámé"]        # default [lemma]


def test_post_roundtrip():
    port = _free_port()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(n))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"echo": body}).encode("utf-8"))

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert _post("127.0.0.1", port, "/x", {"a": 1}) == {"echo": {"a": 1}}
    finally:
        server.shutdown()


def test_service_handle_spawns_and_health(tmp_path):
    script = tmp_path / "svc.py"
    script.write_text(
        "import argparse, http.server\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--port', type=int); p.add_argument('--model')\n"
        "a = p.parse_args()\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "    def log_message(self, *x): pass\n"
        "http.server.HTTPServer(('127.0.0.1', a.port), H).serve_forever()\n",
        encoding="utf-8")
    port = _free_port()
    handle = _ServiceHandle(str(script), "model", "127.0.0.1", port, timeout=10)
    try:
        assert handle.proc.poll() is None    # služba běží
    finally:
        handle.close()
    assert handle.proc.poll() is not None     # po close ukončena
