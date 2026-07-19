"""Sdílený HTTP skelet pro ÚFAL služby.

Každá služba je jen tenký JSON-over-HTTP server na localhostu: `/health` (GET) pro
kontrolu, že model naběhl, a pár POST endpointů. Tady je společný kód, ať se
v každé službě neopakuje boilerplate. Služby se spouštějí jako skripty, takže
importují tenhle modul ze stejného adresáře (`from _common import …`).
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _make_handler(routes, gets=None):
    """Vytvoří HTTP handler, který cesty směruje na funkce z `routes`/`gets`.

    Args:
        routes (dict): POST cesta → funkce(payload_dict) → dict.
        gets (dict | None): GET cesta → funkce() → dict (bez payloadu);
            `/health` odpovídá vždy, i bez `gets`.

    Returns:
        type: Třída handleru pro HTTPServer.
    """
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            fn = (gets or {}).get(self.path)
            if fn is not None:
                try:
                    self._send(200, fn())
                except Exception as exc:  # noqa: BLE001 - chybu vrátíme klientovi
                    self._send(500, {"error": str(exc)})
            elif self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            fn = routes.get(self.path)
            if fn is None:
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length)) if length else {}
            try:
                self._send(200, fn(payload))
            except Exception as exc:  # noqa: BLE001 - chybu vrátíme klientovi, nespadneme
                self._send(500, {"error": str(exc)})

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # ticho — služba nemá spamovat stderr

    return Handler


def serve(host, port, routes, gets=None):
    """Spustí HTTP službu s danými cestami (blokující).

    Args:
        host (str): Adresa (jen localhost).
        port (int): Port.
        routes (dict): POST cesta → funkce(payload) → dict.
        gets (dict | None): GET cesta → funkce() → dict (bez payloadu).
    """
    ThreadingHTTPServer((host, port),
                        _make_handler(routes, gets)).serve_forever()


def parse_args():
    """Rozparsuje společné argumenty služby (--port, --model, --host,
    --telemetry — cesta stopy tahů #38; „off" vypne, default z configu).

    Returns:
        argparse.Namespace: Argumenty.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--telemetry", default=None,
                        help="stopa tahů JSONL (off = bez stopy)")
    return parser.parse_args()
