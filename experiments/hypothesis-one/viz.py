#!/usr/bin/env python3
"""Živá vizualizace dialogu — graf se rozsvěcí, jak se ptáš (viewBase na :8080).

Napojeno na ŽIVÝ systém (Answering) — ne statický snímek. Každý dotaz:
  • DYNAMICKÝ LOADER: namountuje horké soubory → staré uzly zmizí, nové přibudou;
  • slova otázky rozsvítí graf, světlo teče po VODIVÝCH hranách (idf×proximity);
  • VZOR + assurance doostří odpověď → animace světla od dotazu k odpovědi (flow);
  • barva uzlu-odpovědi: zelená = odpověď · oranžová = doptání · šedá = nevím.
Graf DÝCHÁ s rozhovorem: jas = živá aktivace (ne statické tf·idf).

Spuštění (NUTNÝ .venv kvůli viewBase):
  .venv/bin/python experiments/hypothesis-one/viz.py    → http://127.0.0.1:8080
"""
import os
import sys
import time
import atexit
import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                # answering & spol.
sys.path.insert(0, os.path.join(HERE, "..", ".."))     # jellyai (viewBase adaptér)

from answering import Answering                          # noqa: E402
from jellyai.viz.viewbase_view import ViewBaseView       # noqa: E402

PIDFILE = os.path.join(HERE, ".viz.pid")                 # jeden běžící viewer (self-restart)


def _alive(pid):
    """Běží proces s tímto PID?"""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _claim_singleton():
    """Zabij PŘEDCHOZÍ instanci viz.py (z PID souboru) a zaber port.

    Idempotentní spuštění: re-run viz.py sám zabije starou instanci → uvolní se 8080
    (žádné „address already in use"). Počká, až starý proces port skutečně pustí.
    """
    try:
        old = int(open(PIDFILE, encoding="utf-8").read().strip())
    except (FileNotFoundError, ValueError):
        old = None
    if old and old != os.getpid() and _alive(old):
        try:
            os.kill(old, signal.SIGTERM)                 # zdvořile ukonči
            for _ in range(50):                          # počkej max ~5 s na uvolnění portu
                if not _alive(old):
                    break
                time.sleep(0.1)
        except ProcessLookupError:
            pass                                         # už neběžel
        print(f"→ zabita předchozí instance (PID {old})")
    with open(PIDFILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    atexit.register(_release)


def _release():
    """Při ukončení smaž PID soubor (jen když je náš)."""
    try:
        if int(open(PIDFILE, encoding="utf-8").read().strip()) == os.getpid():
            os.remove(PIDFILE)
    except (FileNotFoundError, ValueError, OSError):
        pass

TOP_CTX = 18
SEED_COL = "#5aa0ff"
CTX_COL = "#cfe0ff"
ANSWER_COL = {"answer": "#5aff9a", "clarify": "#ffb15a", "unsure": "#8a94a3"}


def process(answering, q):
    """Spustí jeden tah S ŘETĚZENÍM kontextu; vrátí stav pole + odpověď.

    NEMAŽE pole ani mount — světlo i dokumenty PŘETRVÁVAJÍ mezi tahy a pohasínají
    (decay uvnitř `answer(carry_context=True)`). To je přesně, co vizualizace ukazuje:
    aktivace se přenáší a uhasíná; dokumenty se po pohasnutí odeberou z panelu.
    """
    a = answering
    seed = [l.lower() for l in a._question(a._parse(q))[1]]   # slova otázky (pro flow/barvu)
    r = a.answer(q, carry_context=True)                       # plný tah: mount+světlo přetrvá
    return {"mounted": sorted(a.store.mounted), "seed": seed,
            "words": dict(a.field.words), "files": dict(a.field.files),
            "best": ({"answer": r["answer"]} if r else None),
            "mode": (r["mode"] if r else "unsure"),
            "offer": (r["offer"] if r else [])}


def build_view(answering):
    """Vytvoří viewBase pohled + handler dotazu. Vrací (view, on_ask) — bez serve."""
    a = answering
    view = ViewBaseView("hypothesis-one · živý dialog")
    try:
        view.open_nodes_panel()                          # ⚡ uzly dle jasu
        view.open_docs_panel()                           # 📄 aktivní dokumenty (nahraj → pohasni → odeber)
    except Exception:                                    # panely jsou nice-to-have
        pass
    shown = set()

    def render(st):
        nonlocal shown
        words = st["words"]
        ans = st["best"]["answer"] if st["best"] else None
        ans_id = ans.lower() if ans else None
        ctx = sorted(((h, w) for w, h in words.items() if w not in st["seed"]),
                     reverse=True)[:TOP_CTX]
        hmax = max((h for h, _ in ctx), default=1.0) or 1.0

        new = {}
        for w in st["seed"]:                             # slova otázky (modrá)
            new[w] = dict(label=w, color=SEED_COL, size=1.6)
        for h, w in ctx:                                 # rozsvícený kontext (jas dle tepla)
            new.setdefault(w, dict(label=w, color=CTX_COL,
                                   size=round(0.6 + 1.7 * h / hmax, 2)))
        if ans_id:                                       # odpověď (barva dle režimu)
            new[ans_id] = dict(label=(ans + " ✓") if st["mode"] == "answer" else ans,
                               color=ANSWER_COL.get(st["mode"], "#8a94a3"), size=2.3)

        for old in shown - set(new):                     # DYNAMICKÝ LOADER: odeber staré
            view.remove_node(old)
        for nid, meta in new.items():                    # přidej/aktualizuj nové
            (view.update_node if nid in shown else view.add_node)(nid, **meta)
        ids = set(new)
        for w in new:                                    # hrany = vodivost mezi zobrazenými
            for nb, cond in list(a.field.adj.get(w, {}).items()):
                if nb in ids and w < nb:
                    view.add_edge(w, nb, kind="ctx", weight=round(cond, 3),
                                  brightness=min(1.0, cond / hmax))
        shown = ids
        if ans_id and st["seed"]:                        # světlo teče od dotazu k odpovědi
            view.flow([st["seed"][0], ans_id])
            view.focus(ans_id)
        try:
            view.write_nodes(sorted(words.items(), key=lambda x: -x[1])[:12])
            # 📄 dokumenty dle AKTIVACE (ne mountu): pohaslé pod prahem se z panelu odeberou
            docs = [(d, s) for d, s in sorted(st["files"].items(), key=lambda x: -x[1])
                    if s > 0.05][:12]
            view.write_docs(docs)
        except Exception:
            pass

    def on_ask(q):
        q = (q or "").strip()
        if not q:
            return
        st = process(a, q)
        render(st)
        lit = [d for d, _s in sorted(st["files"].items(), key=lambda x: -x[1])[:4]]
        if st["mode"] == "answer" and st["best"]:
            view.write(f"❓ {q}\n  → {st['best']['answer']}   "
                       f"(rozsvíceno: {', '.join(lit)})\n")
        elif st["mode"] == "clarify":
            view.write(f"❓ {q}\n  nejsem si jist — mám: "
                       f"{', '.join(str(o) for o in st['offer'][:4] if o)}. Upřesni?\n")
        else:
            view.write(f"❓ {q}\n  — o tom nemám jasná data (nehádám)\n")

    return view, on_ask


def main():
    view, on_ask = build_view(Answering())
    view.open_terminal(on_ask)
    view.write("Ptej se — graf se rozsvítí. Např.: Kdo napsal Švejka?\n")
    print("→ viewBase na http://127.0.0.1:8080  (Ctrl-C ukončí)")
    view.serve(open_browser=False, block=True)


if __name__ == "__main__":
    main()
