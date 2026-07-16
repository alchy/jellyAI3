"""Pulzující trasy — aktivace („context window") řídí hustotu 'provozu' paketů.

Aktivace je obdoba kontextového okna jazykového modelu: množina uzlů, které jsou
teď v kontextu, s vahami (jas). Vizualizace to ukazuje dvěma způsoby:

* **rozsvícení uzlů** — velikost ∝ jas (víc uzlů svítí zároveň),
* **provoz paketů po trasách** — každá nedávná trasa (topic→answer) vysílá diskrétní
  pakety s hustotou ∝ své aktuální aktivaci (nejteplejší trasa nejhustší, jako
  „větší/menší provoz" v realtime packet-flow příkladu viewBase).

Model je **per-dotaz** (žádný wall-clock časovač): mezi dotazy je provoz ustálený,
další dotaz posune pole (answerer × decay) a trasy s vyhaslými uzly (jas pod
prahem → mimo pole) přestanou vysílat. Drží vlastní kopii; nezasahuje do answereru.

Souvisí s [[jellyai3-fact-graph]] (ActivationField — 4. využití: vizualizace).
"""

import threading

_GAP_SCALE = 1.5     # jas → mezera mezi pakety: gap = _GAP_SCALE / jas (v ticích)
_GAP_MAX = 25        # nejřidší přípustný provoz (aby velmi slabá trasa občas blikla)
_MAX_TRACES = 6      # kolik nedávných tras držet souběžně


def _gap_ticks(intensity):
    """Jas trasy → mezera mezi pakety (v ticích). Vyšší jas = menší mezera = hustší."""
    if intensity <= 0:
        return _GAP_MAX
    return max(1, min(_GAP_MAX, round(_GAP_SCALE / intensity)))


class TracePulse:
    """Rozsvícení uzlů + provoz po nedávných trasách; hustota ∝ aktivaci."""

    def __init__(self, max_traces=_MAX_TRACES):
        """Args: max_traces (int): Kolik nedávných tras držet souběžně."""
        self._sizes = {}       # uzel → jas (celé aktivační pole = kontext)
        self._traces = []      # [{"path": [...], "intensity": float, "counter": int}]
        self._max = max_traces
        self._lock = threading.Lock()

    def ignite(self, scores, path):
        """Nový dotaz: nastaví pole (kontext) a přidá trasu; přepočte hustoty.

        Args:
            scores (dict): uzel → jas (celé `ActivationField.scores`).
            path (list): Trasa dotazu [topic, answer], nebo prázdná/None.

        Returns:
            dict: {„sizes": {uzel: jas}, „extinguish": [uzly ke zhasnutí]}.
        """
        with self._lock:
            prev = set(self._sizes)
            self._sizes = {node: float(score) for node, score in scores.items()}
            if path and len(path) >= 2:
                self._traces.insert(0, {"path": list(path), "counter": 0})
                self._traces = self._traces[:self._max]
            live = []
            for trace in self._traces:
                intensity = max((self._sizes.get(node, 0.0) for node in trace["path"]),
                                default=0.0)
                if intensity > 0.0:            # uzly trasy jsou ještě v kontextu
                    trace["intensity"] = intensity
                    live.append(trace)
            self._traces = live
            return {"sizes": dict(self._sizes),
                    "extinguish": sorted(prev - set(self._sizes))}

    def tick(self):
        """Jeden tik: vrátí trasy, po nichž je teď čas vyslat paket (rate ∝ jasu).

        Returns:
            dict: {„packets": [[topic, answer], …]} — každá položka je trasa
            jednoho vyslaného paketu (může jich být víc souběžně).
        """
        with self._lock:
            packets = []
            for trace in self._traces:
                trace["counter"] += 1
                if trace["counter"] >= _gap_ticks(trace["intensity"]):
                    trace["counter"] = 0
                    packets.append(list(trace["path"]))
            return {"packets": packets}
