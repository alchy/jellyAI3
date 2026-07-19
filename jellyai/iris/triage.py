"""Triage telemetrie (BACKLOG #38) — shlukování tahů s miss/nízkou assurance.

Stopu tahů píše `IrisAutomaton.turn` (telemetry_path → JSONL); tenhle
modul ji čte a shlukuje, aby provoz plnil etalony průběžně: největší
shluk = nejbolavější díra. Miss pozná podle druhu odpovědi, nízké
assurance a jazykových markerů (`miss_markers` v cs.json — texty
poctivých terminálů jsou jazyková data, ne kód).
"""

import json

from jellyai.graph.canon import deaccent
from jellyai.lang import current

_LOW_ASSURANCE = 0.5


def is_miss(row, threshold=_LOW_ASSURANCE):
    """Je tah kandidátem triage? (ne-odpověď, nízká jistota, marker missu)."""
    if row.get("kind") != "answer":
        return True
    if row.get("assurance", 1.0) < threshold:
        return True
    low = deaccent(str(row.get("answer", "")).lower())
    return any(deaccent(m.lower()) in low
               for m in current().get("miss_markers", ()))


def clusters(rows, threshold=_LOW_ASSURANCE):
    """Shluky missů podle (kind, vystřelené karty) — největší první.

    Returns:
        list[dict]: {"kind", "patterns", "count", "examples" (max 3)}.
    """
    groups = {}
    for row in rows:
        if not is_miss(row, threshold):
            continue
        key = (row.get("kind"), tuple(row.get("patterns", ())))
        bucket = groups.setdefault(key, {"kind": key[0],
                                         "patterns": list(key[1]),
                                         "count": 0, "examples": []})
        bucket["count"] += 1
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(row.get("q", ""))
    return sorted(groups.values(),
                  key=lambda b: (-b["count"], b["kind"] or ""))


def report(rows, threshold=_LOW_ASSURANCE):
    """Lidský výpis triage: shluky s počty a ukázkami otázek."""
    found = clusters(rows, threshold)
    if not found:
        return "Žádné tahy k triage — provoz bez missů. 🎉"
    lines = [f"TRIAGE: {sum(b['count'] for b in found)} tahů "
             f"v {len(found)} shlucích (miss / assurance < {threshold})"]
    for bucket in found:
        cards = ", ".join(bucket["patterns"]) or "bez karty"
        lines.append(f"\n{bucket['count']}× [{bucket['kind']} | {cards}]")
        lines += [f"   ❓ {q}" for q in bucket["examples"]]
    return "\n".join(lines)


def load_rows(path):
    """Načte stopu tahů z JSONL (chybějící soubor = prázdný provoz)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []
