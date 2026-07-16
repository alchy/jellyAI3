"""Rozprostření „teploty" po tokenech věty — pseudo-attention na úrovni slov.

Myšlenka (experiment): každé slovo má jas a ten **difunduje na sousedy** — jak
lineárně („vedle sebe": okno s útlumem exp(−d/τ)), tak po **hranách závislostního
stromu** (head↔dítě). Iterativně: sousedova teplota zvedá teplotu středu. Vznikne
tak „teplotní krajina" věty, kde se slova v bohatém kontextu vzájemně posilují —
vyplavou kolokace („měl rád"), entity („Karel Čapek") i predikát věty.

Je to **odlehčená, interpretovatelná verze attention/embeddingu** a další využití
téhož principu jako `ActivationField` (warm/step) — jen nad lineárním proudem tokenů
místo nad uzly grafu. Čisté, bez UFAL (tokeny se předají hotové).
"""

import math

# obsahové slovní druhy dostanou vyšší základní jas než funkční (spojky, předložky…)
_CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "ADV"}


def _base_heat(tokens):
    """Základní jas tokenu: obsahové slovo 1.0, funkční 0.2; vlastní jméno +0.5
    (je vzácnější a nese víc informace)."""
    heat = []
    for tok in tokens:
        value = 1.0 if tok.get("upos") in _CONTENT else 0.2
        if tok.get("upos") == "PROPN":
            value += 0.5
        heat.append(value)
    return heat


def _neighbor_weights(tokens, window, tau, use_dep, back, fwd):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Váhy sousedství: lineární okno (exp(−d/τ)) se **směrovým** faktorem — levý
    (předchozí) soused × `back`, pravý (následující) × `fwd` — plus symetrické
    hrany závislostí. `back==fwd==1` = symetrie; text je ale autoregresní, tak
    lze levý/pravý kontext vážit různě."""
    n = len(tokens)
    weights = [{} for _ in range(n)]
    for i in range(n):
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if i != j:
                direction = back if j < i else fwd     # levý soused vs pravý
                weights[i][j] = (weights[i].get(j, 0.0)
                                 + direction * math.exp(-abs(i - j) / tau))
        if use_dep:
            head = tokens[i].get("head", 0)
            if head and 1 <= head <= n and head - 1 != i:
                weights[i][head - 1] = weights[i].get(head - 1, 0.0) + 1.0
                weights[head - 1][i] = weights[head - 1].get(i, 0.0) + 1.0
    return weights


def spread_field(tokens, *, window=2, tau=1.5, alpha=0.5, steps=3,  # pylint: disable=too-many-arguments
                 use_dep=True, back=1.0, fwd=1.0):
    """Rozprostře teplotu po tokenech věty (difúze na sousedy), vrátí jas na token.

    Args:
        tokens (list[dict]): Tokeny věty (upos, head 1-based; 0 = kořen).
        window (int): Šířka lineárního okna sousedství (na každou stranu).
        tau (float): Dosah útlumu vzdálenosti (exp(−d/τ)).
        alpha (float): Váha příspěvku sousedů oproti základnímu jasu.
        steps (int): Počet iterací difúze (soused zvedá střed, opakovaně).
        use_dep (bool): Zahrnout i hrany závislostního stromu.
        back (float): Váha levého (předchozího) souseda — autoregresní směr.
        fwd (float): Váha pravého (následujícího) souseda.

    Returns:
        list[float]: Jas (teplota) na token, normovaný na maximum 1.0.
    """
    if not tokens:
        return []
    base = _base_heat(tokens)
    weights = _neighbor_weights(tokens, window, tau, use_dep, back, fwd)
    heat = list(base)
    for _ in range(steps):
        nxt = list(base)
        for i, neighbors in enumerate(weights):
            nxt[i] += alpha * sum(w * heat[j] for j, w in neighbors.items())
        top = max(nxt) or 1.0
        heat = [value / top for value in nxt]     # normalizace drží jas omezený
    return heat


def heat_landscape(tokens, heat, width=24):
    """Textová „teplotní krajina" věty — token + proužek podle jasu (pro demo)."""
    lines = []
    for tok, value in zip(tokens, heat):
        stripe = "█" * round(value * width)
        lines.append(f"{tok.get('form', tok.get('lemma', '?')):<14} "
                     f"{value:.2f} {stripe}")
    return "\n".join(lines)
