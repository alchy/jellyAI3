#!/usr/bin/env python3
"""Natrénuje naučenou ASSURANCE bránu → gate.json (malé pevné váhy, inference deterministická).

Vstup: gate_data.json (rysy vítěze + label správně/špatně; sesbírá gen: answer(return_features)
přes etalon). Výstup: gate.json {w, b, mu, sd, dim}. Brána v answer(): p(správně)<práh → nehádej.
Poctivý odhad dopadu = LOO sweep (gate_sweep): práh 0.20 → −11 confident-wrong / −1 správná.
Spuštění: python3 train_gate.py   (potřebuje gate_data.json — viz gate_pilot / return_features)
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data/gate_data.json")
OUT = os.path.join(HERE, "data/gate.json")


def sigmoid(z): return 1 / (1 + np.exp(-z))


def train(X, y, epochs=1200, lr=0.3, l2=1e-2):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b); g = p - y
        w -= lr * (X.T @ g / len(X) + l2 * w); b -= lr * g.mean()
    return w, b


def collect():
    """Sesbírá (rysy vítěze, správně?) přes velký etalon — answer(return_features).
    Reprodukovatelnost: když gate_data.json chybí, vyrobí ho (pomalé, UDPipe)."""
    from answering import Answering
    a = Answering()
    gold = json.load(open(os.path.join(HERE, "data/gold/gold_large.json"), encoding="utf-8"))

    def acc(got, ans):
        g = (got or "").strip().lower()
        return bool(g) and any(x and (g == x.lower() or x.lower() in g or g in x.lower()) for x in ans)
    rows = []
    for it in gold:
        a.store.mounted.clear(); a.facts.mounted.clear()
        a.field.words.clear(); a.field.files.clear(); a.field.adj.clear()
        r = a.answer(it["q"], return_features=True)
        if r and r["mode"] == "answer" and "features" in r:
            rows.append({"feat": r["features"], "correct": int(acc(r["answer"], it["expect"]))})
    json.dump(rows, open(DATA, "w", encoding="utf-8"), ensure_ascii=False)
    return rows


def main():
    rows = json.load(open(DATA, encoding="utf-8")) if os.path.exists(DATA) else collect()
    X = np.array([r["feat"] for r in rows], float); y = np.array([r["correct"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w, b = train((X - mu) / sd, y)
    json.dump({"w": w.tolist(), "b": float(b), "mu": mu.tolist(), "sd": sd.tolist(),
               "dim": X.shape[1]}, open(OUT, "w", encoding="utf-8"))
    acc = ((sigmoid(((X - mu) / sd) @ w + b) > 0.5).astype(int) == y).mean()
    print(f"gate natrénována: {len(y)} vzorků, {X.shape[1]} rysů, train-acc {acc:.1%} → {OUT}")


if __name__ == "__main__":
    main()
