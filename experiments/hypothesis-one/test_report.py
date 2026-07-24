#!/usr/bin/env python3
"""Zápis výsledku etalonu do HTML scoreboardu (docs/last-test.html).

Offline, bez externích zdrojů, laděné s docs/style.css (světlé i tmavé téma). Univerzální:
bere normalizované řádky {kind, q, got, expect, mode, ok, doc_ok, expect_doc} → per-doména
(kind) skóre + tabulka otázka / odpověď / očekávaná. Volatelné z libovolného etalonu.
"""
import os
import html
from collections import defaultdict

# malý scoped styl navíc k docs/style.css (barvy dle --ok/--warn proměnných, téma-safe)
EXTRA_CSS = """
.big { font-family: var(--serif); font-size: 2.4rem; font-weight: 600; line-height: 1; }
.sub { color: var(--muted); }
.ok { color: var(--ok); font-weight: 700; }
.bad { color: var(--warn); font-weight: 700; }
.pill.ok { color: var(--ok); background: var(--out-bg); }
.pill.bad { color: var(--warn); background: transparent; border: 1px solid var(--warn); }
.meter { display: inline-block; width: 110px; height: 9px; background: var(--line);
  border-radius: 5px; vertical-align: middle; overflow: hidden; }
.meter > i { display: block; height: 100%; background: var(--accent); }
td.got.bad { color: var(--warn); }
td.q { font-family: var(--sans); }
table.items td { font-size: .9rem; }
""".strip()


def _e(s):
    return html.escape("" if s is None else str(s))


def _meter(p, t):
    pct = round(100 * p / t) if t else 0
    return f'<span class="meter"><i style="width:{pct}%"></i></span> {p}/{t} · {pct} %'


def write_scoreboard(out_path, title, rows, *, subtitle=None, config_label=None,
                     timestamp=None, sample=None, mode_summary=None, source=None):
    """rows: list dict {kind, q, got, expect(list), mode, ok(bool), doc_ok(bool|None), expect_doc}.

    sample=None → všechny položky; sample=int → per doména VŠECHNY propady + až `sample` PASSů
    (reprezentativní vzorek). mode_summary: [(mode, pass, total), …] pro sweep porovnání.
    """
    n = len(rows)
    npass = sum(1 for r in rows if r["ok"])
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    order = sorted(by_kind, key=lambda k: -len(by_kind[k]))

    out = []
    w = out.append
    w("<!doctype html>")
    w('<html lang="cs">')
    w("<head>")
    w('<meta charset="utf-8">')
    w('<meta name="viewport" content="width=device-width, initial-scale=1">')
    w(f"<title>{_e(title)} — hypothesis-one</title>")
    w('<link rel="stylesheet" href="style.css">')
    w(f"<style>{EXTRA_CSS}</style>")
    w("</head>")
    w("<body>")
    w('<div class="wrap">')
    w('<nav class="crumbs"><a href="index.html">Dokumentace</a> › Poslední test</nav>')
    meta_bits = []
    if timestamp:
        meta_bits.append(_e(timestamp))
    if config_label:
        meta_bits.append(_e(config_label))
    if source:
        meta_bits.append(_e(source))
    w(f'<p class="eyebrow">{" · ".join(meta_bits)}</p>' if meta_bits else '<p class="eyebrow">Scoreboard</p>')
    w(f"<h1>{_e(title)}</h1>")
    if subtitle:
        w(f'<p class="lead">{_e(subtitle)}</p>')
    pct = round(100 * npass / n) if n else 0
    cls = "ok" if pct >= 80 else ("bad" if pct < 50 else "")
    w(f'<p><span class="big {cls}">{npass}/{n}</span> <span class="sub">= {pct} % správně</span></p>')

    if mode_summary:
        w("<h2>Módy (sweep)</h2>")
        w('<div style="overflow-x:auto"><table>')
        w("<tr><th>mode</th><th>skóre</th></tr>")
        for m, p, t in mode_summary:
            w(f"<tr><td><code>{_e(m)}</code></td><td>{_meter(p, t)}</td></tr>")
        w("</table></div>")

    # per-doména souhrn
    w("<h2>Domény</h2>")
    w('<div style="overflow-x:auto"><table>')
    w("<tr><th>doména</th><th>skóre</th></tr>")
    for k in order:
        rs = by_kind[k]
        p = sum(1 for r in rs if r["ok"])
        w(f"<tr><td>{_e(k)}</td><td>{_meter(p, len(rs))}</td></tr>")
    w("</table></div>")

    # per-doména položky
    for k in order:
        rs = by_kind[k]
        p = sum(1 for r in rs if r["ok"])
        shown = rs
        note = ""
        if sample is not None:
            fails = [r for r in rs if not r["ok"]]
            passes = [r for r in rs if r["ok"]]
            shown = fails + passes[:sample]
            if len(shown) < len(rs):
                note = f' <span class="sub">(zobrazeno {len(shown)}/{len(rs)} — všechny propady + vzorek)</span>'
        has_doc = any(r.get("expect_doc") for r in rs)
        w(f"<h3>{_e(k)} — {p}/{len(rs)}{note}</h3>")
        w('<div style="overflow-x:auto"><table class="items">')
        hdr = "<tr><th></th><th>otázka</th><th>odpověď</th><th>očekávaná</th><th>mód</th>"
        hdr += "<th>soubor</th></tr>" if has_doc else "</tr>"
        w(hdr)
        for r in shown:
            mark = '<span class="pill ok">✓</span>' if r["ok"] else '<span class="pill bad">✗</span>'
            got = _e(r.get("got") or "—")
            gotcls = "got" + ("" if r["ok"] else " bad")
            exp = _e(" / ".join(r.get("expect") or []))
            cells = [f"<td>{mark}</td>", f'<td class="q">{_e(r["q"])}</td>',
                     f'<td class="{gotcls}">{got}</td>', f"<td>{exp}</td>",
                     f'<td><code>{_e(r.get("mode") or "")}</code></td>']
            if has_doc:
                dok = r.get("doc_ok")
                if dok is None:
                    doc_cell = '<span class="sub">—</span>'
                elif dok:
                    doc_cell = '<span class="pill ok">✓</span>'
                else:
                    doc_cell = f'<span class="pill bad">✗</span> <span class="sub">{_e(r.get("expect_doc"))}</span>'
                cells.append(f"<td>{doc_cell}</td>")
            w("<tr>" + "".join(cells) + "</tr>")
        w("</table></div>")

    w('<p class="sub" style="margin-top:2.5rem">Generováno <code>eval_domain.py</code> → '
      '<code>test_report.write_scoreboard</code>. Ground truth kurátorovaný, ověřený proti raw korpusu.</p>')
    w("</div>")
    w("</body></html>")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return out_path
