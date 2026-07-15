"""Stažení českých článků z Wikipedie jako korpusu.

Wikipedia je čistá, dobře psaná próza — na rozdíl od dramatu ideální zdroj pro
syntetická QA data. Stahujeme přes oficiální API v režimu „explaintext" (holý
text bez wiki značek). Samotný fetcher jde injektovat, aby šla logika testovat
bez sítě (a bez toho, aby testy zatěžovaly Wikipedii).
"""

import json
import os
import urllib.parse
import urllib.request

_API = "https://cs.wikipedia.org/w/api.php"


def _slug(title):
    """Převede název článku na bezpečné jméno souboru.

    Malá písmena, podtržítka místo mezer/lomítek; diakritika se zachová.

    Args:
        title (str): Název článku.

    Returns:
        str: Slug pro název souboru.
    """
    return title.lower().replace(" ", "_").replace("/", "_")


def fetch_extract(title):
    """Stáhne holý text jednoho článku z české Wikipedie.

    Args:
        title (str): Název článku (např. „Karel Čapek").

    Returns:
        str: Prostý text článku (prázdný řetězec, když článek neexistuje).
    """
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1", "titles": title,
    }
    url = _API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "jellyAI3/edu (local)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def fetch_articles(titles, dest_dir, fetch=fetch_extract):
    """Stáhne seznam článků do dest_dir jako wiki_<slug>.txt.

    Best-effort: prázdné či chybové články přeskočí a nahlásí, ať jeden nefungující
    název neshodí celý běh.

    Args:
        titles (list[str]): Názvy článků.
        dest_dir (str): Cílový adresář (vytvoří se).
        fetch (callable): Funkce title→text; injektovatelná kvůli testům bez sítě.

    Returns:
        list[str]: Cesty k zapsaným souborům.
    """
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    for title in titles:
        try:
            text = fetch(title)
        except Exception as exc:  # noqa: BLE001 - jeden neúspěch nesmí shodit zbytek
            print(f"Přeskočeno {title}: {exc}")
            continue
        if not text.strip():
            print(f"Přeskočeno (prázdné): {title}")
            continue
        path = os.path.join(dest_dir, f"wiki_{_slug(title)}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Staženo: {path}")
        written.append(path)
    return written
