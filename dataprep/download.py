"""Stahování public-domain knih do korpusu.

Záměrně bez knihovny `requests` — vystačíme si se standardním `urllib`, ať má
projekt co nejmíň závislostí. Stahování je „best effort": když je nějaký zdroj
zrovna nedostupný (servery public-domain archivů mívají své nálady), knihu
přeskočíme a jedeme dál, místo abychom kvůli jednomu odkazu shodili celou
přípravu dat.
"""

import os
import urllib.request


def book_targets(config):
    """Sestaví z konfigurace dvojice (url, cílová cesta) pro stažení.

    Oddělení „co a kam stáhnout" od samotného stahování dělá logiku testovatelnou
    bez sítě a přehlednou (cílová cesta = raw_dir + název souboru).

    Args:
        config (Config): Konfigurace; čte se `config.data.books` (dvojice
            url + filename) a `config.data.raw_dir`.

    Returns:
        list[tuple[str, str]]: Dvojice (url, cesta v raw_dir) v pořadí z configu.
    """
    targets = []
    for url, filename in config.data.books:
        targets.append((url, os.path.join(config.data.raw_dir, filename)))
    return targets


def download_books(config):
    """Stáhne všechny knihy z konfigurace do raw adresáře.

    Vytvoří raw adresář (pokud chybí) a postupně stahuje. Chyba u jednoho zdroje
    se jen ohlásí a pokračuje se dál — přátelské přeskočení místo tvrdého pádu.

    Args:
        config (Config): Konfigurace se seznamem knih a raw adresářem.

    Returns:
        list[tuple[str, bool]]: Dvojice (cílová cesta, úspěch) pro každou knihu.
    """
    os.makedirs(config.data.raw_dir, exist_ok=True)
    results = []
    for url, dest in book_targets(config):
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Staženo: {dest}")
            results.append((dest, True))
        except Exception as exc:  # noqa: BLE001 - chceme pokračovat dalšími zdroji
            print(f"Přeskočeno {url}: {exc}")
            results.append((dest, False))
    return results
