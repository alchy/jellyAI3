"""Čištění syrových stažených textů před indexací.

Public-domain texty (Project Gutenberg, Wikizdroje) obsahují balast, který do
korpusu nepatří: licenční hlavičky a patičky, nejednotné konce řádků, zdvojené
mezery. Tento modul balast odstraní, ale **záměrně zachová vše podstatné pro
češtinu** — diakritiku, interpunkci i velikost písmen — protože právě z toho se
má model/retriever učit správný jazyk.
"""

import os
import re

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF.*", re.IGNORECASE | re.DOTALL)


def clean_text(raw):
    """Vyčistí jeden syrový text do podoby vhodné pro indexaci.

    Cílem je dostat z „ošklivého" staženého souboru čistý běžící text: odstranit
    licenční obálku Project Gutenbergu (vše před značkou START a od značky END),
    sjednotit konce řádků na `\\n` a stlačit vícenásobné mezery a prázdné řádky.
    Diakritika, interpunkce i velikost písmen zůstávají nedotčené.

    Args:
        raw (str): Syrový obsah souboru tak, jak byl stažen/načten.

    Returns:
        str: Vyčištěný text bez balastu a bez okolních bílých znaků.
    """
    text = raw
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_processed(raw_dir, processed_dir):
    """Vyčistí všechny `.txt` z jednoho adresáře do druhého.

    Dávková obálka nad :func:`clean_text` — projde syrové texty v `raw_dir`,
    každý vyčistí a zapíše pod stejným jménem do `processed_dir` (ten se v případě
    potřeby vytvoří). Oddělení „syrové vs. vyčištěné" umožňuje kdykoli přečistit
    data znovu bez opětovného stahování.

    Args:
        raw_dir (str): Adresář se syrovými `.txt` soubory.
        processed_dir (str): Cílový adresář pro vyčištěné texty (vytvoří se).

    Returns:
        list[str]: Cesty k zapsaným vyčištěným souborům (v pořadí zpracování).
    """
    os.makedirs(processed_dir, exist_ok=True)
    written = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(raw_dir, name), encoding="utf-8") as f:
            cleaned = clean_text(f.read())
        dest = os.path.join(processed_dir, name)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(cleaned)
        written.append(dest)
    return written
