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

# Řádek se sekčním nadpisem (Wikipedia „explaintext" je nechává jako „== Nadpis ==").
_SECTION_RE = re.compile(r"^\s*=+\s*(.+?)\s*=+\s*$")
# Sekce, od nichž dál je už jen bibliografický balast (ne próza) → uříznout.
_REFERENCE_SECTIONS = {
    "odkazy", "reference", "literatura", "externí odkazy", "související články",
    "související", "poznámky", "bibliografie", "prameny",
}

# Rozsahová biografická závorka: obsahuje rok (15xx–20xx) i pomlčku (narození–úmrtí).
_DATE_RANGE_PAREN = re.compile(
    r"\s*\([^()]*\b(?:1[5-9]\d\d|20\d\d)\b[^()]*[–—-][^()]*\)"
)


def _strip_wiki_apparatus(text):
    """Odřízne referenční/bibliografické sekce a zahodí sekční nadpisy.

    Wikipedia je skvělá čistá próza, jenže na konci má seznamy zdrojů, ISBN a
    citace, z nichž by generátor QA dat dělal nesmyslné otázky. Od prvního
    „referenčního" nadpisu (Odkazy/Reference/Literatura…) proto text uřízneme;
    ostatní sekční nadpisy (samotné „== Dílo ==") zahodíme, protože to není próza.
    Na běžný text bez „==" nadpisů (třeba drama) tahle funkce nesáhne.

    Args:
        text (str): Text s normalizovanými konci řádků (\\n).

    Returns:
        str: Text bez sekčních nadpisů a bez referenční části.
    """
    kept = []
    for line in text.split("\n"):
        match = _SECTION_RE.match(line)
        if match:
            title = match.group(1).strip().lower()
            if title in _REFERENCE_SECTIONS:
                break            # odtud dál je jen balast
            continue             # ostatní nadpisy nejsou próza → pryč
        kept.append(line)
    return "\n".join(kept)


def _strip_date_range_parens(text):
    """Odstraní jednoznačné rozsahové datové závorky (narození–úmrtí).

    Cílí jen na závorky, které mají rok i pomlčku — ty v úvodních větách dělají
    otázky nečitelnými. Běžné závorky (bez roku a pomlčky) nechá být, ať se
    nesmaže užitečný obsah.

    Args:
        text (str): Vstupní text.

    Returns:
        str: Text bez rozsahových datových závorek.
    """
    return _DATE_RANGE_PAREN.sub("", text)


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
    text = _strip_wiki_apparatus(text)  # pryč s referencemi/nadpisy (řádkově)
    text = _strip_date_range_parens(text)  # pryč s (narození – úmrtí) závorkami
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_processed(raw_dir, processed_dir):
    """Vyčistí `.txt` z raw adresáře do processed tak, aby ho přesně zrcadlil.

    Dávková obálka nad :func:`clean_text` — projde syrové texty v `raw_dir`,
    každý vyčistí a zapíše pod stejným jménem do `processed_dir`. Aby processed
    věrně odpovídal raw, **odstraní i osiřelé** vyčištěné texty, které už v raw
    nemají svůj zdroj (jinak by po smazání knihy z raw strašila dál v korpusu).
    Oddělení „syrové vs. vyčištěné" umožňuje kdykoli přečistit data bez stahování.

    Args:
        raw_dir (str): Adresář se syrovými `.txt` soubory.
        processed_dir (str): Cílový adresář pro vyčištěné texty (vytvoří se).

    Returns:
        list[str]: Cesty k zapsaným vyčištěným souborům (v pořadí zpracování).
    """
    os.makedirs(processed_dir, exist_ok=True)
    raw_names = {n for n in os.listdir(raw_dir) if n.endswith(".txt")}

    # Zrcadlení: co v raw není, nemá co dělat ani v processed.
    for name in os.listdir(processed_dir):
        if name.endswith(".txt") and name not in raw_names:
            os.remove(os.path.join(processed_dir, name))

    written = []
    for name in sorted(raw_names):
        with open(os.path.join(raw_dir, name), encoding="utf-8") as f:
            cleaned = clean_text(f.read())
        dest = os.path.join(processed_dir, name)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(cleaned)
        written.append(dest)
    return written
