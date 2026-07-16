"""Stažení biblických knih do korpusu (getbible.net API, v2).

Rozšíření učebního kontextu o jinou doménu — test univerzálnosti mechanismů
(Čapkův kontext zůstává zakonzervovaný, korpus se rozšiřuje, ne nahrazuje).
Překlad je parametr: `cep` (Český ekumenický, moderní čeština — výchozí,
UDPipe mu rozumí líp) nebo `bkr` (Bible kralická, archaická).

    .venv/bin/python dataprep/fetch_bible.py            # Genesis + Matouš (cep)
    .venv/bin/python dataprep/fetch_bible.py bkr 1 40 43

Po stažení: ./jelly index && python cli.py annotate && python cli.py graph.
"""
import json
import re
import sys
import urllib.request

_API = "https://api.getbible.net/v2/{translation}.json"


def _verse_text(verse):
    return (verse.get("text", "") if isinstance(verse, dict) else str(verse)).strip()


def fetch(translation="cep", numbers=(1, 40), raw_dir="data/raw"):
    """Stáhne vybrané knihy překladu a uloží je jako korpusové `.txt`.

    Args:
        translation (str): Kód překladu na getbible.net (cep/bkr…).
        numbers (iterable[int]): Čísla knih (1=Genesis, 40=Matouš…).
        raw_dir (str): Cílový adresář korpusu.

    Returns:
        list[str]: Zapsané soubory.
    """
    with urllib.request.urlopen(_API.format(translation=translation)) as resp:
        books = json.load(resp)["books"]
    written = []
    for number in numbers:
        book = next(b for b in books if int(b["nr"]) == int(number))
        paragraphs = []
        for chapter in book["chapters"]:
            verses = chapter["verses"] if isinstance(chapter, dict) else chapter
            paragraphs.append(" ".join(_verse_text(v) for v in verses))
        slug = re.sub(r"\W+", "_", book["name"].lower()).strip("_")
        path = f"{raw_dir}/bible_{slug}.txt"
        with open(path, "w", encoding="utf-8") as out:
            out.write("\n\n".join(paragraphs) + "\n")
        print(f"{path} | {book['name']} | {len(book['chapters'])} kapitol")
        written.append(path)
    return written


if __name__ == "__main__":
    ARGS = sys.argv[1:]
    fetch(ARGS[0] if ARGS else "cep",
          [int(n) for n in ARGS[1:]] or (1, 40))
