"""Načtení vyčištěných textů do paměti jako Document objekty.

Loader je první blok knihovní pipeline. Jeho jediným úkolem je oddělit „jak se
text dostane z disku" od zbytku systému — chunker, retriever a answerer už pak
pracují jen s objekty Document a nemusí nic vědět o souborech a kódování.
"""

import os
from dataclasses import dataclass


@dataclass
class Document:
    """Jeden načtený dokument korpusu.

    Atributy:
        doc_id (str): Krátký identifikátor (název souboru bez přípony), používá
            se i ve zdrojích odpovědí (`doc_id#index`).
        title (str): Lidský název dokumentu (zatím shodný s doc_id).
        text (str): Celý vyčištěný text dokumentu.
    """
    doc_id: str
    title: str
    text: str


def load_documents(directory):
    """Načte všechny `.txt` z adresáře do seznamu Document objektů.

    Cílem je dát zbytku pipeline jednotný vstup nezávislý na souborovém systému.
    Načítají se pouze soubory s příponou `.txt`, seřazené podle jména kvůli
    deterministickému a reprodukovatelnému výsledku (důležité pro testy i pro
    stabilní indexy). `doc_id` se odvodí z názvu souboru bez přípony.

    Args:
        directory (str): Cesta k adresáři s vyčištěnými `.txt` texty.

    Returns:
        list[Document]: Načtené dokumenty seřazené podle názvu souboru.

    Raises:
        FileNotFoundError: Když adresář neexistuje (typicky se zapomnělo spustit
            `prepare-data`); hláška na to uživatele navede.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Adresář s texty neexistuje: {directory}. "
            f"Spusť nejdřív `python cli.py prepare-data`."
        )
    docs = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            text = f.read()
        doc_id = os.path.splitext(name)[0]
        docs.append(Document(doc_id=doc_id, title=doc_id, text=text))
    return docs


def explain():
    """Vrátí lidský popis bloku Loader pro výukovou vrstvu.

    Používá ji příkaz `cli.py explain loader`, aby uživatel pochopil roli bloku,
    aniž by musel číst zdrojový kód.

    Returns:
        str: Několikavětý popis účelu, vstupu a výstupu bloku.
    """
    return (
        "Loader načte vyčištěné .txt soubory z adresáře do objektů Document "
        "(doc_id, title, text). Je vstupem pro Chunker."
    )
