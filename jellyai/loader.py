import os
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


def load_documents(directory):
    """Načte všechny .txt z adresáře do Document objektů (seřazené podle jména)."""
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
    return (
        "Loader načte vyčištěné .txt soubory z adresáře do objektů Document "
        "(doc_id, title, text). Je vstupem pro Chunker."
    )
