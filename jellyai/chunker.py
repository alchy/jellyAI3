"""Rozdělení dokumentů na menší překrývající se pasáže.

Retriever nehledá v celých knihách, ale v malých kouscích textu — pasážích.
Kdybychom knihu nakrájeli na kusy „natvrdo" bez překryvu, mohla by se odpověď
octnout přesně na řezu a rozpůlit se mezi dvě pasáže (a v žádné by pak nedávala
smysl). Proto sousední pasáže sdílejí několik vět: raději trochu opakování než
useknutá pointa.
"""

from dataclasses import dataclass

from jellyai.text import split_sentences


@dataclass
class Passage:
    """Souvislý úsek dokumentu (jednotka, ve které retriever hledá).

    Atributy:
        doc_id (str): Identifikátor zdrojového dokumentu.
        index (int): Pořadí pasáže v rámci dokumentu (od 0).
        text (str): Text pasáže (spojené věty).
        start (int): Index první věty pasáže v dokumentu.
        end (int): Index za poslední větou (exkluzivní), takže `end - start`
            je počet vět v pasáži.
    """
    doc_id: str
    index: int
    text: str
    start: int
    end: int


def chunk(document, config):
    """Rozseká dokument na překrývající se pasáže po `size` větách.

    Cílem je připravit „stravitelná sousta" pro retriever: dokument se rozdělí
    na okna o `size` větách, přičemž každé další okno se posune jen o
    `size - overlap` vět dopředu — díky tomu se `overlap` vět objeví ve dvou
    sousedních pasážích a odpověď se neztratí na hranici. Krátký dokument
    (méně vět než `size`) se vrátí jako jediná pasáž.

    Args:
        document (Document): Dokument k rozsekání.
        config (ChunkerConfig): Nastavení — `size` (vět na pasáž) a `overlap`
            (kolik vět sdílejí sousední pasáže). `overlap` se pojistně ořízne
            na maximálně `size - 1`, aby se okno vždy posouvalo dopředu.

    Returns:
        list[Passage]: Pasáže v pořadí výskytu; prázdný seznam pro prázdný text.
    """
    sentences = split_sentences(document.text)
    n = len(sentences)
    if n == 0:
        return []
    size = max(1, config.size)
    # overlap musí být menší než size, jinak by se okno neposouvalo a smyčka
    # by běžela do skonání světa (nebo aspoň do dojití paměti).
    overlap = max(0, min(config.overlap, size - 1))
    step = size - overlap
    passages = []
    idx = 0
    i = 0
    while i < n:
        window = sentences[i:i + size]
        passages.append(Passage(
            doc_id=document.doc_id,
            index=idx,
            text=" ".join(window),
            start=i,
            end=i + len(window),
        ))
        idx += 1
        if i + size >= n:
            break
        i += step
    return passages


def explain():
    """Vrátí lidský popis bloku Chunker pro výukovou vrstvu.

    Returns:
        str: Několikavětý popis účelu, vstupu a výstupu bloku.
    """
    return (
        "Chunker rozseká Document na překrývající se pasáže po `size` větách "
        "(sousední pasáže sdílejí `overlap` vět). Překryv zajišťuje, že se odpověď "
        "neztratí na hranici dvou pasáží."
    )
