from dataclasses import dataclass

from jellyai.text import split_sentences


@dataclass
class Passage:
    doc_id: str
    index: int
    text: str
    start: int   # index první věty
    end: int     # index za poslední větou (exkluzivní)


def chunk(document, config):
    """Rozseká dokument na pasáže po `size` větách s překryvem `overlap`."""
    sentences = split_sentences(document.text)
    n = len(sentences)
    if n == 0:
        return []
    size = max(1, config.size)
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
    return (
        "Chunker rozseká Document na překrývající se pasáže po `size` větách "
        "(sousední pasáže sdílejí `overlap` vět). Překryv zajišťuje, že se odpověď "
        "neztratí na hranici dvou pasáží."
    )
