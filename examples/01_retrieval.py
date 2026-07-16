"""01 — Retrieval od nuly (BM25), bez modelů.

Ukazuje granulární bloky: rozsekej texty na pasáže a najdi nejrelevantnější.
Spusť: python examples/01_retrieval.py
"""

import jellyai
from config import RetrieverConfig
from jellyai.chunker import Passage

# malý „korpus" jako pasáže (doc_id, index, text, start, end)
passages = [
    Passage("d", 0, "Roboti se vzbouřili proti lidem.", 0, 1),
    Passage("d", 1, "Božena Němcová napsala Babičku.", 1, 2),
    Passage("d", 2, "Moře bylo toho dne modré a klidné.", 2, 3),
]

retriever = jellyai.Retriever(RetrieverConfig()).build(passages)

for passage, score in retriever.search("kdo napsal Babičku"):
    print(f"{score:.3f}  {passage.text}")

# teplota shody — širší sada kandidátů (pro pozdější kompozici)
wide = retriever.search("roboti lidé", temperature=1.0)
print("\nširoce:", [p.text for p, _ in wide])
