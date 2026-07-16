"""Větný retriever se vzdálenostním útlumem (B1).

Místo pevných bloků skóruje na úrovni vět: nalezená věta vyzařuje své BM25 skóre
do okolí s exponenciálním útlumem podle vzdálenosti (sever i jih), **soubor je
tvrdá hranice**. Vrchol téhle aktivace je sémantický střed odpovědi; kolem něj se
vyrobí ostřicí okno jako běžná `Passage`, takže se answerer nemění.

Znovu používá V1 `Retriever` jako vnitřní BM25 skórovač (přes `score_all`), takže
se matematika neduplikuje a chování V1 zůstává nedotčené.
"""

from collections import defaultdict

import numpy as np

from jellyai.text import split_sentences
from jellyai.chunker import Passage
from jellyai.retriever import Retriever


def distance_activation(base, sent_doc, sent_local, tau):
    """Rozlije větná skóre do okolí s exponenciálním útlumem uvnitř souboru.

    Pro každou větu s sečte příspěvky všech vět t **téhož souboru** vážené
    `exp(−|pozice_s − pozice_t| / τ)`. Věta obklopená relevantními větami tak
    vyskočí; osamocená shoda zůstane skromná. Napříč soubory je příspěvek nulový —
    soubor je tvrdá hranice, ať systém nezávisí na formátování odstavců.

    Args:
        base (Sequence[float]): Základní (BM25) skóre každé věty.
        sent_doc (Sequence[str]): doc_id každé věty (pro seskupení do souborů).
        sent_local (Sequence[int]): Lokální index věty v jejím dokumentu.
        tau (float): Dosah útlumu; pojistně zdola omezen na 1e-6.

    Returns:
        numpy.ndarray: Aktivované (finální) skóre v pořadí vstupu.
    """
    base = np.asarray(base, dtype=float)
    n = len(base)
    finals = np.zeros(n)
    tau = max(float(tau), 1e-6)
    groups = defaultdict(list)
    for k in range(n):
        groups[sent_doc[k]].append(k)
    for idxs in groups.values():
        idxs = np.array(idxs)
        local = np.array([sent_local[k] for k in idxs], dtype=float)
        dist = np.abs(local[:, None] - local[None, :])
        weight = np.exp(-dist / tau)
        finals[idxs] = weight @ base[idxs]
    return finals


class SentenceRetriever:
    """Retriever nad větami se vzdálenostním útlumem a ostřicím oknem."""

    def __init__(self, config):
        """Vytvoří prázdný větný retriever.

        Args:
            config (RetrieverConfig): Metoda/BM25 parametry + `decay_tau`,
                `focus_radius`, `top_k`. Index vznikne až `build`.
        """
        self.config = config
        self.sent_doc = []
        self.sent_local = []
        self.sent_text = []
        self._bounds = {}
        self._retriever = None

    def build(self, documents):
        """Rozdělí dokumenty na věty a postaví nad nimi vnitřní BM25 index.

        Každý dokument se rozseká `split_sentences` na věty s **lokálním indexem**
        (od 0). Věty se ukládají dokument po dokumentu (souvisle), takže hranice
        souboru je prostě rozsah indexů. Vnitřní `Retriever` skóruje jednotlivé
        věty jako 1větné pasáže.

        Args:
            documents (list[Document]): Dokumenty korpusu.

        Returns:
            SentenceRetriever: `self` (pro řetězení).
        """
        passages = []
        for doc in documents:
            sentences = split_sentences(doc.text)
            start = len(self.sent_text)
            for local, sent in enumerate(sentences):
                self.sent_doc.append(doc.doc_id)
                self.sent_local.append(local)
                self.sent_text.append(sent)
                passages.append(Passage(doc.doc_id, local, sent, local, local + 1))
            if len(self.sent_text) > start:
                self._bounds[doc.doc_id] = (start, len(self.sent_text))
        self._retriever = Retriever(self.config).build(passages)
        return self

    def search(self, query, top_k=None):
        """Najde ostřicí okna kolem vrcholů aktivace k dotazu.

        Základní BM25 skóre vět (`score_all`) se rozlije vzdálenostním útlumem
        (`distance_activation`). Vrcholy se berou hladově odshora; věty, které už
        leží v dřívějším okně, se přeskočí (aby okna byla různá). Kolem každého
        vrcholu se vyrobí okno ± `focus_radius` vět (ořezané na hranice dokumentu)
        jako běžná `Passage` — rozhraní se tak neliší od V1.

        Args:
            query (str): Dotaz v češtině.
            top_k (int | None): Kolik oken vrátit; None = z konfigurace.

        Returns:
            list[tuple[Passage, float]]: (ostřicí okno, skóre vrcholu) sestupně;
                prázdný seznam pro prázdný index nebo nulové skóre.
        """
        if top_k is None:
            top_k = self.config.top_k
        n = len(self.sent_text)
        if n == 0:
            return []
        base = self._retriever.score_all(query)
        finals = distance_activation(base, self.sent_doc, self.sent_local,
                                     self.config.decay_tau)
        radius = self.config.focus_radius
        results = []
        covered = set()
        for k in np.argsort(-finals):
            if finals[k] <= 0:
                break
            if k in covered:
                continue
            doc_id = self.sent_doc[k]
            g0, g1 = self._bounds[doc_id]
            lo = max(g0, k - radius)
            hi = min(g1 - 1, k + radius)
            window = Passage(
                doc_id=doc_id,
                index=self.sent_local[k],
                text=" ".join(self.sent_text[lo:hi + 1]),
                start=self.sent_local[lo],
                end=self.sent_local[hi] + 1,
            )
            results.append((window, float(finals[k])))
            covered.update(range(lo, hi + 1))
            if len(results) >= top_k:
                break
        return results
