"""Vyhledávání relevantních pasáží — TF-IDF a BM25 psané od nuly.

Tohle je srdce celého QA: než může kdokoli odpovědět, musíme najít v korpusu ta
místa, která se dotazu týkají. Děláme to dvěma klasickými, „předneuronovými"
metodami, které jsou ale dodnes překvapivě silné — a hlavně se dají celé
pochopit a naprogramovat ručně, bez jediného staženého modelu:

* **TF-IDF** — každé slovo dostane váhu podle toho, jak je vzácné (idf), pasáže
  i dotaz se zapíší jako vektory a podobnost = kosinus úhlu mezi nimi.
* **BM25** — v jádru totéž, ale s dvěma vychytávkami: nasycením (desáté opakování
  slova už toho moc nepřidá) a normalizací délky (aby ukecané pasáže neměly
  nespravedlivou výhodu jen proto, že se v nich slovo statisticky spíš objeví).

Vše se počítá v numpy nad maticí četností slov — žádná magie, jen lineární
algebra a trocha logaritmů.
"""

import math

import numpy as np

from jellyai.text import tokenize, CZECH_STOPWORDS


class Retriever:
    """Index pasáží s vyhledáváním podle TF-IDF nebo BM25.

    Použití je dvoufázové: nejdřív `build(passages)` postaví index (spočítá
    slovník, četnosti a váhy), pak `search(query)` vrací nejrelevantnější pasáže.
    Metoda vyhledávání se řídí `config.method`.
    """

    def __init__(self, config):
        """Vytvoří prázdný retriever s danou konfigurací.

        Args:
            config (RetrieverConfig): Nastavení metody, top_k, BM25 parametrů
                a použití stopslov. Samotný index vznikne až voláním `build`.
        """
        self.config = config
        self.passages = []
        self.vocab = {}          # term -> sloupcový index v maticích
        self.counts = None       # matice četností [n_pasáží, n_termů]
        self.doc_len = None      # délka každé pasáže v tokenech
        self.avg_len = 0.0       # průměrná délka pasáže (pro BM25 normalizaci)
        self.df = None           # document frequency: v kolika pasážích term je
        self.idf = None          # inverse document frequency pro TF-IDF
        self._tfidf_norm = None  # řádkově normalizovaná TF-IDF matice (pro kosinus)

    def _tok(self, text):
        """Tokenizuje text stejným způsobem jako při stavbě indexu.

        Aby dotaz a pasáže „mluvily stejnou řečí", musí projít identickou
        tokenizací (jinak by se stejná slova nepotkala). Stopslova se řídí
        konfigurací.

        Args:
            text (str): Text k tokenizaci (pasáž nebo dotaz).

        Returns:
            list[str]: Tokeny připravené k indexaci/porovnání.
        """
        sw = CZECH_STOPWORDS if self.config.use_stopwords else None
        return tokenize(text, sw)

    def build(self, passages):
        """Postaví index nad zadanými pasážemi.

        Spočítá slovník všech slov, matici jejich četností v pasážích a z ní
        odvozené statistiky: délky pasáží, document frequency, idf a předpočítanou
        normalizovanou TF-IDF matici (aby kosinová podobnost při dotazu byla jen
        jedno maticové násobení). Vše se ukládá do instance.

        Args:
            passages (list[Passage]): Pasáže k zaindexování.

        Returns:
            Retriever: `self`, aby šlo psát `Retriever(cfg).build(passages)`.
        """
        self.passages = list(passages)
        docs_tokens = [self._tok(p.text) for p in self.passages]

        # Slovník: každému unikátnímu slovu přiřadíme index sloupce.
        vocab = {}
        for toks in docs_tokens:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab

        n_docs = len(self.passages)
        n_terms = len(vocab)
        counts = np.zeros((n_docs, n_terms), dtype=np.float64)
        for i, toks in enumerate(docs_tokens):
            for t in toks:
                counts[i, vocab[t]] += 1.0
        self.counts = counts
        self.doc_len = counts.sum(axis=1)
        self.avg_len = float(self.doc_len.mean()) if n_docs else 0.0
        self.df = (counts > 0).sum(axis=0)

        # TF-IDF: idf se smoothingem (+1), ať se nedělí nulou a vzácná slova váží víc.
        self.idf = np.log((n_docs + 1) / (self.df + 1)) + 1.0
        tfidf = counts * self.idf
        # Normalizace řádků na jednotkovou délku → skalární součin = kosinus.
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # prázdná pasáž ať nezpůsobí dělení nulou
        self._tfidf_norm = tfidf / norms
        return self

    def search(self, query, top_k=None):
        """Najde k dotazu nejrelevantnější pasáže.

        Dotaz se tokenizuje stejně jako pasáže, spočítá se skóre podle zvolené
        metody a vrátí se `top_k` pasáží s nejvyšším **kladným** skóre. Pasáže
        se skóre 0 (žádné společné slovo) se zahazují — je poctivější vrátit míň
        (nebo nic) než nutit dovnitř nesouvisející text.

        Args:
            query (str): Dotaz v češtině.
            top_k (int | None): Kolik pasáží vrátit; None = hodnota z konfigurace.

        Returns:
            list[tuple[Passage, float]]: Dvojice (pasáž, skóre) sestupně podle
                skóre; prázdný seznam, když nic nesedí nebo je index prázdný.
        """
        if top_k is None:
            top_k = self.config.top_k
        if not self.passages:
            return []
        tokens = self._tok(query)
        if self.config.method == "tfidf":
            scores = self._tfidf_scores(tokens)
        else:
            scores = self._bm25_scores(tokens)
        order = np.argsort(-scores)[:top_k]
        return [(self.passages[i], float(scores[i])) for i in order if scores[i] > 0]

    def _tfidf_scores(self, tokens):
        """Spočítá kosinovou podobnost dotazu se všemi pasážemi (TF-IDF).

        Dotaz se převede na TF-IDF vektor (stejné idf jako pasáže) a znormalizuje;
        podobnost s každou pasáží je pak jen skalární součin s předpočítanou
        normalizovanou maticí.

        Args:
            tokens (list[str]): Tokeny dotazu.

        Returns:
            numpy.ndarray: Skóre pro každou pasáž (v pořadí `self.passages`).
        """
        v = np.zeros(len(self.vocab))
        for t in tokens:
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return self._tfidf_norm @ v

    def _bm25_scores(self, tokens):
        """Spočítá BM25 skóre dotazu vůči všem pasážím.

        Pro každé slovo dotazu se sečte příspěvek přes všechny pasáže podle
        vzorce Okapi BM25: idf slova × nasycená četnost normalizovaná délkou
        pasáže. Slova mimo slovník se ignorují (v korpusu se nevyskytují).

        Args:
            tokens (list[str]): Tokeny dotazu.

        Returns:
            numpy.ndarray: Skóre pro každou pasáž (v pořadí `self.passages`).
        """
        k1 = self.config.k1
        b = self.config.b
        n_docs = len(self.passages)
        scores = np.zeros(n_docs)
        if self.avg_len == 0:
            return scores
        for t in set(tokens):  # set: každé slovo započítat jen jednou
            j = self.vocab.get(t)
            if j is None:
                continue
            f = self.counts[:, j]  # četnost slova v každé pasáži
            idf = math.log((n_docs - self.df[j] + 0.5) / (self.df[j] + 0.5) + 1.0)
            denom = f + k1 * (1 - b + b * self.doc_len / self.avg_len)
            scores += idf * (f * (k1 + 1)) / denom
        return scores


def explain():
    """Vrátí lidský popis bloku Retriever pro výukovou vrstvu.

    Returns:
        str: Několikavětý popis obou metod a toho, co blok dělá.
    """
    return (
        "Retriever indexuje pasáže a k dotazu vrátí ty nejrelevantnější.\n"
        "TF-IDF: každé slovo váží podle vzácnosti (idf), dokumenty i dotaz jsou "
        "vektory, podobnost = kosinus úhlu mezi nimi.\n"
        "BM25: vylepšené skórování s nasycením četnosti (k1) a normalizací délky "
        "dokumentu (b) — obvykle lepší než čisté TF-IDF.\n"
        "Vše počítáno od nuly v numpy, žádný stažený model."
    )
