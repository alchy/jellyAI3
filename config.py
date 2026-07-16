"""Konfigurace všech bloků na jednom místě.

Každý blok knihovny (chunker, retriever, answerer, data) má svůj malý dataclass
s parametry a jejich výchozími hodnotami. Sdružení do jednoho `Config` znamená,
že celé chování systému se dá odečíst — a přeladit — z jediného objektu, aniž by
se parametry musely protahovat půltuctem funkcí. Kdo chce experimentovat, mění
hodnoty tady; zbytek kódu se nemusí dotknout.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Kde leží data a odkud se stahují.

    Atributy:
        raw_dir (str): Adresář pro syrové stažené/vložené texty.
        processed_dir (str): Adresář pro vyčištěné texty připravené k indexaci.
        index_path (str): Cesta k uloženému indexu („vektorům"), aby se nemusel
            stavět znovu při každém dotazu.
        books (list): Seznam dvojic (url, filename) public-domain knih ke stažení;
            dostupnost se ověřuje až při běhu `prepare-data`.
        wiki_titles (tuple): Názvy českých wiki článků ke stažení jako korpus
            čisté prózy (dobrý zdroj pro syntetická QA data).
    """
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_path: str = "data/index.pkl"
    books: list = field(default_factory=list)
    wiki_titles: tuple = (
        "Karel Čapek", "Josef Čapek", "R.U.R.", "Válka s mloky",
        "Bílá nemoc", "Božena Němcová", "Jan Neruda",
    )


@dataclass
class ChunkerConfig:
    """Jak krájet dokumenty na pasáže.

    Atributy:
        size (int): Počet vět na jednu pasáž.
        overlap (int): Kolik vět sdílejí sousední pasáže (kvůli odpovědím na řezu).
        unit (str): Jednotka dělení; zatím "sentences" (prostor pro "chars").
    """
    size: int = 3
    overlap: int = 1
    unit: str = "sentences"


@dataclass
class RetrieverConfig:
    """Jak vyhledávat relevantní pasáže.

    Atributy:
        method (str): "bm25" (výchozí, obvykle lepší) nebo "tfidf".
        top_k (int): Kolik nejlepších pasáží vrátit.
        k1 (float): BM25 — nasycení četnosti slova (vyšší = četnost víc rozhoduje).
        b (float): BM25 — míra normalizace délkou dokumentu (0 = žádná, 1 = plná).
        use_stopwords (bool): Zda při tokenizaci zahazovat česká stopslova.
        granularity (str): "passage" (V1) nebo "sentence" (větný retrieval B1).
        decay_tau (float): Dosah útlumu τ pro větný režim (exp(−d/τ)).
        focus_radius (int): Poloměr ostřicího okna ve větách (na každou stranu).
    """
    method: str = "bm25"
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    use_stopwords: bool = True
    granularity: str = "passage"  # "passage" (V1) nebo "sentence" (B1)
    decay_tau: float = 1.6         # dosah exponenciálního útlumu (věty)
    focus_radius: int = 2          # poloměr ostřicího okna (vět na každou stranu)


@dataclass
class AnswererConfig:
    """Jak z nalezených pasáží složit odpověď.

    Atributy:
        max_sentences (int): Kolik vět nejvýš vrátit jako odpověď (V1: typicky 1).
        template (bool): Zda odpověď zabalit do šablony („Podle textu: …").
        mode (str): Který answerer — "extractive" (V1), "template" (V3), "graph".
    """
    max_sentences: int = 1
    template: bool = True
    mode: str = "extractive"


@dataclass
class QagenConfig:
    """Nastavení generování syntetických QA párů (pro trénink generátoru V2).

    Atributy:
        qa_path (str): Kam zapsat JSONL dataset dvojic otázka→odpověď.
        morphodita_model (str): Cesta k modelu MorphoDiTa (POS/lemma).
        nametag_model (str): Cesta k modelu NameTag (NER).
        min_tokens (int): Minimální počet slov ve větě, aby dávala smysl jako kontext.
        max_tokens (int): Maximální počet slov ve větě; delší se přeskočí. Chrání
            před run-on „větami" (scénické poznámky, seznamy), z nichž by šablona
            udělala nesmyslně dlouhou otázku.
        max_answers_per_sentence (int): Kolik nejvíc odpovědí vytěžit z jedné věty.
        types (tuple): Povolené typy otázek.
    """
    qa_path: str = "data/qa/qapairs.jsonl"
    morphodita_model: str = "data/models/czech-morfflex.tagger"
    nametag_model: str = "data/models/czech-cnec.ner"
    min_tokens: int = 5
    max_tokens: int = 40
    max_answers_per_sentence: int = 2
    types: tuple = ("Kdo", "Co", "Kde", "Kdy", "Kolik")


@dataclass
class ServicesConfig:
    """Nastavení ÚFAL služeb (V3 — každý nástroj jako vlastní localhost proces).

    NameTag, UDPipe a MorphoDiTa se v jednom procesu perou (sdílený SWIG typ),
    tak každý běží samostatně a mluví se s ním přes malé HTTP API na localhostu.

    Atributy:
        host (str): Adresa, kde služby poslouchají (jen localhost).
        nametag_port, udpipe_port, morpho_port (int): Porty jednotlivých služeb.
        nametag_model, morphodita_model, udpipe_model (str): Cesty k modelům.
        startup_timeout (float): Max. čekání na naběhnutí služby (s).
        annotations_path (str): Kam ukládat offline anotace pasáží.
    """
    host: str = "127.0.0.1"
    nametag_port: int = 8081
    udpipe_port: int = 8082
    morpho_port: int = 8083
    nametag_model: str = "data/models/czech-cnec.ner"
    morphodita_model: str = "data/models/czech-morfflex.tagger"
    udpipe_model: str = "data/models/udpipe-czech.model"
    startup_timeout: float = 30.0
    annotations_path: str = "data/annotations.pkl"


@dataclass
class GraphConfig:
    """Nastavení faktového grafu.

    Atributy:
        graph_path (str): Cesta k uloženému faktovému grafu.
    """
    graph_path: str = "data/graph.pkl"


@dataclass
class Config:
    """Zastřešující konfigurace — jeden objekt vládne všem blokům.

    Atributy:
        data (DataConfig): Nastavení dat.
        chunker (ChunkerConfig): Nastavení krájení na pasáže.
        retriever (RetrieverConfig): Nastavení vyhledávání.
        answerer (AnswererConfig): Nastavení skládání odpovědi.
        qagen (QagenConfig): Nastavení generování syntetických QA dat (V2).
        services (ServicesConfig): Nastavení ÚFAL localhost služeb (V3).
        graph (GraphConfig): Nastavení faktového grafu.
    """
    data: DataConfig = field(default_factory=DataConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    answerer: AnswererConfig = field(default_factory=AnswererConfig)
    qagen: QagenConfig = field(default_factory=QagenConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
