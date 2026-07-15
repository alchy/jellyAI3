from dataclasses import dataclass, field


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    # seznam (url, filename) public-domain knih; ověřeno při plnění dat
    books: list = field(default_factory=list)


@dataclass
class ChunkerConfig:
    size: int = 3            # počet vět na pasáž
    overlap: int = 1         # počet vět sdílených mezi sousedními pasážemi
    unit: str = "sentences"  # "sentences" | "chars"


@dataclass
class RetrieverConfig:
    method: str = "bm25"     # "bm25" | "tfidf"
    top_k: int = 5
    k1: float = 1.5          # BM25 parametr
    b: float = 0.75          # BM25 parametr
    use_stopwords: bool = True


@dataclass
class AnswererConfig:
    max_sentences: int = 1
    template: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    answerer: AnswererConfig = field(default_factory=AnswererConfig)
