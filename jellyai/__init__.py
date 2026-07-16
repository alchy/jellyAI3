"""jellyAI3 — výuková knihovna pro české QA nad texty.

Skládačka malých bloků (portů): retrieval, faktový graf, answerery, korpusové
nástroje. Píšeš program nad `import jellyai` a skládáš granulární funkce; kdo chce
rychlý výsledek, použije fasádu `Jelly`; kdo chce první „ono to funguje", zavolá
`demo()`. Bohaté české docstringy u každého bloku.
"""

from jellyai.loader import load_documents, Document
from jellyai.text import tokenize, split_sentences
from jellyai.chunker import chunk, Passage
from jellyai.retriever import Retriever
from jellyai.graph.graph import build_graph as build_fact_graph, FactGraph
from jellyai.graph.extract import extract_facts, Fact, Participant
from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.answerer.question import analyze_question
from jellyai.explain import explain_block as explain, list_blocks
from jellyai.ports import (Tokenizer, QuestionAnalyzer, FactExtractor, Composer,
                           CorpusPort)

__version__ = "0.1.0"

__all__ = [
    "load_documents", "Document", "tokenize", "split_sentences", "chunk", "Passage",
    "Retriever", "build_fact_graph", "FactGraph", "extract_facts", "Fact", "Participant",
    "Answer", "Answerer", "ExtractiveAnswerer", "GraphAnswerer", "analyze_question",
    "explain", "list_blocks",
    "Tokenizer", "QuestionAnalyzer", "FactExtractor", "Composer", "CorpusPort",
]
