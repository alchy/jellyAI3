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
from jellyai.graph.graph import build_graph, FactGraph
from jellyai.graph.extract import extract_facts, Fact, Participant
from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.answerer.question import analyze_question
from jellyai.answerer.composer import TemplateComposer
from jellyai.explain import explain_block as explain, list_blocks
from jellyai.ports import (Tokenizer, QuestionAnalyzer, FactExtractor, Composer,
                           CorpusPort, GraphView)
from jellyai.errors import JellyError
from jellyai.corpus import CorpusTools
from jellyai.facade import Jelly
from jellyai.demo import demo
from jellyai.tasks import (annotate_corpus, build_fact_graph, load_fact_graph,
                           make_graph_answerer)
from jellyai.graph.spread import spread_field, heat_landscape, entity_candidates
from jellyai.graph.recover import recover_entities
from jellyai.viz.reflect import reflect
from jellyai.viz.detail import node_detail_rows, fact_detail_rows
from jellyai.viz.pulse import TracePulse
from jellyai.viz.viewbase_view import ViewBaseView

__version__ = "0.1.0"

__all__ = [
    "load_documents", "Document", "tokenize", "split_sentences", "chunk", "Passage",
    "Retriever", "build_graph", "FactGraph", "extract_facts", "Fact", "Participant",
    "Answer", "Answerer", "ExtractiveAnswerer", "GraphAnswerer", "analyze_question",
    "explain", "list_blocks",
    "Tokenizer", "QuestionAnalyzer", "FactExtractor", "Composer", "CorpusPort", "GraphView",
    "JellyError", "CorpusTools", "TemplateComposer", "Jelly", "demo",
    "annotate_corpus", "build_fact_graph", "load_fact_graph", "make_graph_answerer",
    "reflect", "ViewBaseView", "node_detail_rows", "fact_detail_rows", "TracePulse",
    "spread_field", "heat_landscape", "entity_candidates", "recover_entities",
]
