"""Sestavení celého QA datasetu z korpusu.

Spojí V1 bloky (loader → chunker → věty) s qagen logikou (výběr odpovědí →
šablona otázky) a zapíše výsledek jako JSONL — jeden pár na řádek. Kontextem
každého páru je celá pasáž, aby měl budoucí generátor z čeho čerpat. Duplicity
se odfiltrují, ať dataset není zaplavený týmiž otázkami (korpus rád opakuje).
"""

import json
import os

from jellyai.loader import load_documents
from jellyai.chunker import chunk
from jellyai.text import split_sentences
from qagen.answers import candidates
from qagen.questions import build_question


def build_dataset(config, tagger):
    """Vygeneruje QA páry z korpusu a zapíše je do JSONL.

    Projde dokumenty → pasáže → věty; na každou dost dlouhou větu pustí tagger,
    vybere odpovědi a poskládá otázky. Výsledné páry zapíše na `config.qagen.qa_path`.

    Args:
        config (Config): Konfigurace (processed_dir, chunker, qagen).
        tagger (Tagger): Analyzátor entit/tokenů (UfalTagger nebo FakeTagger).

    Returns:
        list[dict]: Páry {question, context, answer, type, doc_id, passage_index}.
    """
    docs = load_documents(config.data.processed_dir)
    directory = os.path.dirname(config.qagen.qa_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    pairs = []
    seen = set()
    for doc in docs:
        for passage in chunk(doc, config.chunker):
            for sentence in split_sentences(passage.text):
                if len(sentence.split()) < config.qagen.min_tokens:
                    continue
                for cand in candidates(sentence, tagger, config.qagen):
                    question = build_question(sentence, cand)
                    key = (question, cand.answer, doc.doc_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({
                        "question": question,
                        "context": passage.text,
                        "answer": cand.answer,
                        "type": cand.qtype,
                        "doc_id": doc.doc_id,
                        "passage_index": passage.index,
                    })

    with open(config.qagen.qa_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return pairs
