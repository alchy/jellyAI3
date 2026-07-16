def test_public_api_exposes_blocks():
    import jellyai
    for name in ["load_documents", "chunk", "tokenize", "split_sentences",
                 "Retriever", "build_fact_graph", "FactGraph", "extract_facts",
                 "Fact", "Participant", "ExtractiveAnswerer", "GraphAnswerer",
                 "analyze_question", "Answer", "explain"]:
        assert hasattr(jellyai, name), f"chybí veřejný symbol {name}"
