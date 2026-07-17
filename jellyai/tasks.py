"""Orchestrace pipeline jako knihovní funkce — sdílí je CLI, web i vlastní programy.

Dřív tahle logika žila v tělech CLI příkazů; teď je v knihovně, takže terminál i web
volají totéž (žádná duplikace). Import ÚFAL klienta je líný — funkce nad hotovými
anotacemi (build/load grafu) jdou použít bez modelů.
"""

# funkce lazy-importují ÚFAL/answerer (optional deps, rychlý start) — záměr
# pylint: disable=import-outside-toplevel
from jellyai.loader import load_documents
from jellyai.annotate import annotate_documents, save_annotations, load_annotations
from jellyai.graph.graph import build_graph, resolve_entities, FactGraph
from jellyai.lang import set_language


def annotate_corpus(config, client=None):
    """Anotuje dokumenty po větách a uloží (entity + syntaktické role).

    Args:
        config (Config): Konfigurace (processed_dir, services).
        client: ÚFAL klient; None = vytvoří `UfalClient` a na konci ho složí.

    Returns:
        int: Počet anotovaných vět.
    """
    documents = load_documents(config.data.processed_dir)
    own = client is None
    if own:
        from jellyai.ufal_client import UfalClient
        client = UfalClient(config.services)
    try:
        annotations = annotate_documents(documents, client)
    finally:
        if own:
            client.close()
    save_annotations(annotations, config.services.annotations_path)
    return len(annotations)


def build_fact_graph(config):
    """Postaví faktový graf z uložených anotací a uloží ho.

    Args:
        config (Config): Konfigurace (annotations_path, graph_path).

    Returns:
        FactGraph: Postavený graf.
    """
    set_language(config.graph.language)       # jazyk kanonizace = zásuvný modul
    annotations = load_annotations(config.services.annotations_path)
    graph = build_graph(annotations)
    from jellyai.graph.recover import recover_entities
    recover_entities(annotations, graph)      # role ②: doplnit tituly, co NER minul
    resolve_entities(graph)   # recover bere podměty ze surového povrchu → srovnat znovu
    graph.save(config.graph.graph_path)
    return graph


def load_fact_graph(config):
    """Načte dříve uložený faktový graf.

    Args:
        config (Config): Konfigurace (graph_path).

    Returns:
        FactGraph: Načtený graf.
    """
    return FactGraph.load(config.graph.graph_path)


def make_graph_answerer(config):
    """Sestaví `GraphAnswerer` nad uloženým grafem (graf + klient + fallback).

    Args:
        config (Config): Konfigurace.

    Returns:
        GraphAnswerer: Připravený answerer.
    """
    from jellyai.answerer.graph_answerer import GraphAnswerer
    from jellyai.answerer.extractive import ExtractiveAnswerer
    from jellyai.ufal_client import UfalClient
    set_language(config.graph.language)       # query-side kmenování týmž jazykem
    graph = load_fact_graph(config)
    return GraphAnswerer(graph, UfalClient(config.services),
                         ExtractiveAnswerer(config.answerer),
                         context_decay=config.graph.context_decay,
                         spread_depth=config.graph.spread_depth,
                         spread_falloff=config.graph.spread_falloff)
