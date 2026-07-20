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
    from jellyai.graph.hygiene import form_case_votes, scrub_entities
    dropped_e = scrub_entities(annotations, form_case_votes(annotations))
    print(f"Hygiena entit: −{dropped_e} osobních slepenců s obecnými slovy")
    graph = build_graph(annotations)
    from jellyai.graph.recover import recover_entities
    recover_entities(annotations, graph)      # role ②: doplnit tituly, co NER minul
    resolve_entities(graph)   # recover bere podměty ze surového povrchu → srovnat znovu
    from jellyai.graph.hygiene import lemma_upos_votes, scrub
    dropped_p, dropped_f = scrub(graph, lemma_upos_votes(annotations))
    print(f"Hygiena: −{dropped_p} mis-tag účastníků, −{dropped_f} faktů "
          f"(korpusová evidence lemmat)")
    from jellyai.graph.hygiene import noun_animacy_votes, scrub_semantics
    dropped_s = scrub_semantics(graph, noun_animacy_votes(annotations))
    print(f"Hygiena sémantiky: −{dropped_s} faktů "
          f"(osoba pod neživotným druhem, vztah bez protistrany)")
    from jellyai.graph.hygiene import name_position_votes, scrub_false_persons
    dropped_i = scrub_false_persons(graph, name_position_votes(annotations))
    print(f"Hygiena jmen: −{dropped_i} falešných osob z imperativních "
          f"začátků vět (Tyč, Proste — dávka D)")
    from jellyai.graph.hygiene import nominativize, propn_lemma_votes
    renamed = nominativize(graph, propn_lemma_votes(annotations))
    print(f"Nominativizace: {renamed} skloněných id → lemma "
          f"(Betlémě→Betlém, Boha→Bůh)")
    from jellyai.graph.instance import resolve_instances
    merged = resolve_instances(graph)
    print(f"Instance: {merged} jmenných střepů srostlo (tvrzené aliasy), "
          f"{len(graph.name_families)} jmenných rodin")
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
                         spread_falloff=config.graph.spread_falloff,
                         context_hub_limit=getattr(
                             config.graph, "context_hub_limit", 50))
