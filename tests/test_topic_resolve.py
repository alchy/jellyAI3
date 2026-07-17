"""Case-insensitive rozřešení tématu — kapitalizovaný uzel („Vějíř") jde dotázat,
i když UDPipe lemmatizuje na malé („vějíř")."""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _answerer(graph):
    return GraphAnswerer(graph, FakeUfalClient(), ExtractiveAnswerer(AnswererConfig()))


def test_term_coverage_beats_single_surface_hit():
    """Dva kmenové hity („Karla"+„Čapka") přebijí jeden povrchový hit („Čapka"
    v uzlu „Antonína Čapka") — pokrytí termů je primární patro."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Antonína Čapka", "person"),
                                 Participant("pred", "otec", "concept")]))
    g.add_fact(make_fact("bratr", [Participant("subj", "Josef Čapek", "person"),
                                   Participant("obj", "Karel Antonín Čapek", "person")]))
    a = _answerer(g)
    assert a._resolve_topic(["Karla", "Čapka"]) == "Karel Antonín Čapek"


def test_same_cluster_prefers_predicate_affinity():
    """Zbytkový skloněný uzel „Babičku" (exact hit) nesmí přebít variantu
    „Babička", o níž se predikát otázky dá vypovědět — v rámci TÉHOŽ
    kmenového clusteru rozhoduje afinita."""
    g = FactGraph()
    g.add_fact(make_fact("kontext", [Participant("subj", "Babičku", "concept"),
                                     Participant("obj", "kraj", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "dílo")]))
    a = _answerer(g)
    assert a._resolve_topic(["Babičku"], "napsat") == "Babička"


def test_loose_stem_reaches_short_inflected_concept():
    """„hru"→„hra": min_stem=3 blokuje kmen, nejvolnější patro (oboustranné
    seříznutí koncové samohlásky) dosáhne."""
    g = FactGraph()
    g.add_fact(make_fact("druh", [Participant("subj", "R.U.R.", "dílo"),
                                  Participant("pred", "hra", "concept")]))
    a = _answerer(g)
    assert a._resolve_topic(["hru"]) == "hra"


def test_function_words_do_not_score():
    """„s" nesmí skórovat: „Válku s mloky" ≠ „Hovory s TGM"."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "Hovory s TGM", "dílo")]))
    a = _answerer(g)
    assert a._resolve_topic(["Válku", "s", "mloky"]) is None


def test_lowercase_lemma_resolves_capitalized_title():
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Jaroslav Seifert", "person"),
                                    Participant("obj", "Vějíř", "dílo")]))
    q = "Kdo napsal Vějíř?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Vějíř", "lemma": "vějíř", "upos": "NOUN", "head": 2, "deprel": "obj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    ans = a.answer(q, [])
    assert ans.text == "Jaroslav Seifert"          # dřív selhalo (case-sensitive)


def test_case_variant_term_resolves_by_stem():
    """Skloněný termín („Galéna" — lemma, které UDPipe nechá v pádu) najde
    kanonický uzel „Galén" kmenovým fallbackem — týž mechanismus jako
    build-side resolver (canon._stem), takže se query a build nerozejdou."""
    g = FactGraph()
    g.add_fact(make_fact("léčit", [Participant("subj", "Galén", "person"),
                                   Participant("obj", "malomocenství", "concept")]))
    q = "Co léčil Galéna?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
        {"form": "léčil", "lemma": "léčit", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Galéna", "lemma": "Galéna", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "malomocenství"


def test_exact_case_still_distinguishes_book_from_common():
    """Zachovaná velikost (PROPN lemma „Babička") drží knihu odlišenou od „babička"."""
    g = FactGraph()
    for _ in range(9):     # obecná „babička" hodně častá
        g.add_fact(make_fact("péct", [Participant("subj", "babička", "concept"),
                                      Participant("obj", "povídka", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")]))
    q = "kdo napsal Babičku?"
    client = FakeUfalClient(parse={q: [[
        {"form": "kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 2, "deprel": "obj"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Božena Němcová"    # přesná shoda „Babička" > častá „babička"


def test_diacritic_insensitive_resolution():
    """Dotaz „cestinou" bez diakritiky se přes fold trefí do uzlu s diakritikou
    („Jezis"→„Ježíš", „capek"→„Čapek") — nese kontext přes aktivaci."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Ježíš", "person"),
                                 Participant("pred", "prorok", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    q = "Kdo je Jezis?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "Jezis", "lemma": "Jezis", "upos": "PROPN", "head": 0, "deprel": "root"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "prorok"
