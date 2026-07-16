from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.graph import FactGraph, build_graph


def _born(year):
    return make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                 Participant("num", year, "number")])


def test_fact_weight_aggregates_repetition():
    g = FactGraph()
    for _ in range(3):
        g.add_fact(_born("1890"))
    g.add_fact(_born("1915"))
    born90 = [f for f in g.facts.values()
              if ("num", "1890", "number") in
              [(p.role, p.node, p.type) for p in f.participants]][0]
    assert born90.weight == 3
    facts = g.facts_of("Čapek", role="subj", predicate="narodit")
    assert len(facts) == 2
    assert g.participants(born90, "num") == ["1890"]


def test_build_graph_from_annotations():
    ann = {("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                      "sentences": [[
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]]}}
    g = build_graph(ann)
    assert g.facts_of("Božena Němcová", role="subj", predicate="napsat")


def test_graph_save_load_roundtrip(tmp_path):
    g = FactGraph()
    g.add_fact(_born("1890"))
    path = str(tmp_path / "graph.pkl")
    g.save(path)
    loaded = FactGraph.load(path)
    assert list(loaded.facts.keys()) == list(g.facts.keys())
    assert loaded.facts_of("Čapek", predicate="narodit")[0].weight == 1


def test_prodrop_follows_active_subject():
    # Karel byl spisovatel. Narodil se 1890. Josef byl malíř. Zemřel 1945.
    # → narození patří Karlovi, smrt Josefovi (aktivace sleduje aktuální subjekt)
    A = {
        ("d", 0): {"entities": [{"text": "Karel Čapek", "type": "P", "start": 0, "end": 11}],
                   "sentences": [[
            {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
            {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 12, "end": 15},
            {"form": "spisovatel", "lemma": "spisovatel", "upos": "NOUN", "head": 0, "deprel": "root", "start": 16, "end": 26},
        ]]},
        ("d", 1): {"entities": [], "sentences": [[
            {"form": "Narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 7},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 1, "deprel": "expl", "start": 8, "end": 10},
            {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 1, "deprel": "obl", "start": 15, "end": 19},
        ]]},
        ("d", 2): {"entities": [{"text": "Josef Čapek", "type": "P", "start": 0, "end": 11}],
                   "sentences": [[
            {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
            {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 12, "end": 15},
            {"form": "malíř", "lemma": "malíř", "upos": "NOUN", "head": 0, "deprel": "root", "start": 16, "end": 21},
        ]]},
        ("d", 3): {"entities": [], "sentences": [[
            {"form": "Zemřel", "lemma": "zemřít", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 6},
            {"form": "1945", "lemma": "1945", "upos": "NUM", "head": 1, "deprel": "obl", "start": 7, "end": 11},
        ]]},
    }
    g = build_graph(A)
    assert g.facts_of("Karel Čapek", role="subj", predicate="narodit")
    assert g.facts_of("Josef Čapek", role="subj", predicate="zemřít")
    assert not g.facts_of("Karel Čapek", role="subj", predicate="zemřít")


def test_person_canonicalization_unifies_prodrop():
    # #0: 'Karel Čapek byl spisovatel' (entity i fragment 'Karel'); #1 pro-drop narození
    A = {
        ("d", 0): {"entities": [{"text": "Karel Čapek", "type": "P", "start": 0, "end": 11},
                                {"text": "Karel", "type": "P", "start": 0, "end": 5}],
                   "sentences": [[
            {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
            {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 12, "end": 15},
            {"form": "spisovatel", "lemma": "spisovatel", "upos": "NOUN", "head": 0, "deprel": "root", "start": 16, "end": 26},
        ]]},
        ("d", 1): {"entities": [{"text": "Praze", "type": "G", "start": 15, "end": 20}],
                   "sentences": [[
            {"form": "Narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 7},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 1, "deprel": "expl", "start": 8, "end": 10},
            {"form": "v", "lemma": "v", "upos": "ADP", "head": 4, "deprel": "case", "start": 11, "end": 12},
            {"form": "Praze", "lemma": "Praha", "upos": "PROPN", "head": 1, "deprel": "obl", "start": 15, "end": 20},
        ]]},
    }
    g = build_graph(A)
    assert g.facts_of("Karel Čapek", role="subj", predicate="narodit")   # sjednoceno
    assert not g.facts_of("Karel", role="subj", predicate="narodit")     # fragment nezůstal


def test_build_graph_merges_case_variants_across_documents():
    # d1: nominativ (malovat); d2: genitivní zmínka (bratr-relace) → po buildu 1 uzel
    A = {
        ("d1", 0): {"entities": [{"text": "Josef Čapek", "type": "P", "start": 0, "end": 11}],
                    "sentences": [[
            {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
            {"form": "maloval", "lemma": "malovat", "upos": "VERB", "head": 0, "deprel": "root", "start": 12, "end": 19},
            {"form": "obrazy", "lemma": "obraz", "upos": "NOUN", "head": 3, "deprel": "obj", "start": 20, "end": 26},
        ]]},
        ("d2", 0): {"entities": [{"text": "Karel", "type": "P", "start": 0, "end": 5},
                                 {"text": "Josefa Čapka", "type": "P", "start": 16, "end": 28}],
                    "sentences": [[
            {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 6, "end": 9},
            {"form": "bratr", "lemma": "bratr", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 15},
            {"form": "Josefa", "lemma": "Josef", "upos": "PROPN", "head": 3, "deprel": "nmod", "start": 16, "end": 22},
            {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat", "start": 23, "end": 28},
        ]]},
    }
    g = build_graph(A)
    assert "Josefa Čapka" not in g.nodes
    bratr = g.facts_of("Josef Čapek", role="obj", predicate="bratr")
    assert bratr and g.participants(bratr[0], "subj") == ["Karel"]


def test_context_association_binds_entities_to_hot_subject():
    """Role ③ aktivačního pole: bezslovesná zmínka entity (bibliografický řádek)
    se váže faktem „kontext" na aktuální subjekt dokumentu; fragmenty entit se
    neasociují. Kontextové porozumění strukturou, ne SELECT vzorem."""
    A = {
        ("d", 0): {"entities": [{"text": "Karel Čapek", "type": "P", "start": 0, "end": 11}],
                   "sentences": [[
            {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
            {"form": "psal", "lemma": "psát", "upos": "VERB", "head": 0, "deprel": "root", "start": 12, "end": 16},
            {"form": "knihy", "lemma": "kniha", "upos": "NOUN", "head": 3, "deprel": "obj", "start": 17, "end": 22},
        ]]},
        ("d", 1): {"entities": [{"text": "R.U.R.", "type": "P", "start": 0, "end": 6},
                                {"text": "R.", "type": "pf", "start": 0, "end": 2}],
                   "sentences": [[
            {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 0, "deprel": "root", "start": 0, "end": 6},
        ]]},
    }
    g = build_graph(A)
    kontext = g.facts_of("R.U.R.", role="obj", predicate="kontext")
    assert kontext and g.participants(kontext[0], "subj") == ["Karel Čapek"]
    assert not g.facts_of("R.", predicate="kontext")     # fragment ne


def _capek_birth_annotations():
    return {("d", 0): {"entities": [{"text": "Karel Čapek", "type": "P", "start": 0, "end": 11},
                                    {"text": "13. ledna 1890", "type": "T", "start": 24, "end": 37}],
                       "sentences": [[
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 12, "end": 19},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 20, "end": 22},
        {"form": "13", "lemma": "13", "upos": "NUM", "head": 6, "deprel": "nummod", "start": 24, "end": 26},
        {"form": "ledna", "lemma": "leden", "upos": "NOUN", "head": 3, "deprel": "obl", "start": 27, "end": 32},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 6, "deprel": "nummod", "start": 33, "end": 37},
    ]]}}


def test_date_decomposition_subfacts():
    g = build_graph(_capek_birth_annotations())
    # narození má celé datum jako hodnotu
    born = g.facts_of("Karel Čapek", role="subj", predicate="narodit")
    assert born and "13. ledna 1890" in g.participants(born[0], "time")
    # datum je uzel s pod-faktem rok → 1890 (zanoření v grafu)
    rok = g.facts_of("13. ledna 1890", role="subj", predicate="rok")
    assert rok and g.participants(rok[0], "val") == ["1890"]


def test_concept_subject_gets_context_binding():
    """Konceptový podmět faktu („rodina se ocitla…") se váže na aktuální osobu
    dokumentu — slovo „rodina" musí mít vazby na další elementy."""
    A = {
        ("d", 0): {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
                   "sentences": [[
            {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
            {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
            {"form": "psala", "lemma": "psát", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 20},
            {"form": "knihy", "lemma": "kniha", "upos": "NOUN", "head": 3, "deprel": "obj", "start": 21, "end": 26},
        ]]},
        ("d", 1): {"entities": [], "sentences": [[
            {"form": "Rodina", "lemma": "rodina", "upos": "NOUN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
            {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl", "start": 7, "end": 9},
            {"form": "ocitla", "lemma": "ocitnout", "upos": "VERB", "head": 0, "deprel": "root", "start": 10, "end": 16},
            {"form": "v", "lemma": "v", "upos": "ADP", "head": 5, "deprel": "case", "start": 17, "end": 18},
            {"form": "Domažlicích", "lemma": "Domažlice", "upos": "PROPN", "head": 3, "deprel": "obl", "start": 19, "end": 30},
        ]]},
    }
    g = build_graph(A)
    kontext = g.facts_of("rodina", role="obj", predicate="kontext")
    assert kontext and g.participants(kontext[0], "subj") == ["Božena Němcová"]
