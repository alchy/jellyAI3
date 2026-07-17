"""Hygiena grafu — korpusová evidence lemmat proti mis-tagům.

Jedna věta občas projde se špatnou značkou („hoď" jako NOUN → účastník
„hodit"; „Izaiáš" jako VERB → predikát). CELÝ korpus ale ví lépe: hlasování
upos přes všechny výskyty lemmatu odhalí, čím slovo převážně je. Účastník
s převahou slovesných hlasů není entita; predikát s převahou jmenných hlasů
není děj. Bez hlasů (paměť Mnemos, řídká slova) se nesoudí.
"""

from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.graph.hygiene import lemma_upos_votes, scrub


def _annotations(tokens_by_sentence):
    """Anotace z [(lemma, upos), …] vět (minimum pro hlasování)."""
    return {("doc", i): {"sentences": [[{"lemma": l, "upos": u, "form": l,
                                         "head": 0, "deprel": "dep"}
                                        for l, u in sent]]}
            for i, sent in enumerate(tokens_by_sentence)}


def test_votes_count_upos_per_lemma():
    votes = lemma_upos_votes(_annotations([
        [("hodit", "VERB"), ("koleno", "NOUN")],
        [("hodit", "VERB"), ("hodit", "VERB"), ("koleno", "NOUN")],
    ]))
    assert votes["hodit"]["VERB"] == 3
    assert votes["koleno"]["NOUN"] == 2


def test_scrub_drops_verb_dominant_participant():
    """vzít(Mojžíš, obj=hodit) → „hodit" je v korpusu sloveso → účastník
    pryč; fakt zůstává (má ještě podmět a předmět)."""
    g = FactGraph()
    g.add_fact(make_fact("vzít", [Participant("subj", "Mojžíš", "person"),
                                  Participant("obj", "hůl", "concept"),
                                  Participant("obj", "hodit", "concept")]))
    votes = lemma_upos_votes(_annotations([[("hodit", "VERB")] * 3]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_participants == 1 and dropped_facts == 0
    nodes = {p.node for f in g.facts.values() for p in f.participants}
    assert "hodit" not in nodes and "hůl" in nodes


def test_scrub_keeps_noun_dominant_and_unvoted():
    """„koleno" (NOUN hlasy) i „knedlíky" (bez hlasů — paměť) zůstávají."""
    g = FactGraph()
    g.add_fact(make_fact("bolet", [Participant("subj", "koleno", "concept"),
                                   Participant("obj", "knedlíky", "concept")]))
    votes = lemma_upos_votes(_annotations([[("koleno", "NOUN")] * 3]))
    scrub(g, votes)
    nodes = {p.node for f in g.facts.values() for p in f.participants}
    assert "koleno" in nodes and "knedlíky" in nodes


def test_scrub_drops_fact_with_name_dominant_predicate():
    """Predikát „Izaiáš" (v korpusu PROPN) není děj → celý fakt pryč;
    legitimní slovesný predikát zůstává."""
    g = FactGraph()
    g.add_fact(make_fact("Izaiáš", [Participant("subj", "prorok", "concept"),
                                    Participant("obj", "svitek", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Izaiáš", "person"),
                                    Participant("obj", "svitek", "concept")]))
    votes = lemma_upos_votes(_annotations([
        [("Izaiáš", "PROPN")] * 3 + [("napsat", "VERB")] * 3]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_facts == 1
    predicates = {f.predicate for f in g.facts.values()}
    assert "Izaiáš" not in predicates and "napsat" in predicates


def test_structural_predicates_survive_noun_votes():
    """Reifikované vztahy (bratr), druh, kontext i dekompozice dat (rok)
    jsou JMENNÉ predikáty ZÁMĚRNĚ — hlasování je nesmí smazat."""
    g = FactGraph()
    g.add_fact(make_fact("bratr", [Participant("subj", "Josef", "person"),
                                   Participant("obj", "Karel", "person")]))
    g.add_fact(make_fact("rok", [Participant("subj", "2. května 1818", "time"),
                                 Participant("val", "1818", "number")]))
    votes = lemma_upos_votes(_annotations([
        [("bratr", "NOUN")] * 5 + [("rok", "NOUN")] * 5]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_facts == 0
    assert {f.predicate for f in g.facts.values()} == {"bratr", "rok"}


def test_entity_scrub_drops_lowercase_glued_person():
    """NameTag kontejner „Abraham chléb" (chléb mis-tagnutý jako příjmení):
    slovo v korpusu převážně malé není součást jména → entita pryč; čisté
    „Abraham" (a geo entity) zůstávají."""
    from jellyai.graph.hygiene import form_case_votes, scrub_entities
    annotations = {("doc", 0): {"sentences": [[
        {"form": "Abraham", "lemma": "Abraham", "upos": "PROPN"},
        {"form": "vzal", "lemma": "vzít", "upos": "VERB"},
        {"form": "chléb", "lemma": "chléb", "upos": "NOUN"},
        {"form": "chléb", "lemma": "chléb", "upos": "NOUN"},
        {"form": "chléb", "lemma": "chléb", "upos": "NOUN"},
    ]], "entities": [
        {"text": "Abraham chléb", "type": "P", "start": 0, "end": 13},
        {"text": "Abraham", "type": "pf", "start": 0, "end": 7},
        {"text": "chléb", "type": "ps", "start": 8, "end": 13},
        {"text": "Betlém", "type": "gu", "start": 20, "end": 26},
    ]}}
    votes = form_case_votes(annotations)
    dropped = scrub_entities(annotations, votes)
    kept = [e["text"] for e in annotations[("doc", 0)]["entities"]]
    assert dropped == 2                      # kontejner + falešné příjmení
    assert kept == ["Abraham", "Betlém"]


def test_entity_scrub_drops_case_disagreeing_name():
    """NameTag kontejner „Ježíš Martu" (Jan 11,5 — akuzativní Marta jako
    „příjmení" nominativního Ježíše): české víceslovné jméno se skloňuje
    VE SHODĚ, pádově neshodné PROPN členy jsou dva větní účastníci, ne
    jméno (jazykové pravidlo `name_case_agreement`). Členy kontejneru
    (skutečné osoby věty) zůstávají."""
    from jellyai.graph.hygiene import form_case_votes, scrub_entities
    annotations = {("doc", 0): {"sentences": [[
        {"form": "Ježíš", "upos": "PROPN", "start": 0, "end": 5,
         "feats": {"Case": "Nom"}},
        {"form": "Martu", "upos": "PROPN", "start": 6, "end": 11,
         "feats": {"Case": "Acc"}},
        {"form": "miloval", "upos": "VERB", "start": 12, "end": 19,
         "feats": {}},
    ]], "entities": [
        {"text": "Ježíš Martu", "type": "P", "start": 0, "end": 11},
        {"text": "Ježíš", "type": "pf", "start": 0, "end": 5},
        {"text": "Martu", "type": "ps", "start": 6, "end": 11},
    ]}}
    dropped = scrub_entities(annotations, form_case_votes(annotations))
    kept = [e["text"] for e in annotations[("doc", 0)]["entities"]]
    assert dropped == 1                      # jen kontejner
    assert kept == ["Ježíš", "Martu"]


def test_entity_scrub_case_rule_survives_local_mistag():
    """Reálný tvar chyby z Jan 11,5: UDPipe v TÉ větě otagoval „Ježíš" jako
    VERB (bez pádu), takže lokální neshoda nevznikne — ale 3+ nominativní
    výskyty tvaru jinde v korpusu vědí lépe (korpusový fallback pádu)."""
    from jellyai.graph.hygiene import form_case_votes, scrub_entities
    elsewhere = [[{"form": "Ježíš", "upos": "PROPN", "start": 100 + i,
                   "end": 105 + i, "feats": {"Case": "Nom"}}]
                 for i in range(3)]
    annotations = {("doc", 0): {"sentences": [[
        {"form": "Ježíš", "upos": "VERB", "start": 0, "end": 5,
         "feats": {"Mood": "Ind", "Person": "2"}},      # mis-tag, žádný pád
        {"form": "Martu", "upos": "PROPN", "start": 6, "end": 11,
         "feats": {"Case": "Acc"}},
    ]], "entities": [
        {"text": "Ježíš Martu", "type": "P", "start": 0, "end": 11},
        {"text": "Ježíš", "type": "pf", "start": 0, "end": 5},
        {"text": "Martu", "type": "ps", "start": 6, "end": 11},
    ]}}
    annotations.update({("doc", i + 1): {"sentences": [s], "entities": []}
                        for i, s in enumerate(elsewhere)})
    dropped = scrub_entities(annotations, form_case_votes(annotations))
    kept = [e["text"] for e in annotations[("doc", 0)]["entities"]]
    assert dropped == 1
    assert kept == ["Ježíš", "Martu"]


def test_entity_scrub_keeps_agreeing_and_uncertain_names():
    """Skloněné jméno ve shodě („Karla Čapka" Gen+Gen) zůstává; člen bez
    jistého pádu (nesklonné příjmení bez feats, víceznačné „Acc,Nom")
    neshodu nezakládá — soudí se jen dva JISTÉ různé pády."""
    from jellyai.graph.hygiene import form_case_votes, scrub_entities
    annotations = {("doc", 0): {"sentences": [[
        {"form": "Karla", "upos": "PROPN", "start": 0, "end": 5,
         "feats": {"Case": "Gen"}},
        {"form": "Čapka", "upos": "PROPN", "start": 6, "end": 11,
         "feats": {"Case": "Gen"}},
        {"form": "Marii", "upos": "PROPN", "start": 20, "end": 25,
         "feats": {"Case": "Dat"}},
        {"form": "Curie", "upos": "PROPN", "start": 26, "end": 31,
         "feats": {}},
    ]], "entities": [
        {"text": "Karla Čapka", "type": "P", "start": 0, "end": 11},
        {"text": "Marii Curie", "type": "P", "start": 20, "end": 31},
    ]}}
    dropped = scrub_entities(annotations, form_case_votes(annotations))
    kept = [e["text"] for e in annotations[("doc", 0)]["entities"]]
    assert dropped == 0
    assert kept == ["Karla Čapka", "Marii Curie"]


def _animacy(annotations_spec):
    """Fake anotace pro hlasování životnosti: [(form, lemma, animacy), …]."""
    from jellyai.graph.hygiene import noun_animacy_votes
    sent = [{"form": f, "lemma": l, "upos": "NOUN",
             "feats": {"Gender": "Masc", "Animacy": a}}
            for f, l, a in annotations_spec]
    return noun_animacy_votes({("doc", 0): {"sentences": [sent]}})


def test_semantics_drops_person_under_inanimate_kind():
    """druh(Dorothea, vztah) — osoba nemůže být instancí neživotné věci
    („Ze vztahu Dorothey…": nesklonné jméno bez pádu proklouzlo pádovým
    guardům; životnost je druhá korpusová evidence)."""
    from jellyai.graph.hygiene import scrub_semantics
    g = FactGraph()
    g.add_fact(make_fact("druh", [Participant("subj", "Dorothea", "person"),
                                  Participant("pred", "vztah", "concept")]))
    g.add_fact(make_fact("druh", [Participant("subj", "Dorothea", "person"),
                                  Participant("pred", "hraběnka", "concept")]))
    votes = _animacy([("vztahu", "vztah", "Inan")] * 3)
    assert scrub_semantics(g, votes) == 1
    kept = {p.node for f in g.facts.values() for p in f.participants}
    assert "hraběnka" in kept and "vztah" not in kept


def test_semantics_keeps_mistyped_inanimate_subject():
    """druh(Týdenník, časopis) — subjekt mylně typovaný person, ale sám
    hlasuje neživotně: obě strany neživotné = konzistentní zařazení, drží."""
    from jellyai.graph.hygiene import scrub_semantics
    g = FactGraph()
    g.add_fact(make_fact("druh", [Participant("subj", "Týdenník", "person"),
                                  Participant("pred", "časopis", "concept")]))
    votes = _animacy([("časopis", "časopis", "Inan")] * 3
                     + [("týdenník", "týdenník", "Inan")] * 3)
    assert scrub_semantics(g, votes) == 0


def test_semantics_drops_relation_without_object():
    """Reifikovaný vztah („žena") bez obj protistrany je troska parseru —
    fakt jen s časy nic nenese; vztah s obj zůstává."""
    from jellyai.graph.hygiene import scrub_semantics
    g = FactGraph()
    g.add_fact(make_fact("žena", [Participant("subj", "českým", "concept"),
                                  Participant("time", "1844", "time")]))
    g.add_fact(make_fact("bratr", [Participant("subj", "Karel Čapek", "person"),
                                   Participant("obj", "Josef Čapek", "person")]))
    assert scrub_semantics(g, {}) == 1
    assert {f.predicate for f in g.facts.values()} == {"bratr"}


def test_case_fallback_two_unanimous_votes():
    """Pád tvaru je morfologie — 2 jednomyslné hlasy stačí: „Masaryka"
    (PROPN Gen×2) proti lokálním mis-tagům NOUN Nom rozbije slepenec
    „Masaryka Svatopluk Beneš" (fragment popisku)."""
    from jellyai.graph.hygiene import form_case_votes, scrub_entities
    gen = [[{"form": "Masaryka", "upos": "PROPN", "start": 100 + i,
             "end": 108 + i, "feats": {"Case": "Gen"}}] for i in range(2)]
    annotations = {("doc", 0): {"sentences": [[
        {"form": "Masaryka", "upos": "NOUN", "start": 0, "end": 8,
         "feats": {"Case": "Nom"}},               # mis-tag bez důvěry
        {"form": "Svatopluk", "upos": "PROPN", "start": 9, "end": 18,
         "feats": {"Case": "Nom"}},
        {"form": "Beneš", "upos": "PROPN", "start": 19, "end": 24,
         "feats": {"Case": "Nom"}},
    ]], "entities": [
        {"text": "Masaryka Svatopluk Beneš", "type": "P", "start": 0,
         "end": 24},
        {"text": "Svatopluk Beneš", "type": "P", "start": 9, "end": 24},
    ]}}
    annotations.update({("doc", i + 1): {"sentences": [s], "entities": []}
                        for i, s in enumerate(gen)})
    dropped = scrub_entities(annotations, form_case_votes(annotations))
    kept = [e["text"] for e in annotations[("doc", 0)]["entities"]]
    assert dropped == 1
    assert kept == ["Svatopluk Beneš"]


def test_clean_lemma_strips_glued_quotes():
    """Tokenizace občas přilepí uvozovku k lemmatu („hřích“") — uzel grafu
    nesmí nést interpunkci jména."""
    from jellyai.answerer.selection import _clean_lemma
    assert _clean_lemma("hřích“") == "hřích"
    assert _clean_lemma("„tábor") == "tábor"
    assert _clean_lemma("R.U.R.") == "R.U.R."


def test_scrub_drops_fact_left_without_partner():
    """Když po vyřazení účastníka zbude faktu jediný účastník, fakt padá
    (fakt bez protistrany nic nenese)."""
    g = FactGraph()
    g.add_fact(make_fact("nechat", [Participant("subj", "mluvit", "concept"),
                                    Participant("obj", "džbán", "concept")]))
    votes = lemma_upos_votes(_annotations([[("mluvit", "VERB")] * 3]))
    dropped_participants, dropped_facts = scrub(g, votes)
    assert dropped_facts == 1
    assert not g.facts
