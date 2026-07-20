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


def test_scrub_drops_adverb_identity():
    """být(Jan, brzy) — příslovce identitou není (mis-parse spony);
    adjektivum v pred/attr naopak zůstává (vlastnost)."""
    g = FactGraph()
    g.add_fact(make_fact("být", [Participant("subj", "Jan", "person"),
                                 Participant("pred", "brzy", "concept")]))
    g.add_fact(make_fact("být", [Participant("subj", "Jan", "person"),
                                 Participant("attr", "hořící", "concept")]))
    votes = lemma_upos_votes(_annotations([[("brzy", "ADV")] * 3
                                           + [("hořící", "ADJ")] * 3]))
    dropped_p, dropped_f = scrub(g, votes)
    assert dropped_f == 1                        # brzy-fakt bez protistrany padá
    kept = {p.node for f in g.facts.values() for p in f.participants}
    assert "hořící" in kept and "brzy" not in kept


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


def _lemma_votes(spec):
    """Fake anotace pro hlasování lemmat: [(form, lemma), …] jako PROPN."""
    from jellyai.graph.hygiene import propn_lemma_votes
    sent = [{"form": f, "lemma": l, "upos": "PROPN"} for f, l in spec]
    return propn_lemma_votes({("doc", 0): {"sentences": [sent]}})


def test_nominativize_merges_inflected_id_into_lemma_node():
    """„Betlémě" (geo) → „Betlém": skloněné id fragmentuje fakty — po
    nominativizaci uzly splynou (součet vah), typ zůstává geo, starý
    povrch je alias. Časový uzel se nedotýká (genitivní formát dat)."""
    from jellyai.graph.hygiene import nominativize
    g = FactGraph()
    g.add_fact(make_fact("narodit", [Participant("subj", "Ježíš", "person"),
                                     Participant("loc", "Betlémě", "geo")]))
    g.add_fact(make_fact("ležet", [Participant("subj", "Betlém", "geo"),
                                   Participant("loc", "Judsko", "geo")]))
    g.add_fact(make_fact("stát", [Participant("subj", "Bůh", "person"),
                                  Participant("time", "17. července 2026",
                                              "time")]))
    votes = _lemma_votes([("Betlémě", "Betlém")] * 2
                         + [("července", "červenec")] * 3)
    assert nominativize(g, votes) == 1
    assert "Betlémě" not in g.nodes
    assert g.nodes["Betlém"].type == "geo"
    assert "17. července 2026" in g.nodes          # čas nedotčen
    assert "Betlémě" in g.aliases.get("Betlém", [])
    locs = {p.node for f in g.facts_of("Ježíš") for p in f.participants
            if p.role == "loc"}
    assert locs == {"Betlém"}


def test_nominativize_capitalizes_lemma_and_respects_dominance():
    """Lemma jména malými písmeny se kapitalizuje („Boha"→bůh→„Bůh",
    „Válku"→„Válka" — jméno zůstává jménem); rozštěpené hlasy (homograf)
    tvar nechají být; zmrzačené krátké lemma („Lea"→„Le") nesoudí."""
    from jellyai.graph.hygiene import nominativize, propn_lemma_votes
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel", "person"),
                                    Participant("obj", "Válku", "dílo")]))
    g.add_fact(make_fact("vidět", [Participant("subj", "Lea", "person"),
                                   Participant("obj", "Boha", "person")]))
    sent = [{"form": "Válku", "lemma": "válka", "upos": "PROPN"}] * 3 \
        + [{"form": "Boha", "lemma": "bůh", "upos": "PROPN"}] * 3 \
        + [{"form": "Lea", "lemma": "Le", "upos": "PROPN"}] * 3 \
        + [{"form": "Karel", "lemma": "Karel", "upos": "PROPN"},
           {"form": "Karel", "lemma": "Karla", "upos": "PROPN"}]
    votes = propn_lemma_votes({("doc", 0): {"sentences": [sent]}})
    assert nominativize(g, votes) == 2
    assert "Válka" in g.nodes and g.nodes["Válka"].type == "dílo"
    assert "Bůh" in g.nodes and g.nodes["Bůh"].type == "person"
    assert "Lea" in g.nodes                        # krátké lemma nesoudí
    assert "Karel" in g.nodes                      # homograf rozštěpen


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


def test_falesna_osoba_z_imperativu_padne():
    """Dávka D (T14 „Tyč"): osoba, jejíž tvary se v korpusu NIKDY nepíší
    velkým písmenem uprostřed věty a mají malopísmenné výskyty (≥2), není
    jméno — je to věcný/imperativní začátek věty. Skutečné jméno s jedním
    výskytem (Verunka: malých 0) přežije."""
    from jellyai.graph.extract import make_fact, Participant
    from jellyai.graph.graph import FactGraph
    from jellyai.graph.hygiene import scrub_false_persons
    g = FactGraph()
    g.add_fact(make_fact("zhotovit", [Participant("subj", "Tyč", "person"),
                                      Participant("obj", "cherub", "concept")]))
    g.add_fact(make_fact("potkat", [Participant("subj", "Verunka", "person"),
                                    Participant("obj", "Barunka", "person")]))
    g.aliases = {"Tyč": ["Tyče"]}
    votes = {"tyč": [0, 1], "tyče": [0, 22],       # [uprostřed velké, malé]
             "verunka": [0, 0], "barunka": [3, 0]}
    dropped = scrub_false_persons(g, votes)
    assert dropped == 1
    assert "Tyč" not in g.nodes
    assert any(p.node == "Verunka" for f in g.facts.values()
               for p in f.participants)


def test_fold_participia_na_slovesny_predikat():
    """Dávka D zbytek (a): predikát z ADJ pasivního participia („pokřtěný")
    se složí na slovesné lemma („pokřtít"), když sloveso v grafu existuje
    (korpusová evidence — tagger VERB dvojníka participia nedává)."""
    from jellyai.graph.hygiene import fold_participles
    g = FactGraph()
    g.add_fact(make_fact("pokřtěný", [Participant("subj", "Jan", "person"),
                                      Participant("obj", "Ježíš", "person")]))
    g.add_fact(make_fact("pokřtít", [Participant("subj", "Pilát", "person"),
                                     Participant("theme", "kniha", "concept")]))
    folded = fold_participles(g)
    assert folded == 1
    preds = {f.predicate for f in g.facts.values()}
    assert "pokřtěný" not in preds
    krest = next(f for f in g.facts.values()
                 if ("obj", "Ježíš") in {(p.role, p.node)
                                         for p in f.participants})
    assert krest.predicate == "pokřtít"


def test_fold_bez_slovesneho_protejsku_zustava():
    """Participium bez slovesného protějšku v grafu poctivě zůstává
    adjektivním predikátem — fold nespekuluje derivací."""
    from jellyai.graph.hygiene import fold_participles
    g = FactGraph()
    g.add_fact(make_fact("ukřižovaný", [Participant("obj", "Ježíš", "person"),
                                        Participant("loc", "Golgota", "geo")]))
    folded = fold_participles(g)
    assert folded == 0
    assert any(f.predicate == "ukřižovaný" for f in g.facts.values())


def test_fold_slouci_vahy_shodnych_faktu():
    """Fold na už existující shodný fakt = agregace vah a zdrojů."""
    from jellyai.graph.hygiene import fold_participles
    g = FactGraph()
    g.add_fact(make_fact("vydaný", [Participant("obj", "kniha", "concept"),
                                    Participant("num", "1920", "number")]))
    g.add_fact(make_fact("vydat", [Participant("obj", "kniha", "concept"),
                                   Participant("num", "1920", "number")]))
    g.add_fact(make_fact("vydat", [Participant("subj", "Čapek", "person"),
                                   Participant("obj", "drama", "concept")]))
    fold_participles(g)
    merged = next(f for f in g.facts.values()
                  if ("num", "1920") in {(p.role, p.node)
                                         for p in f.participants})
    assert merged.predicate == "vydat" and merged.weight == 2


def _form_annotations(tokens_by_sentence):
    """Anotace z [(form, upos), …] vět — hlasování povrchových tvarů."""
    return {("doc", i): {"sentences": [[{"form": f, "lemma": f.lower(),
                                         "upos": u, "head": 0,
                                         "deprel": "dep"}
                                        for f, u in sent]]}
            for i, sent in enumerate(tokens_by_sentence)}


def test_scrub_clause_objects_drops_glued_clause():
    """T10 (dávka D zbytek c): uzel „vydal František Borový Praha" je
    slepená klauzule, ne entita — slovo „vydal" korpus převážně značí
    VERB. Fakt bez protistrany padá s ním."""
    from jellyai.graph.hygiene import form_upos_votes, scrub_clause_objects
    g = FactGraph()
    g.add_fact(make_fact("vydat", [
        Participant("subj", "František Borový", "person"),
        Participant("obj", "vydal František Borový Praha", "person")]))
    votes = form_upos_votes(_form_annotations(
        [[("vydal", "VERB")]] * 3 + [[("František", "PROPN")] * 3]))
    dropped_p, dropped_f = scrub_clause_objects(g, votes)
    assert dropped_p == 1 and dropped_f == 1


def test_scrub_clause_objects_keeps_l_shaped_names():
    """Jména tvaru l-příčestí (Karel, Pavel) mají PROPN hlasy tvarů —
    guard je NEsmí zahodit; hlasuje korpus, ne lokální tvar."""
    from jellyai.graph.hygiene import form_upos_votes, scrub_clause_objects
    g = FactGraph()
    g.add_fact(make_fact("napsat", [
        Participant("subj", "Karel Čapek", "person"),
        Participant("obj", "Bílá nemoc", "dílo")]))
    votes = form_upos_votes(_form_annotations(
        [[("Karel", "PROPN"), ("Čapek", "PROPN")]] * 3))
    dropped_p, dropped_f = scrub_clause_objects(g, votes)
    assert dropped_p == 0 and dropped_f == 0
    assert any(p.node == "Karel Čapek" for f in g.facts.values()
               for p in f.participants)


def test_scrub_clause_objects_nesaha_na_vyrok_a_cas():
    """Kanál obsahu řeči (typ výrok) a časové uzly („19. června 1841")
    jsou záměrné víceslovné uzly — guard je míjí."""
    from jellyai.graph.hygiene import form_upos_votes, scrub_clause_objects
    g = FactGraph()
    g.add_fact(make_fact("říci", [
        Participant("subj", "Mojžíš", "person"),
        Participant("obj", "budu s tebou", "výrok")]))
    g.add_fact(make_fact("pokřtít", [
        Participant("obj", "dcera", "concept"),
        Participant("time", "19. června 1841", "time")]))
    votes = form_upos_votes(_form_annotations([[("budu", "VERB")]] * 3))
    dropped_p, dropped_f = scrub_clause_objects(g, votes)
    assert dropped_p == 0 and dropped_f == 0


def test_scrub_clause_objects_vyrok_na_pohybovem_slovese():
    """T10 příklad: přijít(obj=„odešel opět na horu zcela sám", výrok) —
    pohybové sloveso (movement_predicates) obsah řeči nenese, výrok na
    něm je slepená klauzule. Na řečovém slovese výrok zůstává."""
    from jellyai.graph.hygiene import form_upos_votes, scrub_clause_objects
    g = FactGraph()
    g.add_fact(make_fact("přijít", [
        Participant("subj", "Ježíš", "person"),
        Participant("obj", "odešel opět na horu zcela sám", "výrok")]))
    g.add_fact(make_fact("říci", [
        Participant("subj", "Mojžíš", "person"),
        Participant("obj", "budu s tebou", "výrok")]))
    dropped_p, dropped_f = scrub_clause_objects(
        g, form_upos_votes(_form_annotations([])))
    assert dropped_p == 1 and dropped_f == 1
    assert any(p.node == "budu s tebou" for f in g.facts.values()
               for p in f.participants)
