from jellyai.graph.extract import extract_facts, make_fact, Fact, Participant


def _ann(sent, entities=None):
    return {"entities": entities or [], "sentences": [sent]}


def test_svo_fact():
    sent = [
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]
    ents = [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}]
    facts = extract_facts(_ann(sent, ents))
    expected = make_fact("napsat", [Participant("subj", "Božena Němcová", "person"),
                                    Participant("obj", "Babička", "concept")])
    assert expected in facts


def test_copula_fact():
    sent = [
        {"form": "Rossum", "lemma": "Rossum", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 7, "end": 9},
        {"form": "vynálezce", "lemma": "vynálezce", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 19},
    ]
    facts = extract_facts(_ann(sent))
    expected = make_fact("být", [Participant("subj", "Rossum", "concept"),
                                 Participant("pred", "vynálezce", "concept")])
    assert expected in facts


def test_copula_adjective_goes_to_attr_not_identity():
    """„Božena je nemocná" → role „attr" (vlastnost), ne „pred" (identita)."""
    sent = [
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 7, "end": 9},
        {"form": "nemocná", "lemma": "nemocný", "upos": "ADJ", "head": 0, "deprel": "root", "start": 10, "end": 17},
    ]
    facts = extract_facts(_ann(sent))
    fact = next(f for f in facts if f.predicate == "být")
    roles = {p.role for p in fact.participants}
    assert "attr" in roles and "pred" not in roles   # vlastnost, ne identita


def test_nary_fact_place_and_time():
    sent = [
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 6, "end": 13},
        {"form": "Praze", "lemma": "Praha", "upos": "PROPN", "head": 2, "deprel": "obl", "start": 17, "end": 22},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 2, "deprel": "obl", "start": 23, "end": 27},
    ]
    ents = [{"text": "Čapek", "type": "P", "start": 0, "end": 5},
            {"text": "Praha", "type": "G", "start": 17, "end": 22}]
    facts = extract_facts(_ann(sent, ents))
    expected = make_fact("narodit", [Participant("subj", "Čapek", "person"),
                                     Participant("loc", "Praha", "geo"),
                                     Participant("num", "1890", "number")])
    assert expected in facts


def test_skips_pronoun_object():
    # objekt-zájmeno „který" se nezahrne → z (subj) samotného nevznikne fakt
    sent = [
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 2, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 6, "end": 12},
        {"form": "který", "lemma": "který", "upos": "PRON", "head": 2, "deprel": "obj", "start": 13, "end": 18},
    ]
    ents = [{"text": "Čapek", "type": "P", "start": 0, "end": 5}]
    assert extract_facts(_ann(sent, ents)) == []


def test_prodrop_attributes_default_subject():
    # „Narodil se roku 1890." bez explicitního podmětu → doplní default subjekt
    sent = [
        {"form": "Narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 7},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 1, "deprel": "expl", "start": 8, "end": 10},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 1, "deprel": "obl", "start": 15, "end": 19},
    ]
    facts = extract_facts(_ann(sent), default_subject=("Karel Čapek", "person"))
    expected = make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                                     Participant("num", "1890", "number")])
    assert expected in facts


def test_no_default_no_subject_no_fact():
    # bez explicitního podmětu i bez defaultu → žádný fakt (nespekuluje)
    sent = [
        {"form": "Narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 7},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 1, "deprel": "obl", "start": 8, "end": 12},
    ]
    assert extract_facts(_ann(sent)) == []


def test_nested_date_uses_full_time_entity():
    # "Narodil se 13. ledna 1890." — datum je zanořené (nummod pod měsícem),
    # ale zachytí se celá časová entita a naveze na sloveso
    sent = [
        {"form": "Narodil", "lemma": "narodit", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 7},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 1, "deprel": "expl", "start": 8, "end": 10},
        {"form": "13", "lemma": "13", "upos": "NUM", "head": 4, "deprel": "nummod", "start": 11, "end": 13},
        {"form": "ledna", "lemma": "leden", "upos": "NOUN", "head": 1, "deprel": "obl", "start": 14, "end": 19},
        {"form": "1890", "lemma": "1890", "upos": "NUM", "head": 4, "deprel": "nummod", "start": 20, "end": 24},
    ]
    ents = [{"text": "13. ledna 1890", "type": "T", "start": 11, "end": 24},
            {"text": "ledna", "type": "tm", "start": 14, "end": 19}]
    facts = extract_facts(_ann(sent, ents), default_subject=("Karel Čapek", "person"))
    expected = make_fact("narodit", [Participant("subj", "Karel Čapek", "person"),
                                     Participant("time", "13. ledna 1890", "time")])
    assert expected in facts


def test_nested_number_attached_to_governing_verb():
    # "Napsal knihu roku 1920." — 1920 je nummod pod 'roku' (obl), naveze na 'napsat'
    sent = [
        {"form": "Napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 6},
        {"form": "knihu", "lemma": "kniha", "upos": "NOUN", "head": 1, "deprel": "obj", "start": 7, "end": 12},
        {"form": "roku", "lemma": "rok", "upos": "NOUN", "head": 1, "deprel": "obl", "start": 13, "end": 17},
        {"form": "1920", "lemma": "1920", "upos": "NUM", "head": 3, "deprel": "nummod", "start": 18, "end": 22},
    ]
    facts = extract_facts(_ann(sent), default_subject=("Autor", "person"))
    assert any(p.role == "num" and p.node == "1920" for f in facts for p in f.participants)


def test_attribute_goes_to_its_own_clause_verb():
    # dvě slovesa, dvě čísla — každé číslo ke svému slovesu (nejbližší předek)
    sent = [
        {"form": "Debutoval", "lemma": "debutovat", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 9},
        {"form": "1918", "lemma": "1918", "upos": "NUM", "head": 1, "deprel": "obl", "start": 10, "end": 14},
        {"form": "a", "lemma": "a", "upos": "CCONJ", "head": 4, "deprel": "cc", "start": 15, "end": 16},
        {"form": "zemřel", "lemma": "zemřít", "upos": "VERB", "head": 1, "deprel": "conj", "start": 17, "end": 23},
        {"form": "1938", "lemma": "1938", "upos": "NUM", "head": 4, "deprel": "obl", "start": 24, "end": 28},
    ]
    facts = extract_facts(_ann(sent), default_subject=("X", "person"))
    debut = next(f for f in facts if f.predicate == "debutovat")
    zemr = next(f for f in facts if f.predicate == "zemřít")
    assert any(p.node == "1918" for p in debut.participants)
    assert any(p.node == "1938" for p in zemr.participants)
    assert not any(p.node == "1938" for p in debut.participants)
