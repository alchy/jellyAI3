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


def test_concept_oblique_becomes_theme_participant():
    """„Uvažoval o literatuře." → uvažovat(osoba, theme=literatura) —
    konceptové obl se už nezahazuje (největší kbelík coverage auditu)."""
    sent = [
        {"form": "Uvažoval", "lemma": "uvažovat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 8},
        {"form": "o", "lemma": "o", "upos": "ADP", "head": 3, "deprel": "case",
         "start": 9, "end": 10},
        {"form": "literatuře", "lemma": "literatura", "upos": "NOUN", "head": 1,
         "deprel": "obl", "start": 11, "end": 21},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("Karel Čapek", "person"))
    fact = next(f for f in facts if f.predicate == "uvažovat")
    assert ("theme", "literatura") in [(p.role, p.node) for p in fact.participants]


def test_function_noun_oblique_is_not_theme():
    """Funkční substantivum („v souvislosti s…") theme účastníka NEvytvoří —
    je to spojovací vata, ne obsah; do grafu by nesla šumové uzly."""
    sent = [
        {"form": "Uvažoval", "lemma": "uvažovat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 8},
        {"form": "o", "lemma": "o", "upos": "ADP", "head": 3, "deprel": "case",
         "start": 9, "end": 10},
        {"form": "souvislosti", "lemma": "souvislost", "upos": "NOUN", "head": 1,
         "deprel": "obl", "start": 11, "end": 22},
        {"form": "díla", "lemma": "dílo", "upos": "NOUN", "head": 1,
         "deprel": "obj", "start": 23, "end": 27},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("Karel Čapek", "person"))
    fact = next(f for f in facts if f.predicate == "uvažovat")
    assert not any(p.node == "souvislost" for p in fact.participants)


def test_adverb_is_not_theme():
    """Příslovce ani přídavné jméno roli theme nedostane (jen obl substantiva)."""
    sent = [
        {"form": "Psal", "lemma": "psát", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 4},
        {"form": "rychle", "lemma": "rychle", "upos": "ADV", "head": 1,
         "deprel": "advmod", "start": 5, "end": 11},
        {"form": "romány", "lemma": "román", "upos": "NOUN", "head": 1,
         "deprel": "obj", "start": 12, "end": 18},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("Karel Čapek", "person"))
    fact = next(f for f in facts if f.predicate == "psát")
    assert not any(p.role == "theme" for p in fact.participants)


def test_apposition_across_comma_stays_out_of_object_group():
    """„Dvě budou mlít obilí, jedna přijata, druhá zanechána" — mis-tagnutá
    „apozice" z vedlejší klauze (přes čárku) do předmětové skupiny NEpatří;
    přilehlá apozice („hru R.U.R.") zůstává."""
    sent = [
        {"form": "Dvě", "lemma": "dva", "upos": "NUM", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "budou", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "aux", "start": 4, "end": 9},
        {"form": "mlít", "lemma": "mlít", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 10, "end": 14},
        {"form": "obilí", "lemma": "obilí", "upos": "NOUN", "head": 3,
         "deprel": "obj", "start": 15, "end": 20},
        {"form": ",", "lemma": ",", "upos": "PUNCT", "head": 3,
         "deprel": "punct", "start": 20, "end": 21},
        {"form": "druhá", "lemma": "druhý", "upos": "ADJ", "head": 7,
         "deprel": "amod", "start": 22, "end": 27},
        {"form": "zanechána", "lemma": "zanechaný", "upos": "NOUN", "head": 4,
         "deprel": "appos", "start": 28, "end": 37},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("X", "person"))
    fact = next(f for f in facts if f.predicate == "mlít")
    nodes = [p.node for p in fact.participants]
    assert "obilí" in nodes and "zanechaný" not in nodes


def test_verbal_conj_member_stays_out_of_object_group():
    """„Vezmi svou hůl a hoď ji" — mis-tagované sloveso („hoď" jako NOUN
    s VerbForm) pověšené jako conj na předmět do faktu NEpatří; legitimní
    výčet substantiv („romány a dramata") zůstává."""
    sent = [
        {"form": "Vezmi", "lemma": "vzít", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 5},
        {"form": "hůl", "lemma": "hůl", "upos": "NOUN", "head": 1,
         "deprel": "obj", "start": 6, "end": 9},
        {"form": "a", "lemma": "a", "upos": "CCONJ", "head": 4,
         "deprel": "cc", "start": 10, "end": 11},
        {"form": "hoď", "lemma": "hodit", "upos": "NOUN", "head": 2,
         "deprel": "conj", "feats": {"VerbForm": "Fin"}, "start": 12, "end": 15},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("X", "person"))
    nodes = [p.node for f in facts if f.predicate == "vzít"
             for p in f.participants]
    assert "hůl" in nodes and "hodit" not in nodes and "hoď" not in nodes


def test_legit_apposition_across_single_comma_survives():
    """„Jidáše, syna Šimonova" — legitimní apozice s čárkou (bez slovesa
    mezi, pádová shoda) se do faktu VRACÍ; klauzový únik zůstává venku."""
    sent = [
        {"form": "Zradil", "lemma": "zradit", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 6},
        {"form": "Jidáše", "lemma": "Jidáš", "upos": "PROPN", "head": 1,
         "deprel": "obj", "feats": {"Case": "Acc"}, "start": 7, "end": 13},
        {"form": ",", "lemma": ",", "upos": "PUNCT", "head": 4,
         "deprel": "punct", "start": 13, "end": 14},
        {"form": "syna", "lemma": "syn", "upos": "NOUN", "head": 2,
         "deprel": "appos", "feats": {"Case": "Acc"}, "start": 15, "end": 19},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("X", "person"))
    fact = next(f for f in facts if f.predicate == "zradit")
    assert "syn" in [p.node for p in fact.participants]


def test_coordinated_subjects_distribute():
    """„Karel a Josef psali romány." → psát(Karel, …) i psát(Josef, …)."""
    sent = [
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 4,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "a", "lemma": "a", "upos": "CCONJ", "head": 3, "deprel": "cc",
         "start": 6, "end": 7},
        {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 1,
         "deprel": "conj", "start": 8, "end": 13},
        {"form": "psali", "lemma": "psát", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 14, "end": 19},
        {"form": "romány", "lemma": "román", "upos": "NOUN", "head": 4,
         "deprel": "obj", "start": 20, "end": 26},
    ]
    entities = [{"text": "Karel", "type": "P", "start": 0, "end": 5},
                {"text": "Josef", "type": "P", "start": 8, "end": 13}]
    facts = [f for f in extract_facts({"entities": entities, "sentences": [sent]})
             if f.predicate == "psát"]
    subjects = {p.node for f in facts for p in f.participants if p.role == "subj"}
    assert subjects == {"Karel", "Josef"}


def test_coordinated_objects_distribute():
    """„Psal romány a dramata." → fakt s obj=román i fakt s obj=drama."""
    sent = [
        {"form": "Psal", "lemma": "psát", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 4},
        {"form": "romány", "lemma": "román", "upos": "NOUN", "head": 1,
         "deprel": "obj", "start": 5, "end": 11},
        {"form": "a", "lemma": "a", "upos": "CCONJ", "head": 4, "deprel": "cc",
         "start": 12, "end": 13},
        {"form": "dramata", "lemma": "drama", "upos": "NOUN", "head": 2,
         "deprel": "conj", "start": 14, "end": 21},
    ]
    facts = [f for f in extract_facts({"entities": [], "sentences": [sent]},
                                      default_subject=("Karel Čapek", "person"))
             if f.predicate == "psát"]
    objs = {p.node for f in facts for p in f.participants if p.role == "obj"}
    assert objs == {"román", "drama"}


def test_apposition_joins_object_in_same_fact():
    """„Napsal hru R.U.R." → napsat(osoba, obj=hra, obj=R.U.R.) v JEDNOM faktu —
    „Jakou hru napsal X?" pak najde díru R.U.R. (známí: hra + osoba)."""
    sent = [
        {"form": "Napsal", "lemma": "napsat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 6},
        {"form": "hru", "lemma": "hra", "upos": "NOUN", "head": 1,
         "deprel": "obj", "start": 7, "end": 10},
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 2,
         "deprel": "appos", "start": 11, "end": 17},
    ]
    facts = [f for f in extract_facts({"entities": [], "sentences": [sent]},
                                      default_subject=("Karel Čapek", "person"))
             if f.predicate == "napsat"]
    assert len(facts) == 1
    objs = {p.node for p in facts[0].participants if p.role == "obj"}
    assert objs == {"hra", "R.U.R."}


def test_nominal_apposition_creates_identity_fact():
    """„…podle hry R.U.R. …" → být(R.U.R., pred=hra) — apozice je identita
    (instance ↔ druh), dostupná pro typové otázky „Jakou hru…?"."""
    sent = [
        {"form": "podle", "lemma": "podle", "upos": "ADP", "head": 2,
         "deprel": "case", "start": 0, "end": 5},
        {"form": "hry", "lemma": "hra", "upos": "NOUN", "head": 4,
         "deprel": "obl", "start": 6, "end": 9},
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 2,
         "deprel": "appos", "start": 10, "end": 16},
        {"form": "složil", "lemma": "složit", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 17, "end": 23},
        {"form": "operu", "lemma": "opera", "upos": "NOUN", "head": 4,
         "deprel": "obj", "start": 24, "end": 29},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("Zdeněk Blažek", "person"))
    byt = next(f for f in facts if f.predicate == "druh")
    roles = [(p.role, p.node) for p in byt.participants]
    assert ("subj", "R.U.R.") in roles and ("pred", "hra") in roles


def test_metalanguage_noun_is_not_identity_kind():
    """„rodným jménem Karel Antonín Čapek" — „jméno" je metajazyk (tabulka
    v lang), ne druh → žádné být(Karel Antonín Čapek, jméno)."""
    sent = [
        {"form": "rodným", "lemma": "rodný", "upos": "ADJ", "head": 2,
         "deprel": "amod", "start": 0, "end": 6},
        {"form": "jménem", "lemma": "jméno", "upos": "NOUN", "head": 4,
         "deprel": "obl", "start": 7, "end": 13},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2,
         "deprel": "nmod", "start": 14, "end": 19},
        {"form": "psal", "lemma": "psát", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 20, "end": 24},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]})
    assert not any(f.predicate == "být" for f in facts)


def test_common_noun_with_flat_propn_is_instance_identity():
    """„MATKA Božena Čapková sbírala…" — PROPN flat pod obecným substantivem
    je instance (titul+jméno): být(Božena Čapková, matka). Totéž „prezident
    Masaryk". Pádová shoda nutná (mis-tagged genitiv řeší pattern)."""
    sent = [
        {"form": "Matka", "lemma": "matka", "upos": "NOUN", "head": 4,
         "deprel": "nsubj", "start": 0, "end": 5, "feats": {"Case": "Nom"}},
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 1,
         "deprel": "flat", "start": 6, "end": 12, "feats": {"Case": "Nom"}},
        {"form": "Čapková", "lemma": "Čapková", "upos": "PROPN", "head": 2,
         "deprel": "flat", "start": 13, "end": 20, "feats": {"Case": "Nom"}},
        {"form": "sbírala", "lemma": "sbírat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 21, "end": 28},
        {"form": "folklor", "lemma": "folklor", "upos": "NOUN", "head": 4,
         "deprel": "obj", "start": 29, "end": 36},
    ]
    entities = [{"text": "Božena Čapková", "type": "P", "start": 6, "end": 20}]
    facts = extract_facts({"entities": entities, "sentences": [sent]})
    byt = next(f for f in facts if f.predicate == "druh")
    roles = [(p.role, p.node) for p in byt.participants]
    assert ("subj", "Božena Čapková") in roles and ("pred", "matka") in roles


def test_dash_definition_creates_kind_identity():
    """Bezslovesný řádek „1926 Adam stvořitel – Divadelní hra, …" →
    druh(Adam stvořitel, hra) — pomlčková definice encyklopedické položky."""
    sent = [
        {"form": "1926", "lemma": "1926", "upos": "NUM", "head": 2,
         "deprel": "nummod", "start": 0, "end": 4},
        {"form": "Adam", "lemma": "Adam", "upos": "PROPN", "head": 0,
         "deprel": "root", "start": 5, "end": 9},
        {"form": "stvořitel", "lemma": "stvořitel", "upos": "NOUN", "head": 2,
         "deprel": "flat", "start": 10, "end": 19},
        {"form": "–", "lemma": "–", "upos": "PUNCT", "head": 6,
         "deprel": "punct", "start": 20, "end": 21},
        {"form": "Divadelní", "lemma": "divadelní", "upos": "ADJ", "head": 6,
         "deprel": "amod", "start": 22, "end": 31},
        {"form": "hra", "lemma": "hra", "upos": "NOUN", "head": 2,
         "deprel": "appos", "start": 32, "end": 35},
    ]
    entities = [{"text": "Adam stvořitel", "type": "P", "start": 5, "end": 19}]
    facts = extract_facts({"entities": entities, "sentences": [sent]})
    druh = next(f for f in facts if f.predicate == "druh")
    roles = [(p.role, p.node) for p in druh.participants]
    assert ("subj", "Adam stvořitel") in roles and ("pred", "hra") in roles


def test_negated_verb_keeps_negation_in_predicate():
    """„Nemusel bojovat ve válce." — polarita patří do predikátu: nemuset,
    a xcomp dědí negaci (nebojovat) — žádný falešný fakt bojovat."""
    sent = [
        {"form": "Nemusel", "lemma": "muset", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 7,
         "feats": {"Polarity": "Neg", "Gender": "Masc"}},
        {"form": "bojovat", "lemma": "bojovat", "upos": "VERB", "head": 1,
         "deprel": "xcomp", "start": 8, "end": 15, "feats": {"Polarity": "Pos"}},
        {"form": "ve", "lemma": "v", "upos": "ADP", "head": 4,
         "deprel": "case", "start": 16, "end": 18},
        {"form": "válce", "lemma": "válka", "upos": "NOUN", "head": 2,
         "deprel": "obl", "start": 19, "end": 24},
    ]
    facts = extract_facts({"entities": [], "sentences": [sent]},
                          default_subject=("Karel Čapek", "person"))
    preds = {f.predicate for f in facts}
    assert "bojovat" not in preds and "muset" not in preds
    assert "nebojovat" in preds or "nemuset" in preds


def test_speech_content_clause_becomes_object():
    """„Ježíš řekl: Jdi za mnou." — obsah řeči (ccomp/parataxis) je účastník:
    „Co řekl Ježíš?" pak odpoví obsahem, ne adresátem."""
    sent = [
        {"form": "Ježíš", "lemma": "Ježíš", "upos": "PROPN", "head": 2,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "řekl", "lemma": "říci", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 6, "end": 10, "feats": {"Gender": "Masc"}},
        {"form": "Jdi", "lemma": "jít", "upos": "VERB", "head": 2,
         "deprel": "parataxis", "start": 12, "end": 15},
        {"form": "za", "lemma": "za", "upos": "ADP", "head": 5,
         "deprel": "case", "start": 16, "end": 18},
        {"form": "mnou", "lemma": "já", "upos": "PRON", "head": 3,
         "deprel": "obl", "start": 19, "end": 23},
    ]
    entities = [{"text": "Ježíš", "type": "P", "start": 0, "end": 5}]
    facts = extract_facts({"entities": entities, "sentences": [sent]})
    rekl = next(f for f in facts if f.predicate == "říci")
    objs = [p.node for p in rekl.participants if p.role == "obj"]
    assert objs and "Jdi za mnou" in objs[0]
