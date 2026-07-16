"""Reifikace vztahů — relační podstatné jméno + genitivní osoba → osoba–vztah–osoba."""

from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import extract_facts


def _ann(sent, entities):
    return {"entities": entities, "sentences": [sent]}


def test_relational_noun_with_genitive_person():
    """„Josef byl bratr Karla Čapka" → bratr(Josef, Karel Čapek), ne jen být→bratr."""
    sent = [
        {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 6, "end": 9},
        {"form": "bratr", "lemma": "bratr", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 10, "end": 15},
        {"form": "Karla", "lemma": "Karel", "upos": "PROPN", "head": 3,
         "deprel": "nmod", "start": 16, "end": 21},
        {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 4,
         "deprel": "flat", "start": 22, "end": 27},
    ]
    entities = [{"text": "Josef", "type": "P", "start": 0, "end": 5},
                {"text": "Karel Čapek", "type": "P", "start": 16, "end": 27}]
    facts = extract_facts(_ann(sent, entities))
    rel = next(f for f in facts if f.predicate == "bratr")
    g = FactGraph()
    for f in facts:
        g.add_fact(f)
    # z Karla se dá dojít na bratra Josefa (obj-role dotaz „Kdo byl bratr X?")
    br = g.facts_of("Karel Čapek", role="obj", predicate="bratr")
    assert br and "Josef" in g.participants(br[0], "subj")
    assert not any(p.role == "pred" and p.node == "bratr" for p in rel.participants)


def test_any_copular_noun_with_genitive_person_reifies():
    """„R.U.R. je (vědeckofantastické) drama Karla Čapka" → drama(R.U.R., Karel
    Čapek) univerzálně (vztah = struktura, ne slovník); k tomu autorství
    napsat(Karel Čapek, R.U.R.) (work_nouns z jazykových dat) a identita
    být(R.U.R., drama) pro „Co je R.U.R.?"."""
    sent = [
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 4,
         "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 4,
         "deprel": "cop", "start": 7, "end": 9},
        {"form": "vědeckofantastické", "lemma": "vědeckofantastický", "upos": "ADJ",
         "head": 4, "deprel": "amod", "start": 10, "end": 28},
        {"form": "drama", "lemma": "drama", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 29, "end": 34},
        {"form": "Karla", "lemma": "Karel", "upos": "PROPN", "head": 4,
         "deprel": "nmod", "start": 35, "end": 40},
        {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 5,
         "deprel": "flat", "start": 41, "end": 46},
    ]
    entities = [{"text": "R.U.R.", "type": "P", "start": 0, "end": 6},
                {"text": "Karla Čapka", "type": "P", "start": 35, "end": 46}]
    facts = extract_facts(_ann(sent, entities))
    assert {"drama", "napsat", "být"} <= {f.predicate for f in facts}
    napsat = next(f for f in facts if f.predicate == "napsat")
    roles = [(p.role, p.node) for p in napsat.participants]
    assert ("subj", "Karla Čapka") in roles and ("obj", "R.U.R.") in roles


def test_adjective_root_copula_with_nominal_predicate_reifies():
    """„Válka s Mloky je satirický sci-fi román Karla Čapka" — parser dává kořen
    na adjektivum a jmenný přísudek („román" s genitivem) věší jako druhý nsubj
    za sponu; kompenzace ho najde → román(válka, Karel Čapek) + napsat(Karel
    Čapek, válka) + být(válka, román)."""
    sent = [
        {"form": "Válka", "lemma": "válka", "upos": "NOUN", "head": 5,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "s", "lemma": "s", "upos": "ADP", "head": 3,
         "deprel": "case", "start": 6, "end": 7},
        {"form": "Mloky", "lemma": "mlok", "upos": "NOUN", "head": 1,
         "deprel": "nmod", "start": 8, "end": 13},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 5,
         "deprel": "cop", "start": 14, "end": 16},
        {"form": "satirický", "lemma": "satirický", "upos": "ADJ", "head": 0,
         "deprel": "root", "start": 17, "end": 26},
        {"form": "román", "lemma": "román", "upos": "NOUN", "head": 5,
         "deprel": "nsubj", "start": 27, "end": 32},
        {"form": "Karla", "lemma": "Karel", "upos": "PROPN", "head": 6,
         "deprel": "nmod", "start": 33, "end": 38},
        {"form": "Čapka", "lemma": "Čapek", "upos": "PROPN", "head": 7,
         "deprel": "flat", "start": 39, "end": 44},
    ]
    entities = [{"text": "Karla Čapka", "type": "P", "start": 33, "end": 44}]
    facts = extract_facts(_ann(sent, entities))
    assert {"román", "napsat"} <= {f.predicate for f in facts}
    napsat = next(f for f in facts if f.predicate == "napsat")
    roles = [(p.role, p.node) for p in napsat.participants]
    assert ("subj", "Karla Čapka") in roles and ("obj", "válka") in roles
    byt = next(f for f in facts if f.predicate == "být")
    assert any(p.role == "pred" and p.node == "román" for p in byt.participants)


def test_prepositional_person_phrase_is_not_authorship():
    """„X je drama o Karlu Čapkovi" — osoba s předložkou (aboutness) NENÍ genitivní
    vztah → žádné drama(X, Karel) ani napsat(Karel, X)."""
    sent = [
        {"form": "Hra", "lemma": "hra", "upos": "NOUN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 3},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 4, "end": 6},
        {"form": "drama", "lemma": "drama", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 7, "end": 12},
        {"form": "o", "lemma": "o", "upos": "ADP", "head": 5,
         "deprel": "case", "start": 13, "end": 14},
        {"form": "Karlu", "lemma": "Karel", "upos": "PROPN", "head": 3,
         "deprel": "nmod", "start": 15, "end": 20},
        {"form": "Čapkovi", "lemma": "Čapek", "upos": "PROPN", "head": 5,
         "deprel": "flat", "start": 21, "end": 28},
    ]
    entities = [{"text": "Karlu Čapkovi", "type": "P", "start": 15, "end": 28}]
    facts = extract_facts(_ann(sent, entities))
    assert not any(f.predicate in ("drama", "napsat") for f in facts)


def test_copular_noun_with_nonperson_genitive_stays_identity():
    """„Praha je město kontrastů" — genitiv není osoba → žádná relace, jen identita."""
    sent = [
        {"form": "Praha", "lemma": "Praha", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 6, "end": 8},
        {"form": "město", "lemma": "město", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 9, "end": 14},
        {"form": "kontrastů", "lemma": "kontrast", "upos": "NOUN", "head": 3,
         "deprel": "nmod", "start": 15, "end": 24},
    ]
    entities = [{"text": "Praha", "type": "G", "start": 0, "end": 5}]
    facts = extract_facts(_ann(sent, entities))
    assert not any(f.predicate == "město" for f in facts)
    byt = next(f for f in facts if f.predicate == "být")
    assert any(p.role == "pred" and p.node == "město" for p in byt.participants)


def test_overt_pronoun_subject_blocks_prodrop():
    """„Je to lepra." má overtní zájmenný podmět („to") — to NENÍ pro-drop
    elize; dosazení nejteplejší osoby by vyrobilo šum být(Karel, lepra),
    který po sloučení uzlů přebíjí skutečnou identitu."""
    sent = [
        {"form": "Je", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 0, "end": 2},
        {"form": "to", "lemma": "ten", "upos": "DET", "head": 3,
         "deprel": "nsubj", "start": 3, "end": 5},
        {"form": "lepra", "lemma": "lepra", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 6, "end": 11},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    assert not facts


def test_true_prodrop_copula_still_inherits_subject():
    """„Byl spisovatel." bez podmětu = skutečný pro-drop → osoba se dosadí."""
    sent = [
        {"form": "Byl", "lemma": "být", "upos": "AUX", "head": 2,
         "deprel": "cop", "start": 0, "end": 3},
        {"form": "spisovatel", "lemma": "spisovatel", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 4, "end": 14},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    byt = next(f for f in facts if f.predicate == "být")
    assert any(p.role == "subj" and p.node == "Karel Čapek"
               for p in byt.participants)


def test_nonrelational_copula_stays_identity():
    """Nevztahové sponové podstatné jméno („spisovatel") zůstane identitou (pred)."""
    sent = [
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 6, "end": 9},
        {"form": "spisovatel", "lemma": "spisovatel", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 10, "end": 20},
    ]
    entities = [{"text": "Karel", "type": "P", "start": 0, "end": 5}]
    facts = extract_facts(_ann(sent, entities))
    byt = next(f for f in facts if f.predicate == "být")
    assert any(p.role == "pred" and p.node == "spisovatel" for p in byt.participants)


def test_existential_copula_blocked_by_gender_disagreement():
    """„Byla válka." (Fem) při nejteplejším Čapkovi (Masc) → existenciál,
    ŽÁDNÉ být(Čapek, válka). Rod slovesného tvaru rozhoduje."""
    sent = [
        {"form": "Byla", "lemma": "být", "upos": "AUX", "head": 2,
         "deprel": "cop", "start": 0, "end": 4, "feats": {"Gender": "Fem"}},
        {"form": "válka", "lemma": "válka", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 5, "end": 10},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    assert not facts


def test_prodrop_with_gender_agreement_survives():
    """„Byl spisovatelem." (Masc) při Čapkovi (Masc) → pro-drop platí."""
    sent = [
        {"form": "Byl", "lemma": "být", "upos": "AUX", "head": 2,
         "deprel": "cop", "start": 0, "end": 3, "feats": {"Gender": "Masc"}},
        {"form": "spisovatelem", "lemma": "spisovatel", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 4, "end": 16},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    byt = next(f for f in facts if f.predicate == "být")
    assert any(p.role == "subj" and p.node == "Karel Čapek"
               for p in byt.participants)


def test_verb_prodrop_gender_mismatch_blocked():
    """„Narodila se roku 1820." (Fem) při Čapkovi (Masc) → fakt nevznikne."""
    sent = [
        {"form": "Narodila", "lemma": "narodit", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 0, "end": 8, "feats": {"Gender": "Fem"}},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 1,
         "deprel": "expl", "start": 9, "end": 11},
        {"form": "1820", "lemma": "1820", "upos": "NUM", "head": 1,
         "deprel": "obl", "start": 17, "end": 21},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    assert not any(p.node == "Karel Čapek" for f in facts for p in f.participants)


def test_interrogative_pred_makes_no_identity():
    """Sponová věta s tázacím kořenem („…jaký je.") nezakládá být(osoba, jaký)."""
    sent = [
        {"form": "jaký", "lemma": "jaký", "upos": "ADJ", "head": 0,
         "deprel": "root", "start": 0, "end": 4,
         "feats": {"PronType": "Int"}},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 1,
         "deprel": "cop", "start": 5, "end": 7},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    assert not facts


def test_demonstrative_determiner_blocks_identity():
    """„…byl TOUTO válkou (ovlivněn)" — demonstrativum u přísudkového jména
    značí adjunkt (parser-quirk), ne identitu → žádné být(osoba, válka)."""
    sent = [
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3,
         "deprel": "cop", "start": 0, "end": 3, "feats": {"Gender": "Masc"}},
        {"form": "touto", "lemma": "tento", "upos": "DET", "head": 3,
         "deprel": "det", "start": 4, "end": 9, "feats": {"PronType": "Dem"}},
        {"form": "válkou", "lemma": "válka", "upos": "NOUN", "head": 0,
         "deprel": "root", "start": 10, "end": 16},
    ]
    facts = extract_facts(_ann(sent, []),
                          default_subject=("Karel Čapek", "person"))
    assert not any(f.predicate == "být" for f in facts)
