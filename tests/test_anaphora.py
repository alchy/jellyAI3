"""Zájmenná anafora — zobecněný pro-drop přes aktivační pole + rodovou shodu.

Osobní zájmeno 3. osoby (on/ona/ji/mu) se rozváže na nejteplejší rodově
shodnou osobu kontextu (rod zájmena z feats, rod jména z tvaru příjmení —
jazyková data). Demonstrativum („to") ani reflexivum („se") osobu neváže.
"""

from jellyai.graph.extract import extract_facts
from jellyai.graph.canon import name_gender


def _ann(sent, entities=()):
    return {"entities": list(entities), "sentences": [sent]}


def test_name_gender_from_language_data():
    assert name_gender("Božena Němcová") == "Fem"
    assert name_gender("Karel Čapek") == "Masc"
    assert name_gender("Jaroslav Seifert") == "Masc"


def test_object_pronoun_binds_gender_matching_person():
    """„Fučík ji vnímal." → vnímat(Fučík, Božena Němcová) — „ji" je Fem,
    první rodově shodná osoba kontextu vyhrává (Josef Němec je teplejší,
    ale Masc)."""
    sent = [
        {"form": "Fučík", "lemma": "Fučík", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5, "feats": {"Gender": "Masc"}},
        {"form": "ji", "lemma": "on", "upos": "PRON", "head": 3, "deprel": "obj",
         "start": 6, "end": 8,
         "feats": {"Case": "Acc", "Gender": "Fem", "PronType": "Prs", "Person": "3"}},
        {"form": "vnímal", "lemma": "vnímat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 9, "end": 15},
    ]
    entities = [{"text": "Fučík", "type": "P", "start": 0, "end": 5}]
    context = [("Josef Němec", "person"), ("Božena Němcová", "person")]
    facts = extract_facts(_ann(sent, entities), context=context)
    fact = next(f for f in facts if f.predicate == "vnímat")
    roles = [(p.role, p.node) for p in fact.participants]
    assert ("obj", "Božena Němcová") in roles and ("subj", "Fučík") in roles


def test_subject_pronoun_binds_person():
    """„On napsal drama." → napsat(Karel Čapek, drama)."""
    sent = [
        {"form": "On", "lemma": "on", "upos": "PRON", "head": 2, "deprel": "nsubj",
         "start": 0, "end": 2,
         "feats": {"Gender": "Masc", "PronType": "Prs", "Person": "3"}},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 3, "end": 9},
        {"form": "drama", "lemma": "drama", "upos": "NOUN", "head": 2,
         "deprel": "obj", "start": 10, "end": 15},
    ]
    context = [("Božena Němcová", "person"), ("Karel Čapek", "person")]
    facts = extract_facts(_ann(sent), context=context)
    fact = next(f for f in facts if f.predicate == "napsat")
    assert ("subj", "Karel Čapek") in [(p.role, p.node) for p in fact.participants]


def test_demonstrative_subject_never_binds_person():
    """„To vedlo k žalobě." — demonstrativum nesmí zdědit osobu (ani pro-drop)."""
    sent = [
        {"form": "To", "lemma": "ten", "upos": "DET", "head": 2, "deprel": "nsubj",
         "start": 0, "end": 2, "feats": {"Gender": "Neut", "PronType": "Dem"}},
        {"form": "vedlo", "lemma": "vést", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 3, "end": 8},
        {"form": "žalobě", "lemma": "žaloba", "upos": "NOUN", "head": 2,
         "deprel": "obj", "start": 11, "end": 17},
    ]
    facts = extract_facts(_ann(sent), default_subject=("Karel Čapek", "person"),
                          context=[("Karel Čapek", "person")])
    assert not any(p.node == "Karel Čapek" for f in facts for p in f.participants)


def test_gender_mismatch_does_not_bind():
    """„ji" bez ženské osoby v kontextu se neváže (žádný falešný účastník)."""
    sent = [
        {"form": "Fučík", "lemma": "Fučík", "upos": "PROPN", "head": 3,
         "deprel": "nsubj", "start": 0, "end": 5, "feats": {"Gender": "Masc"}},
        {"form": "ji", "lemma": "on", "upos": "PRON", "head": 3, "deprel": "obj",
         "start": 6, "end": 8,
         "feats": {"Case": "Acc", "Gender": "Fem", "PronType": "Prs", "Person": "3"}},
        {"form": "vnímal", "lemma": "vnímat", "upos": "VERB", "head": 0,
         "deprel": "root", "start": 9, "end": 15},
    ]
    entities = [{"text": "Fučík", "type": "P", "start": 0, "end": 5}]
    facts = extract_facts(_ann(sent, entities),
                          context=[("Karel Čapek", "person")])
    assert not any(p.node == "Karel Čapek" for f in facts for p in f.participants)
