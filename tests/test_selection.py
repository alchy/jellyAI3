from jellyai.answerer.selection import select_answer


def _annotation_nemcova():
    # „Božena Němcová napsala Babičku." (head = 1-based id v UD)
    sent = [
        {"form": "Božena", "lemma": "Božena", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 6},
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 7, "end": 14},
        {"form": "napsala", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 15, "end": 22},
        {"form": "Babičku", "lemma": "Babička", "upos": "PROPN", "head": 3, "deprel": "obj", "start": 23, "end": 30},
    ]
    return {"entities": [{"text": "Božena Němcová", "type": "P", "start": 0, "end": 14}],
            "sentences": [sent]}


def test_kdo_selects_subject_in_nominative():
    c = select_answer("Kdo", "napsat", _annotation_nemcova())
    assert c.form == "Božena Němcová"
    assert c.lemma == "Božena Němcová"   # nominativ z lemmat


def test_co_selects_object_normalized():
    c = select_answer("Co", "napsat", _annotation_nemcova())
    assert c.form == "Babičku"
    assert c.lemma == "Babička"          # 4. pád → 1. pád přes lemma


def test_disambiguates_two_persons():
    # „Karel Čapek a Josef napsali …" — podmětem je Karel Čapek, ne Josef.
    sent = [
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 5, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 1, "deprel": "flat", "start": 6, "end": 11},
        {"form": "a", "lemma": "a", "upos": "CCONJ", "head": 4, "deprel": "cc", "start": 12, "end": 13},
        {"form": "Josef", "lemma": "Josef", "upos": "PROPN", "head": 1, "deprel": "conj", "start": 14, "end": 19},
        {"form": "napsali", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 20, "end": 27},
    ]
    ann = {"entities": [
        {"text": "Karel Čapek", "type": "P", "start": 0, "end": 11},
        {"text": "Josef", "type": "P", "start": 14, "end": 19},
    ], "sentences": [sent]}
    c = select_answer("Kdo", "napsat", ann)
    assert c.form == "Karel Čapek"       # podmět slovesa, ne druhá osoba


def test_select_predicate_copula():
    from jellyai.answerer.selection import select_predicate
    # „Němcová byla významná spisovatelka." (přísudek = kořen se sponou)
    sent = [
        {"form": "Němcová", "lemma": "Němcová", "upos": "PROPN", "head": 4, "deprel": "nsubj", "start": 0, "end": 7},
        {"form": "byla", "lemma": "být", "upos": "AUX", "head": 4, "deprel": "cop", "start": 8, "end": 12},
        {"form": "významná", "lemma": "významný", "upos": "ADJ", "head": 4, "deprel": "amod", "start": 13, "end": 21},
        {"form": "spisovatelka", "lemma": "spisovatelka", "upos": "NOUN", "head": 0, "deprel": "root", "start": 22, "end": 34},
    ]
    c = select_predicate({"entities": [], "sentences": [sent]}, "Jaký")
    assert c.form == "významná spisovatelka"


def test_select_predicate_requires_subject_match():
    from jellyai.answerer.selection import select_predicate
    # spona je o „Čapkovi", ne o tématu otázky → nesmí se vzít (jinak „sporné")
    sent = [
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "nsubj", "start": 0, "end": 5},
        {"form": "byl", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop", "start": 6, "end": 9},
        {"form": "spisovatel", "lemma": "spisovatel", "upos": "NOUN", "head": 0, "deprel": "root", "start": 10, "end": 20},
    ]
    ann = {"entities": [], "sentences": [sent]}
    assert select_predicate(ann, "Jaký", topic_terms=["Němcová"]) is None
    assert select_predicate(ann, "Jaký", topic_terms=["Čapek"]).form == "spisovatel"


def test_co_skips_pronoun_object():
    # předmět-zájmeno „který" se přeskočí, vybere se podstatné jméno „knihu"
    sent = [
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 6},
        {"form": "který", "lemma": "který", "upos": "PRON", "head": 1, "deprel": "obj", "start": 7, "end": 12},
        {"form": "knihu", "lemma": "kniha", "upos": "NOUN", "head": 1, "deprel": "obj", "start": 13, "end": 18},
    ]
    c = select_answer("Co", "napsat", {"entities": [], "sentences": [sent]})
    assert c.form == "knihu"


def test_kolik_selects_number():
    sent = [
        {"form": "Měl", "lemma": "mít", "upos": "VERB", "head": 0, "deprel": "root", "start": 0, "end": 3},
        {"form": "tři", "lemma": "tři", "upos": "NUM", "head": 1, "deprel": "obj", "start": 4, "end": 7},
        {"form": "děti", "lemma": "dítě", "upos": "NOUN", "head": 1, "deprel": "obj", "start": 8, "end": 12},
    ]
    c = select_answer("Kolik", None, {"entities": [], "sentences": [sent]})
    assert c.form == "tři"
