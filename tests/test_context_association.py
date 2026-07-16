"""Predikát jako preference — druhé patro matchování přes kontextovou asociaci.

Když přesný predikát otázky nemá v grafu fakt, odpoví asociační fakty
„kontext" (dokumentová blízkost entit — role ③ aktivačního pole). Porozumění
z grafu a vah, ne z ručně vytěžených řádků.
"""

from config import AnswererConfig
from jellyai.answerer.extractive import ExtractiveAnswerer
from jellyai.answerer.graph_answerer import GraphAnswerer
from jellyai.graph.graph import FactGraph
from jellyai.graph.extract import make_fact, Participant
from jellyai.ufal_client import FakeUfalClient


def _client(q):
    return FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 2, "deprel": "nsubj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "R.U.R.", "lemma": "R.U.R.", "upos": "PROPN", "head": 2, "deprel": "obj"},
    ]]})


def test_predicate_falls_back_to_context_association():
    """„Kdo napsal R.U.R.?" bez napsat-faktu → nejtěžší asociovaná osoba."""
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "R.U.R.", "person")]))
    g.add_fact(make_fact("kontext", [Participant("subj", "Harry Domin", "person"),
                                     Participant("obj", "R.U.R.", "person")]))
    q = "Kdo napsal R.U.R.?"
    a = GraphAnswerer(g, _client(q), ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Karel Čapek"     # váha 3 > 1


def test_identity_hole_never_guesses_from_context():
    """„Kdo je X?" (díra pred) kontextovou asociací NEhádá — bez být-faktu je
    poctivá odpověď „nenašel", ne nejtěžší soused (dialog: Ludvík Němec)."""
    g = FactGraph()
    for _ in range(3):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "Ludvík Němec", "person")]))
    q = "Kdo je Ludvík Němec?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kdo", "lemma": "kdo", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "je", "lemma": "být", "upos": "AUX", "head": 3, "deprel": "cop"},
        {"form": "Ludvík", "lemma": "Ludvík", "upos": "PROPN", "head": 0,
         "deprel": "root"},
        {"form": "Němec", "lemma": "Němec", "upos": "PROPN", "head": 3,
         "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert "Karel" not in a.answer(q, []).text


def test_exact_predicate_beats_association():
    """Existuje-li přesný fakt, asociace se nepoužije (predikát = preference)."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Josef Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    for _ in range(9):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "R.U.R.", "dílo")]))
    q = "Kdo napsal R.U.R.?"
    a = GraphAnswerer(g, _client(q), ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Josef Čapek"


def test_concept_known_acts_as_type_filter():
    """„Jakou hru napsal Karel Čapek?" — koncept „hra" nemá místo ve faktu →
    stane se typovým filtrem díry: napsat/kontext(KČ, ?) ∧ být(?, hra) → R.U.R."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "válka", "concept")]))
    for _ in range(2):
        g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                         Participant("obj", "R.U.R.", "person")]))
    g.add_fact(make_fact("kontext", [Participant("subj", "Karel Čapek", "person"),
                                     Participant("obj", "Josef Čapek", "person")]))
    g.add_fact(make_fact("být", [Participant("subj", "R.U.R.", "person"),
                                 Participant("pred", "hra", "concept")]))
    q = "Jakou hru napsal Karel Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Jakou", "lemma": "jaký", "upos": "DET", "head": 2, "deprel": "amod",
         "feats": {"PronType": "Int"}},
        {"form": "hru", "lemma": "hra", "upos": "NOUN", "head": 3, "deprel": "obj"},
        {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 3, "deprel": "nsubj"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 4, "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "R.U.R."


def test_followup_attr_uses_context_when_topic_is_hot():
    """„Jaká rodina?" jako NAVAZUJÍCÍ otázka (rodina svítí z minulé odpovědi)
    smí čerpat z kontextových vazeb; svěží identitní otázka dál nehádá."""
    g = FactGraph()
    g.add_fact(make_fact("kontext", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("obj", "rodina", "concept")]))
    q = "Jaká rodina?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Jaká", "lemma": "jaký", "upos": "DET", "head": 2, "deprel": "amod",
         "feats": {"PronType": "Int"}},
        {"form": "rodina", "lemma": "rodina", "upos": "NOUN", "head": 0,
         "deprel": "root"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    cold = a.answer(q, []).text
    assert "Božena" not in cold                  # svěží dotaz nehádá
    a.context.warm("rodina", 2.0)                # rodina byla minulou odpovědí
    hot = a.answer(q, []).text
    assert "Božena Němcová" in hot               # navazující čerpá z kontextu


def test_generic_event_question_returns_event_of_topic():
    """„Co se stalo s rodinou?" — „stát se" je lehké sloveso (jazyková data):
    odpověď = nejsilnější UDÁLOST tématu s účastníky, ne holé jméno."""
    g = FactGraph()
    g.add_fact(make_fact("ocitnout", [Participant("subj", "rodina", "concept"),
                                      Participant("loc", "Domažlicích", "geo")]))
    g.add_fact(make_fact("kontext", [Participant("subj", "Božena Němcová", "person"),
                                     Participant("obj", "rodina", "concept")]))
    q = "Co se stalo s rodinou?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Co", "lemma": "co", "upos": "PRON", "head": 3, "deprel": "nsubj"},
        {"form": "se", "lemma": "se", "upos": "PRON", "head": 3, "deprel": "expl"},
        {"form": "stalo", "lemma": "stát", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "s", "lemma": "s", "upos": "ADP", "head": 5, "deprel": "case"},
        {"form": "rodinou", "lemma": "rodina", "upos": "NOUN", "head": 3, "deprel": "obl"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    text = a.answer(q, []).text
    assert "ocitnout" in text and "Domažlicích" in text


def test_elided_question_subject_inherits_conversation_person():
    """„Co napsal KČ?" → …; „Jakou hru napsal?" — elidovaný podmět OTÁZKY
    se doplní rodově shodnou osobou z těžiště (query-side pro-drop; dialog
    z logu odpovídal osobou místo hry)."""
    g = FactGraph()
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "hra", "concept")]))
    g.add_fact(make_fact("napsat", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("obj", "R.U.R.", "dílo")]))
    g.add_fact(make_fact("být", [Participant("subj", "R.U.R.", "dílo"),
                                 Participant("pred", "hra", "concept")]))
    q1 = "Co napsal Karel Čapek?"
    q2 = "Jakou hru napsal?"
    client = FakeUfalClient(parse={
        q1: [[
            {"form": "Co", "lemma": "co", "upos": "PRON", "head": 2, "deprel": "obj"},
            {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root"},
            {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
            {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "flat"},
        ]],
        q2: [[
            {"form": "Jakou", "lemma": "jaký", "upos": "DET", "head": 2, "deprel": "amod",
             "feats": {"PronType": "Int"}},
            {"form": "hru", "lemma": "hra", "upos": "NOUN", "head": 3, "deprel": "obj"},
            {"form": "napsal", "lemma": "napsat", "upos": "VERB", "head": 0, "deprel": "root",
             "feats": {"Gender": "Masc"}},
        ]],
    })
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    a.answer(q1, [])                       # rozsvítí Karla i hru
    text = a.answer(q2, []).text
    assert text == "R.U.R."


def test_predicate_synonyms_from_language_data():
    """„Kde žili?" najde bydlet-fakt — synonymní predikáty jsou jazyková data
    (bydlet/žít/sídlit), přesný predikát má přednost bonusem."""
    g = FactGraph()
    g.add_fact(make_fact("bydlet", [Participant("subj", "Karel Čapek", "person"),
                                    Participant("loc", "Praze", "geo")]))
    q = "Kde žil Karel Čapek?"
    client = FakeUfalClient(parse={q: [[
        {"form": "Kde", "lemma": "kde", "upos": "ADV", "head": 2, "deprel": "advmod"},
        {"form": "žil", "lemma": "žít", "upos": "VERB", "head": 0, "deprel": "root"},
        {"form": "Karel", "lemma": "Karel", "upos": "PROPN", "head": 2, "deprel": "nsubj"},
        {"form": "Čapek", "lemma": "Čapek", "upos": "PROPN", "head": 3, "deprel": "flat"},
    ]]})
    a = GraphAnswerer(g, client, ExtractiveAnswerer(AnswererConfig()))
    assert a.answer(q, []).text == "Praze"   # nominativizaci dělá živá morfologie
