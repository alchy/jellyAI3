"""Template query parser — otázka → vzor → pseudo-QL Query, bez UDPipe.

Slovník dotazu je sám graf (predikáty + uzly); diakritika ani mis-tagging
UDPipe nehrají roli. Korpus jen staví graf; dotaz jede šablonami.
"""

from jellyai.answerer.query import build_query
from jellyai.answerer.pattern import SubQuery


# slovník grafu, který parser vidí (predikáty, které fakty nesou)
PREDS = {"napsat", "narodit", "být", "bratr", "zemřít"}


def test_kdo_predicate_entity():
    """"Kdo napsal Babičku?" → predikát napsat, known obj=Babička, díra subj."""
    q = build_query("Kdo napsal Babičku?", PREDS)
    assert q.pattern.predicate == "napsat"
    assert q.pattern.hole_role == "subj"
    assert ("obj", "Babičku") in q.pattern.known
    assert q.qtype == "Kdo" and q.verb_lemma == "napsat"
    assert q.gender == "Masc"          # „napsal" — l-ové příčestí mužské


def test_kde_diacritic_free():
    """Bez diakritiky: „Kde se narodil Jezis?" → narodit, díra loc."""
    q = build_query("Kde se narodil Jezis?", PREDS)
    assert q.pattern.predicate == "narodit"
    assert q.pattern.hole_role == "loc"
    assert ("subj", "Jezis") in q.pattern.known


def test_kdo_je_identity():
    """"Kdo je Karel Capek?" → identita být, díra pred."""
    q = build_query("Kdo je Karel Capek?", PREDS)
    assert q.pattern.predicate == "být"
    assert q.pattern.hole_role == "pred"
    assert ("subj", "Karel Capek") in q.pattern.known


def test_relational_recursion():
    """"Kdo byl bratr Karla Capka?" → predikát bratr (vztahové jméno), díra
    subj, known obj = Karel Capek (rekurzivní předmět)."""
    q = build_query("Kdo byl bratr Karla Capka?", PREDS)
    assert q.pattern.predicate == "bratr"
    assert q.pattern.hole_role == "subj"
    assert ("obj", "Karla Capka") in q.pattern.known


def test_nested_subquery():
    """"Kdo byl bratr autora ktery napsal RUR?" → bratr(obj=SubQuery(napsat,
    obj=RUR)) — dotaz expanduje rekurzivně přes vztažné „který"."""
    q = build_query("Kdo byl bratr autora ktery napsal RUR?", PREDS)
    assert q.pattern.predicate == "bratr"
    sub = next(k for _, k in q.pattern.known if isinstance(k, SubQuery))
    assert sub.predicate == "napsat"
    assert ("obj", "RUR") in sub.known


def test_non_question_returns_none():
    """Věta bez otazníku není dotaz → None (parser řeší jen otázky)."""
    assert build_query("Karel napsal Babičku.", PREDS) is None


def test_query_gender_from_verb_form():
    q = build_query("Kdy se narodila Božena Němcová?", {"narodit"})
    assert q.gender == "Fem" and q.qtype == "Kdy"


def test_query_copula_flag():
    q = build_query("Kdo je Karel Capek?", PREDS)
    assert q.is_copula is True and q.verb_lemma is None
