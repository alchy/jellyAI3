"""Template query parser — otázka → vzor → pseudo-QL Pattern, bez UDPipe.

Slovník dotazu je sám graf (predikáty + uzly); diakritika ani mis-tagging
UDPipe nehrají roli. Korpus jen staví graf; dotaz jede šablonami.
"""

from jellyai.answerer.query import build_query
from jellyai.answerer.pattern import SubQuery


# slovník grafu, který parser vidí (predikáty, které fakty nesou)
PREDS = {"napsat", "narodit", "být", "bratr", "zemřít"}


def test_kdo_predicate_entity():
    """"Kdo napsal Babičku?" → predikát napsat, known obj=Babička, díra subj."""
    pat = build_query("Kdo napsal Babičku?", PREDS)
    assert pat.predicate == "napsat"
    assert pat.hole_role == "subj"
    assert ("obj", "Babičku") in pat.known


def test_kde_diacritic_free():
    """Bez diakritiky: „Kde se narodil Jezis?" → narodit, díra loc."""
    pat = build_query("Kde se narodil Jezis?", PREDS)
    assert pat.predicate == "narodit"
    assert pat.hole_role == "loc"
    assert ("subj", "Jezis") in pat.known


def test_kdo_je_identity():
    """"Kdo je Karel Capek?" → identita být, díra pred."""
    pat = build_query("Kdo je Karel Capek?", PREDS)
    assert pat.predicate == "být"
    assert pat.hole_role == "pred"
    assert ("subj", "Karel Capek") in pat.known


def test_relational_recursion():
    """"Kdo byl bratr Karla Capka?" → predikát bratr (vztahové jméno), díra
    subj, known obj = Karel Capek (rekurzivní předmět)."""
    pat = build_query("Kdo byl bratr Karla Capka?", PREDS)
    assert pat.predicate == "bratr"
    assert pat.hole_role == "subj"
    assert ("obj", "Karla Capka") in pat.known


def test_nested_subquery():
    """"Kdo byl bratr autora ktery napsal RUR?" → bratr(obj=SubQuery(napsat,
    obj=RUR)) — dotaz expanduje rekurzivně přes vztažné „který"."""
    pat = build_query("Kdo byl bratr autora ktery napsal RUR?", PREDS)
    assert pat.predicate == "bratr"
    sub = next(k for _, k in pat.known if isinstance(k, SubQuery))
    assert sub.predicate == "napsat"
    assert ("obj", "RUR") in sub.known


def test_non_question_returns_none():
    """Věta bez otazníku není dotaz → None (parser řeší jen otázky)."""
    assert build_query("Karel napsal Babičku.", PREDS) is None
