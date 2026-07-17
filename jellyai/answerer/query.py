"""Šablonový parser otázek → pseudo-QL `Pattern`, bez UDPipe.

Mechanismus: **otázka → identifikace vzoru → transformace do pseudo query
language → dotaz do grafu.** Slovník dotazu je **sám graf** (predikáty, které
fakty nesou) plus jazykové tabulky (tázací slova, spona, vztahová jména,
synonyma). Diakritika ani mis-tagging ML parseru nehrají roli — korpus staví
graf, dotaz jede deterministickými šablonami.

Slovesné tvary se párují s predikáty grafu **prefixem** (napsal≡napsat,
narodil≡narodit) — český tvar se liší až koncovkou, kmen (prefix) drží.
`Pattern`/`SubQuery` jsou sdílené s `pattern.py`, takže answerer se nemění.
"""

import re

from dataclasses import dataclass, field

from jellyai.answerer.pattern import Pattern, SubQuery
from jellyai.graph.canon import deaccent
from jellyai.lang import current


def _norm(token):
    """Bezdiakritický klíč tokenu malými písmeny."""
    return deaccent(token.lower())


def _verb_gender(form):
    """Rod z l-ového příčestí: „narodila"→Fem, „narodil"→Masc, jinak None."""
    low = form.lower()
    if low.endswith("la"):
        return "Fem"
    if low.endswith("l"):
        return "Masc"
    return None


@dataclass
class Query:
    """Šablonový rozbor otázky: pseudo-QL pattern + signály pro answerer.

    Duck-type náhrada `QuestionAnalysis` (stejná jména atributů) — po přepnutí
    na šablony answerer nepotřebuje UDPipe ani na qtype/rod/sponu."""
    pattern: Pattern = None
    qtype: str = None
    verb_lemma: str = None
    is_copula: bool = False
    topic_terms: list = field(default_factory=list)
    gender: str = None


def _known_words(known):
    """Slova zachycená ve známých termech (rekurzivně přes SubQuery)."""
    words = set()
    for _, term in known:
        if isinstance(term, SubQuery):
            words |= _known_words(term.known)
        elif isinstance(term, str):
            words.update(term.split())
    return words


def _leftover_terms(tokens, pattern, predicates):
    """Obsahové tokeny, které vzor nezachytil — guard proti hádání z kontextu."""
    lang = current()
    used = _known_words(pattern.known)
    if pattern.predicate:
        used.add(pattern.predicate)
    out = []
    for tok in tokens:
        low = _norm(tok)
        if (low in lang["interrogatives"] or low in lang["copula_forms"]
                or low in lang["query_skip_words"]
                or low in lang["relative_pronouns"]
                or low in lang["date_part_forms"]
                or _verb_match(tok, predicates) is not None):
            continue
        if tok in used:
            continue
        out.append(tok)
    return out


def _verb_match(token, predicates):
    """Predikát grafu, jehož kmen je prefixem tvaru dotazu (napsal→napsat).

    Český slovesný tvar sdílí s lemmatem počáteční kmen (liší se koncovka);
    shoda = delší prefix pokrývající většinu kratšího slova (min 4 znaky).
    """
    low = _norm(token)
    # slovesný tvar v otázce je malými písmeny; velké písmeno = vlastní jméno
    # („Němec" nesmí matchnout sloveso „neměnit" přes prefix „nem")
    if len(low) < 4 or low in current()["copula_forms"] or token[:1].isupper():
        return None
    best = None
    for pred in predicates:
        p = _norm(pred)
        common = 0
        for a, b in zip(low, p):
            if a != b:
                break
            common += 1
        if common >= 4 and common >= min(len(low), len(p)) - 2:
            if best is None or common > best[1]:  # pylint: disable=unsubscriptable-object
                best = (pred, common)
    return best[0] if best else None


def build_query(question, predicates):  # pylint: disable=too-many-locals,too-many-return-statements
    """Otázku (končící „?") přeloží šablonou nad slovníkem grafu na `Query`.

    Args:
        question (str): Dotaz uživatele.
        predicates (set[str]): Predikáty, které graf zná (jeho slovník).

    Returns:
        Query | None: Pseudo-QL pattern + šablonová analýza (qtype/rod/spona);
        None když věta není otázka nebo z ní nelze vzor bezpečně sestavit
        (pak volající spadne na UDPipe fallback, dokud existuje).
    """
    if "?" not in question:
        return None
    tokens = re.findall(r"[\w.]+", question)
    if not tokens:
        return None

    lang = current()
    hole_role, hole_type, qtype = None, None, None
    for tok in tokens:                         # díra = první tázací slovo
        entry = lang["interrogatives"].get(_norm(tok))
        if entry:
            hole_role, hole_type, qtype = entry
            break

    relational = lang["relational_nouns"]
    copula_tok = next((t for t in tokens
                       if _norm(t) in lang["copula_forms"]), None)
    verb_tok = next((t for t in tokens if _verb_match(t, predicates)), None)
    verb = _verb_match(verb_tok, predicates) if verb_tok else None
    # rod nese tvar slovesa (l-ové příčestí), u spony tvar spony („byla")
    gender = _verb_gender(verb_tok or copula_tok or "")
    known = _collect_known(tokens, predicates, relational)

    def _wrap(pattern):
        return Query(pattern, qtype=qtype, verb_lemma=verb,
                     is_copula=copula_tok is not None,
                     topic_terms=_leftover_terms(tokens, pattern, predicates),
                     gender=gender)

    # 1) vztahová otázka: „Kdo byl bratr X?" → vztahové jméno = predikát
    for role, term in list(known):
        head = term.split()[0] if isinstance(term, str) and term else None
        if head and _norm(head) in relational:
            rest = " ".join(term.split()[1:])
            if rest:
                return _wrap(Pattern(_norm(head), [("obj", rest)],
                                     "subj", "person"))
        if isinstance(term, SubQuery):         # „bratr autora který napsal X"
            rel = next((t for t in tokens if _norm(t) in relational), None)
            if rel:
                return _wrap(Pattern(_norm(rel), [("obj", term)],
                                     "subj", "person"))

    # 2) predikát ze slovníku grafu (slovesný tvar → lemma prefixem)
    if verb is not None and known:
        # role známé entity je komplement díry: díra subj → entita obj,
        # jinak entita je podmět tématu („Kde se narodil X" → subj=X, díra loc)
        role = "obj" if hole_role == "subj" else "subj"
        known = [(role if isinstance(term, str) else r, term)
                 for r, term in known]
        return _wrap(Pattern(verb, known, hole_role, hole_type))

    # 3) sponová identita: „Kdo je X?" / „Jaký je X?"
    if copula_tok is not None and known:
        entity = known[0][1]
        role = "attr" if hole_role == "attr" else "pred"
        return _wrap(Pattern("být", [("subj", entity)], role, hole_type))

    return None


def _collect_known(tokens, predicates, relational):
    """Známé účastníky = spojité běhy obsahových tokenů (kandidáti na entitu/
    vztah); tázací/spona/předložka/sloveso běh ukončí, „který" → pod-dotaz.
    Vztahové jméno se ponechá NA ZAČÁTKU běhu (řeší ho vztahová šablona)."""
    lang = current()
    known, run = [], []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = _norm(tok)
        if low in lang["relative_pronouns"]:   # „…autora KTERÝ napsal X"
            if run:
                known.append(("obj", " ".join(run)))
                run = []
            sub = _subquery(tokens[i + 1:], predicates)
            if sub is not None:
                if known:
                    known[-1] = (known[-1][0], sub)
                else:
                    known.append(("obj", sub))
            break
        boundary = (low in lang["interrogatives"] or low in lang["copula_forms"]
                    or low in lang["query_skip_words"]
                    or _verb_match(tok, predicates) is not None)
        if boundary and low not in relational:
            if run:
                known.append(("obj", " ".join(run)))
                run = []
            i += 1
            continue
        run.append(tok)
        i += 1
    if run:
        known.append(("obj", " ".join(run)))
    return known


def _subquery(rest, predicates):
    """Z „…který PREDIKÁT PŘEDMĚT" složí SubQuery(predikát, obj=předmět)."""
    lang = current()
    verb, obj = None, []
    for tok in rest:
        match = _verb_match(tok, predicates)
        if verb is None and match:
            verb = match
        elif verb is not None and _norm(tok) not in lang["copula_forms"] \
                and _norm(tok) not in lang["query_skip_words"]:
            obj.append(tok)
    if verb is None or not obj:
        return None
    return SubQuery(verb, [("obj", " ".join(obj))], "subj")
