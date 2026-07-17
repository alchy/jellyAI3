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


def _exact_predicate(token, predicates):
    """Tvar je DOSLOVA predikátem grafu (i jako l-kmen: „měli"→„měl").

    Přesná shoda přebíjí čtení entity (entity-first veto platí jen pro
    prefixové odhady) — graf tento predikát opravdu nese, typicky z paměti
    Mnemos, a nesmí ho přehlušit náhodný uzel s volnou kmenovou shodou.
    """
    low = _norm(token)
    normalized = {_norm(p) for p in predicates}
    if low in normalized:
        return True
    stripped = low.rstrip("aioy")
    return stripped.endswith("l") and stripped in normalized


def _verb_match(token, predicates, first=False):
    """Predikát grafu, jehož kmen je prefixem tvaru dotazu (napsal→napsat).

    Český slovesný tvar sdílí s lemmatem počáteční kmen (liší se koncovka);
    shoda = delší prefix pokrývající většinu kratšího slova (min 4 znaky).
    `first=True` = první token věty: velké písmeno tam nese začátek věty
    („Napsal…?"), ne vlastní jméno — guard se neuplatní.
    """
    low = _norm(token)
    # slovesný tvar v otázce je malými písmeny; velké písmeno = vlastní jméno
    # („Němec" nesmí matchnout sloveso „neměnit" přes prefix „nem")
    if low in current()["copula_forms"] or (token[:1].isupper() and not first):
        return None
    # PŘESNÁ shoda s predikátem grafu obchází délkový práh — paměťové
    # predikáty Mnemos jsou krátké l-ové kmeny („měl"); porovná se i kmen
    # l-formy tvaru („měli" → „měl")
    normalized = {_norm(p): p for p in predicates}
    if low in normalized:
        return normalized[low]
    stripped = low.rstrip("aioy")
    if stripped.endswith("l") and stripped in normalized:
        return normalized[stripped]
    if len(low) < 4:
        return None
    # generická dějová slovesa („stalo"→stát) mají tvary v jazykové tabulce —
    # prefix na krátké kmeny nestačí; predikát nemusí být ve slovníku grafu
    # (_event_answer fakt s tímto predikátem nepotřebuje)
    lemma = current()["event_verb_forms"].get(low)
    if lemma:
        return lemma
    best = None
    # páruje se i proti synonymům predikátů (spec 4.2): „žili"→žít, expanzi
    # na bydlet-fakty pak dělá answererův _synonym_ring
    for pred in set(predicates) | set(current()["predicate_synonyms"]):
        p = _norm(pred)
        common = 0
        for a, b in zip(low, p):
            if a != b:
                break
            common += 1
        ok = common >= 4 and common >= min(len(low), len(p)) - 2
        if not ok and len(p) <= 3:
            # krátké lemma („žít"): stačí kmen bez koncové souhlásky, ale tvar
            # musí být l-ové příčestí („zima" ≠ „žít")
            ok = common >= len(p) - 1 and low.rstrip("aeiouy").endswith("l")
        if ok and (best is None or common > best[1]):  # pylint: disable=unsubscriptable-object
            best = (pred, common)
    return best[0] if best else None


def build_query(question, predicates, is_node=None):  # pylint: disable=too-many-locals,too-many-return-statements
    """Otázku (končící „?") přeloží šablonou nad slovníkem grafu na `Query`.

    Args:
        question (str): Dotaz uživatele.
        predicates (set[str]): Predikáty, které graf zná (jeho slovník).
        is_node (callable | None): Přísný test `span → bool`, zda se rozpětí
            rozřeší na uzel grafu (slovník entit je graf — spec 4.3). None =
            bez dělení běhů (celý běh je entita).

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
    if hole_role is None:
        # zjišťovací (ano/ne) otázka (spec 4.5): bez tázacího slova, věta
        # začíná slovesem spárovaným s predikátem grafu; díra žádná —
        # answerer ji vykoná jako existenční test („Ano"/nenašel)
        verb = _verb_match(tokens[0], predicates, first=True)
        if verb is None or (is_node is not None and is_node(tokens[0])):
            return None
        known = _collect_known(tokens[1:], predicates, relational, is_node)
        if not known:                     # None (sirotek) i [] (bez entit)
            return None
        known = [("subj" if i == 0 else "obj", term)
                 for i, (_, term) in enumerate(known)]
        return Query(Pattern(verb, known, None, None), qtype=None,
                     verb_lemma=verb, gender=_verb_gender(tokens[0]))
    copula_tok = next((t for t in tokens
                       if _norm(t) in lang["copula_forms"]), None)
    # hlavní sloveso se hledá PŘED vztažným zájmenem — sloveso vztažné věty
    # patří pod-dotazu („bratr autora, KTERÝ NAPSAL…"), ne hlavnímu vzoru
    rel_idx = next((k for k, t in enumerate(tokens)
                    if k and _norm(t) in lang["relative_pronouns"]),
                   len(tokens))
    # ENTITY-FIRST: tvar, který se rozřeší na uzel grafu, je entita, ne
    # sloveso — „rodinou" se nesmí prefixem spárovat s predikátem „rodit".
    # PŘESNÝ predikát („měl" z paměti Mnemos) ale entita nepřebije.
    verb_tok = next((t for t in tokens[:rel_idx]
                     if _norm(t) not in relational
                     and (_exact_predicate(t, predicates)
                          or not (is_node is not None and is_node(t)))
                     and _verb_match(t, predicates)), None)
    verb = _verb_match(verb_tok, predicates) if verb_tok else None
    # rod nese tvar slovesa (l-ové příčestí), u spony tvar spony („byla")
    gender = _verb_gender(verb_tok or copula_tok or "")
    # „v kterém ROCE" = 2-skokový drill přes časový uzel; tvar → část data
    # z jazykové tabulky, tázací „kterém" pak míří na čas, ne na vlastnost.
    # „v TOMTO roce" drill NENÍ — je to časový filtr (interval řeší Chronos)
    current_time_words = frozenset(
        lang["temporal"].get("current_words", ()))
    date_part = None
    for k, tok in enumerate(tokens):
        if _norm(tok) in lang["date_part_forms"]:
            if k and _norm(tokens[k - 1]) in current_time_words:
                continue
            date_part = lang["date_part_forms"][_norm(tok)]
            break
    if date_part and hole_role == "attr":
        hole_role, hole_type = "time", "time"
    known = _collect_known(tokens, predicates, relational, is_node)
    if known is None:                          # sirotek v běhu — nehádat
        return None

    def _wrap(pattern):
        return Query(pattern, qtype=qtype, verb_lemma=verb,
                     is_copula=copula_tok is not None,
                     topic_terms=_leftover_terms(tokens, pattern, predicates),
                     gender=gender)

    # 1) vztahová otázka: „Kdo byl bratr X?" → vztahové jméno = predikát;
    #    POD SLOVESEM je vztah vnořený dotaz („Kde se narodil bratr X?" →
    #    narodit(subj=SubQuery(bratr, obj=X), díra loc)) — sloveso vládne
    for k, (role, term) in enumerate(list(known)):
        head = term.split()[0] if isinstance(term, str) and term else None
        if head and _norm(head) in relational:
            rest = " ".join(term.split()[1:])
            if not rest:
                continue
            if verb is not None:
                known[k] = ("subj", SubQuery(_norm(head), [("obj", rest)],
                                             "subj"))
            else:
                return _wrap(Pattern(_norm(head), [("obj", rest)],
                                     "subj", "person"))
        if isinstance(term, SubQuery) and verb is None:
            rel = next((t for t in tokens if _norm(t) in relational), None)
            if rel:                            # „bratr autora který napsal X"
                return _wrap(Pattern(_norm(rel), [("obj", term)],
                                     "subj", "person"))

    # 2) predikát ze slovníku grafu (slovesný tvar → lemma prefixem)
    if verb is not None and known:
        if any(_norm(t) in lang["first_person"] for t in tokens):
            # 1. OSOBA („Kdy jsem měl…") — podmětem je IDENTITA UŽIVATELE
            # (uzel Mnemos); pojmenované entity jsou předměty
            known = [("obj", term) if isinstance(term, str) else (r, term)
                     for r, term in known]
            known.append(("subj", lang["user_entity"]))
        elif hole_role == "attr":
            # výběrová otázka (spec 4.6): první known (hned za tázacím slovem)
            # = typový filtr díry (obj), další entity jsou téma (subj) — join
            # napsat(X,?) ∧ druh/být(?,hra) řeší answererův _typed_match
            known = [("obj" if i == 0 else "subj", term)
                     for i, (r, term) in enumerate(known)]
        else:
            # role známé entity je komplement díry: díra subj → entita obj,
            # jinak entita je podmět tématu („Kde se narodil X" → subj=X)
            role = "obj" if hole_role == "subj" else "subj"
            known = [(role if isinstance(term, str) else r, term)
                     for r, term in known]
        return _wrap(Pattern(verb, known, hole_role, hole_type, date_part))

    # 3) sponová identita: „Kdo je X?" / „Jaký je X?"
    if copula_tok is not None and known:
        entity = known[0][1]
        role = "attr" if hole_role == "attr" else "pred"
        return _wrap(Pattern("být", [("subj", entity)], role, hole_type))

    return None


def _trim(run):
    """Okrajová skip-slova rozpětí pryč („Válku s" → „Válku")."""
    skip = current()["query_skip_words"]
    while run and _norm(run[0]) in skip:
        run = run[1:]
    while run and _norm(run[-1]) in skip:
        run = run[:-1]
    return run


def _split_run(run, is_node, relational):
    """Greedy longest-match: běh → maximální `is_node` rozpětí (spec 4.3).

    Vedoucí sirotek (první obsahové slovo bez shody) = nebezpečný vzor → None;
    sirotek PO shodě je pokračování titulu → zahodit. Vztahové jméno na začátku
    se přilepí k první entitě (vztahovou šablonu řeší build_query)."""
    run = _trim(list(run))
    if not run:
        return []
    rel = run[0] if _norm(run[0]) in relational else None
    if rel:
        run = _trim(run[1:])
        if not run:
            return [("obj", rel)]
    if is_node is None:
        parts = [("obj", " ".join(run))]
    else:
        skip = current()["query_skip_words"]
        parts, i, matched = [], 0, False
        while i < len(run):
            if _norm(run[i]) in skip:
                if matched:
                    # předložková fráze PO entitě je pokračování jejího titulu
                    # („Válku S MLOKY" — mlok je uzel, ale ne účastník otázky)
                    break
                i += 1
                continue
            hit = None
            for j in range(len(run), i, -1):
                if _norm(run[j - 1]) in skip:
                    continue             # rozpětí nesmí končit předložkou
                span = run[i:j]
                if is_node(" ".join(span)):
                    hit = (j, " ".join(span))
                    break
            if hit is None:
                if not matched:
                    return None          # vedoucí sirotek — nikdy chybný Pattern
                i += 1                   # koncový sirotek po shodě → zahodit
                continue
            parts.append(("obj", hit[1]))
            matched, i = True, hit[0]
        if not parts:
            return None
    if rel:
        parts[0] = ("obj", rel + " " + parts[0][1])
    return parts


def _collect_known(tokens, predicates, relational, is_node):
    """Známé účastníky = spojité běhy obsahových tokenů (kandidáti na entitu/
    vztah); tázací/spona/sloveso běh ukončí, „který" → pod-dotaz; skip-slova
    zůstávají UVNITŘ běhu (titul „Válka s mloky" drží pohromadě) a běh se
    dělí greedy na uzlová rozpětí (`_split_run`). Vztahové jméno se ponechá
    NA ZAČÁTKU běhu (řeší ho vztahová šablona). None = sirotek (nehádat)."""
    lang = current()
    known, run = [], []

    def flush():
        parts = _split_run(run, is_node, relational)
        if parts is None:
            return False
        known.extend(parts)
        run.clear()
        return True

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = _norm(tok)
        if low in lang["relative_pronouns"] and i:   # „…autora KTERÝ napsal X"
            if not flush():
                return None
            sub = _subquery(tokens[i + 1:], predicates)
            if sub is not None:
                if known:
                    known[-1] = (known[-1][0], sub)
                else:
                    known.append(("obj", sub))
            break
        boundary = (low in lang["interrogatives"] or low in lang["copula_forms"]
                    or low in lang["date_part_forms"]
                    or (_verb_match(tok, predicates) is not None
                        and (_exact_predicate(tok, predicates)
                             or not (is_node is not None and is_node(tok)))))
        if boundary and low not in relational:
            if not flush():
                return None
            i += 1
            continue
        run.append(tok)
        i += 1
    if run and not flush():
        return None
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
