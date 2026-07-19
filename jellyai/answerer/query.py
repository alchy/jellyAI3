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
    place: str = None      # oblast otázky („v Čechách") — filtr Topos


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
    # NEGAČNÍ PÁR (#24): tvar je slovesem, i když graf zná jen jeho OPAK
    # („Prší?" nad pamětí s jediným faktem `neprší`); vrací se protějšek
    # bez/s prefixem — párování evidence dělá answererova _existence.
    # Jádro páru MUSÍ mít ≥ 3 znaky (konvence _l_form/_finite_verb) — jinak
    # šumový predikát „nes" udělá z předložky „s" sloveso a rozbije span
    prefix = current().get("negation_prefix", "")
    if prefix and len(low) >= 3 and prefix + low in normalized:
        return normalized[prefix + low][len(prefix):]
    if prefix and low.startswith(prefix) and len(low) - len(prefix) >= 3 \
            and low[len(prefix):] in normalized:
        return prefix + normalized[low[len(prefix):]]
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
        if not ok and low.rstrip("aeiouy").endswith("l"):
            # paměťový predikát v PRÉZENTU („prší") vs. l-ové příčestí otázky
            # („Pršelo") — kmen musí pokrýt celé lemma kromě koncové samohlásky
            ok = common >= max(3, len(p) - 1)
        if ok and (best is None or common > best[1]):  # pylint: disable=unsubscriptable-object
            best = (pred, common)
    return best[0] if best else None


_QUERY_DECK = None


def _query_deck():
    """Balíček karet se vzorovými dotazy (lazy, čte se jednou za proces)."""
    global _QUERY_DECK    # pylint: disable=global-statement
    if _QUERY_DECK is None:
        from jellyai.iris.patterns import PatternDeck
        deck = PatternDeck.for_language("cs")
        deck.load()
        _QUERY_DECK = deck
    return _QUERY_DECK


def _card_query(question, predicates, is_node=None, is_word=None):
    """Dotaz podle VZOROVÉ KARTY (#46 fáze 2): regulární sekvence tříd
    lexeru na kartě (event `utterance.query`) → pseudo-QL `Pattern`.

    Nový tázací tvar = nová karta, žádný Python. Vybírá se nejtěsnější
    match (priorita, délka vzoru); predikát prochází TOUŽ normalizací
    jako šablony (`_verb_match`) — zápis a dotaz se potkají. Díru nese
    tabulka `interrogatives` (role, typ) přes tázací token vzoru;
    víceslovné entity dělí spanový prvek `uzel+` orákulem grafu
    (`is_node`). Karta bez naplněných známých = holá existence.

    Returns:
        Query | None: Rozbor, nebo None (žádná karta nesedí → šablony).
    """
    from jellyai.lang.lexer import classify
    from jellyai.lang.matcher import match_sequence
    tagged = classify(question, is_node=is_word)
    lang = current()
    best = None
    for card in _query_deck().cards:
        if card.trigger.get("event") != "utterance.query":
            continue
        sequence = card.trigger.get("pattern")
        if not sequence:
            continue
        binding = match_sequence(sequence, tagged, is_span=is_node)
        if binding is None:
            continue
        key = (card.trigger.get("priority", 0), len(sequence))
        if best is None or key > best[0]:
            best = (key, card, binding)
    if best is None:
        return None
    _, card, binding = best
    spec = card.action.get("query", {})

    def ref(value):
        if isinstance(value, str) and value.startswith("$"):
            return binding.get(int(value[1:]))
        return None

    def surface(bound):
        if isinstance(bound, list):
            return " ".join(tok.form for tok in bound)
        return bound.form if bound is not None else None

    pattern = Pattern()
    qtype = None
    hole = ref(spec.get("hole"))
    if hole is not None:
        entry = lang["interrogatives"].get(hole.norm)
        if entry:
            pattern.hole_role, pattern.hole_type, qtype = entry
    verb = ref(spec.get("predicate"))
    if verb is not None:
        pattern.predicate = _verb_match(surface(verb), predicates,
                                        first=True)
        if pattern.predicate is None:
            # NEZNÁMÉ sloveso (překlep „sworil") — karta se nehlásí;
            # otázka jde šablonou/UDPipe fallbackem, který má vlastní léky
            return None
    # role známého plyne z díry (subj-díra → známý je obj, jinak subj);
    # explicitní pár ["role", "$N"] na kartě má přednost
    default_role = "obj" if pattern.hole_role == "subj" else "subj"
    for known_ref in spec.get("known", ()):
        role, value = (known_ref if isinstance(known_ref, list)
                       else (default_role, known_ref))
        term = surface(ref(value))
        if term is not None:
            pattern.known.append((role, term))
    if pattern.predicate is None:
        return None
    return Query(pattern=pattern, qtype=qtype, verb_lemma=pattern.predicate)


def build_query(question, predicates, is_node=None, is_word=None,
                is_area=None):  # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches
    """Otázku (končící „?") přeloží šablonou nad slovníkem grafu na `Query`.

    Args:
        question (str): Dotaz uživatele.
        predicates (set[str]): Predikáty, které graf zná (jeho slovník).
        is_node (callable | None): Přísný test `span → bool`, zda se rozpětí
            rozřeší na uzel grafu (slovník entit je graf — spec 4.3). None =
            bez dělení běhů (celý běh je entita).
        is_word (callable | None): DOSLOVNÉ slovo některého uzlu (bez
            kmenových pater) — veto pro čtení neznámého slovesa; volná
            shoda by sloveso zabila šumem („Měl"≈uzel „mle").

    Returns:
        Query | None: Pseudo-QL pattern + šablonová analýza (qtype/rod/spona);
        None když věta není otázka nebo z ní nelze vzor bezpečně sestavit
        (pak volající spadne na UDPipe fallback, dokud existuje).
    """
    if "?" not in question:
        return None
    # VZOROVÉ KARTY mají přednost (#46 fáze 2): plně ukotvený match je
    # těsnější než poziční šablony; nesedí-li žádná, jede se postaru
    card_query = _card_query(question, predicates, is_node, is_word)
    if card_query is not None:
        return card_query
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
    if hole_role is not None \
            and any(_norm(t) in lang["relation_query_nouns"] for t in tokens):
        # VZTAHOVÁ OTÁZKA („Jaký měl vztah k Janovi?"): „vztah" není entita,
        # ale OPERÁTOR spojení — odpovědí jsou fakty sdílené oběma uzly;
        # elidovaného druhého účastníka doplní answerer z těžiště. Sloveso
        # otázky (l-příčestí, které není uzlem) tu nenese nic — pryč s ním,
        # jinak je vedoucím sirotkem běhu
        # skip-slova (předložky) pryč: „k Janovi" je DRUHÝ účastník, ne
        # pokračování titulu prvního — pravidlo polykání tu neplatí.
        # Veto l-tvaru je DOSLOVNÉ (is_word) — volná shoda by sloveso
        # nechala žít jako šumový uzel („měl"≈„mle")
        rest = [t for t in tokens
                if _norm(t) not in lang["relation_query_nouns"]
                and _norm(t) not in lang["query_skip_words"]
                and not (t[:1].islower()               # sloveso je malými —
                         and t.rstrip("aioy").endswith("l")  # „Křtiteli" ne!
                         and not (is_word is not None and is_word(t)))]
        known = _collect_known(rest, predicates, relational, is_node)
        if known:
            known = [("obj", term) for _, term in known]
            return Query(Pattern(None, known, "relation", None),
                         qtype=qtype)
    if hole_role is None:
        # zjišťovací (ano/ne) otázka (spec 4.5): bez tázacího slova, věta
        # začíná slovesem spárovaným s predikátem grafu; díra žádná —
        # answerer ji vykoná jako existenční test („Ano"/nenašel)
        verb = _verb_match(tokens[0], predicates, first=True)
        if verb is not None:
            if is_node is not None and is_node(tokens[0]) \
                    and not _exact_predicate(tokens[0], predicates):
                return None
        elif _norm(tokens[0]) not in lang["copula_forms"]:
            # NEZNÁMÉ sloveso: l-ové příčestí na začátku zjišťovací otázky
            # („Měl … rád knedlíky?" před prvním záznamem paměti) nese
            # existenční dotaz i bez slovníku grafu — odpoví se poctivé
            # „nenašel", ale entity otázky se rozřeší a ROZSVÍTÍ, takže
            # navazující potvrzení („ano, měl rád knedlíky.") ví, o kom
            # byla řeč. Veto entity je tu DOSLOVNÉ (`is_word`).
            stem = tokens[0].lower().rstrip("aioy")
            if stem.endswith("l") and len(stem) >= 3 \
                    and not (is_word is not None and is_word(tokens[0])):
                verb = stem
        if verb is None:
            return None
        # časová slova nejsou účastníci — filtr intervalu drží Chronos;
        # OBLAST („v Čechách") stejně tak — filtr kontejnmentu drží Topos
        temporal = {w for key in ("day_words", "units", "now_words",
                                  "current_words")
                    for w in lang["temporal"].get(key, ())}
        rest = [t for t in tokens[1:] if _norm(t) not in temporal]
        place = None
        if is_area is not None:
            place = next((t for t in rest if is_area(t)), None)
            if place is not None:
                rest = [t for t in rest if t != place]
        known = _collect_known(rest, predicates, relational, is_node)
        if known is None:                 # sirotek v běhu — nehádat
            return None
        if not known and _verb_match(tokens[0], predicates,
                                     first=True) is None:
            return None    # bez entit i bez známého predikátu není co testovat
        known = [("subj" if i == 0 else "obj", term)
                 for i, (_, term) in enumerate(known)]
        return Query(Pattern(verb, known, None, None), qtype=None,
                     verb_lemma=verb, gender=_verb_gender(tokens[0]),
                     place=place)
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
        parts, i, matched, titled = [], 0, False, False
        while i < len(run):
            if _norm(run[i]) in skip:
                if matched and titled:
                    # předložková fráze PO KAPITALIZOVANÉ entitě je pokračování
                    # jejího titulu („Válku S MLOKY" — mlok je uzel, ale ne
                    # účastník otázky); po obecném slově je to NOVÝ účastník
                    # („vztah K MARTĚ" — Marta se nesmí spolknout)
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
            titled = hit[1][:1].isupper()
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
