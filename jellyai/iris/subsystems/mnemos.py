"""Mnemos — paměť Iris: časově vázaná komunikace uživatele v grafu.

Uživatel nemluví jen v otázkách. Konstatování („Dnes jsem měl knedlíky.")
je sdělení do PAMĚTI — Mnemos ho uloží jako běžný fakt grafu, kde:

* **uživatel je entita** (uzel `user_entity` z jazykových dat) — jeho výroky
  se vážou na jeho identitu jako u kterékoli jiné osoby grafu;
* **čas se ukotvuje HNED** (Chronos): „dnes" se přeloží na absolutní datum
  v okamžiku uložení — paměť nesmí držet relativní slovo, zítra by
  znamenalo jiný den;
* fakt žije v TÉMŽE grafu jako korpus — otázka „Kdy jsem měl v tomto roce
  knedlíky?" pak jede běžnou cestou (pseudo-QL → match → díra time)
  a Mnemos s Chronosem jen pomohly správné aktivaci.

První osoba se pozná podle pomocného slovesa („jsem/sem/jsme" — tabulka
`first_person`); predikát je l-ové příčestí věty (uložené v kmenovém tvaru
bez koncovky rodu/čísla, aby se „měl/měla/měli" potkaly).
"""

import json
import os
import re

from jellyai.graph.canon import deaccent
from jellyai.graph.extract import make_fact, Participant
from jellyai.iris.subsystems.chronos import resolve_temporal
from jellyai.lang import current


def _l_form(token):
    """Kmen l-ového příčestí („měl"/„měla"/„měli" → „měl"); jinak None."""
    low = token.lower()
    stripped = low.rstrip("aioy")
    return stripped if stripped.endswith("l") and len(stripped) >= 3 else None


def _date_label(interval):
    """Interval → povrch časového uzlu grafu („17. července 2026").

    Genitivy měsíců jsou jazyková data; formát odpovídá datům z korpusu,
    takže `parse_date` i dekompozice fungují nad pamětí stejně jako nad texty.
    """
    start = interval.start
    month = current()["temporal"]["month_genitives"][start.month - 1]
    return f"{start.day}. {month} {start.year}"


def _finite_verb(tokens, norms, is_node=None):
    """Prézentní sloveso výroku („prší") — koncovky z jazykové tabulky.

    ENTITY-FIRST: tvar, který se rozřeší na uzel grafu („nádraží" končí -í,
    ale je to věc, ne děj), slovesem není. Funkční slova se přeskakují.

    Returns:
        str | None: Slovesný tvar (povrch), nebo None.
    """
    lang = current()
    endings = lang["present_verb_endings"]
    for tok, norm in zip(tokens, norms):
        if len(tok) < 3 or norm in lang["first_person"] \
                or norm in lang["copula_forms"] \
                or norm in lang["query_skip_words"]:
            continue
        if not tok.lower().endswith(endings):
            continue
        if is_node is not None and is_node(tok):
            continue
        return tok
    return None


def utterance_features(tokens, norms, is_node=None):
    """RYSY výroku pro kartové triggery — kód nerozhoduje, jen měří.

    Returns:
        set[str]: Podmnožina {"first_person", "copula", "l_verb",
        "finite_verb"}.
    """
    lang = current()
    features = set()
    if any(n in lang["first_person"] for n in norms):
        features.add("first_person")
    if any(n in lang["copula_forms"] for n in norms):
        features.add("copula")
    finite = _finite_verb(tokens, norms, is_node)
    if finite is not None:
        features.add("finite_verb")
    # l-příčestí: jméno „Marcela" po ořezu vypadá jako l-tvar — KAPITALIZOVANÝ
    # začátek věty se počítá, jen když věta nemá prézentní sloveso („Pršelo…")
    if any(_l_form(t) and t != finite
           and (t[:1].islower() or (i == 0 and finite is None))
           for i, t in enumerate(tokens)):
        features.add("l_verb")
    return features


def parse_statement(text, now, deck=None, is_node=None):
    """Rozpozná konstatování a rozloží ho na časově ukotvený fakt paměti.

    O DRUHU konstatování nerozhoduje kód, ale **karty** (ZÁKON: logika se
    nestaví fixně programově): kód spočítá rysy výroku (1. osoba, spona,
    l-příčestí) a balíček karet vybere vzor události `utterance.statement`
    — jeho akce určí druh (`memorize`), predikát i filtrování objektů.
    Přidání karty = nový rozpoznávaný tvar konstatování, bez zásahu do kódu.

    Timestamp se přidává VŽDY — i bez časového slova platí čas výroku
    (interakce je časově vázaná); explicitní primitivum kotvu posune.

    Args:
        text (str): Vstup uživatele.
        now (datetime): Okamžik „teď" (Chronos kotva — zvenku).
        deck (PatternDeck | None): Karty; None = vestavěné karty jazyka.

    Returns:
        dict | None: {"kind", "predicate", "objects", "time", "card"};
        None když žádná karta výrok nerozpozná (dotaz, holá entita…).
    """
    if "?" in text:
        return None
    if deck is None:
        from jellyai.iris.patterns import PatternDeck
        deck = PatternDeck.for_language("cs")
        deck.load()
    lang = current()
    # koncová větná tečka pryč; tečkované zkratky (R.U.R.) zůstávají
    tokens = [t.rstrip(".") if "." not in t[:-1] else t
              for t in re.findall(r"[\w.]+", text)]
    tokens = [t for t in tokens if t]
    norms = [deaccent(t.lower()) for t in tokens]
    card = deck.best("utterance.statement",
                      {"features": utterance_features(tokens, norms, is_node)})
    if card is None or "memorize" not in card.action:
        return None
    kind = card.action["memorize"]
    source = card.action.get("predicate_from")
    if source == "l_verb":
        predicate = next((_l_form(t) for t in tokens if _l_form(t)), None)
    elif source == "finite_verb":
        predicate = _finite_verb(tokens, norms, is_node)
    else:
        predicate = card.action.get("predicate")
    if predicate is None:
        return None
    interval = resolve_temporal(text, now)
    time_label = None
    if interval is None:
        interval = resolve_temporal("dnes", now)   # čas výroku = dnešek
        if card.action.get("time_granularity") == "moment":
            # DĚJ se kotví na OKAMŽIK výroku — den nestačí („Venku prší"
            # v 14:32 ≠ celý den); přesnost určuje karta, ne kód
            time_label = f"{_date_label(interval)} {now.hour}:{now.minute:02d}"
    if time_label is None:
        time_label = _date_label(interval)
    temporal_words = (set(lang["temporal"].get("day_words", ()))
                      | set(lang["temporal"].get("units", ()))
                      | set(lang["temporal"].get("now_words", ())))
    exclude_l = card.action.get("exclude_l_forms", False)
    objects = [tok for tok, norm in zip(tokens, norms)
               if norm not in lang["first_person"]
               and norm not in lang["copula_forms"]
               and norm not in lang["query_skip_words"]
               and norm not in lang["confirmation_words"]
               and norm not in temporal_words
               and not (exclude_l and _l_form(tok) is not None)
               and tok != predicate            # sloveso není účastník
               and len(tok) > 1]
    if not objects or (kind == "observation" and len(objects) < 2):
        return None
    # MÍSTA (brána E Toposu v dialogu): objekt za předložkou „v/ve/na"
    # je místo — dostane roli loc/geo („Marcela bydlí V PETROVICÍCH"),
    # aby „Kde bydlí…?" i kontejnment („v Čechách?") měly za co vzít
    places = []
    for i, tok in enumerate(tokens[:-1]):
        if deaccent(tok.lower()) in ("v", "ve", "na") \
                and tokens[i + 1] in objects:
            places.append(tokens[i + 1])
    return {"kind": kind, "predicate": predicate, "objects": objects,
            "places": places, "time": time_label, "card": card.name,
            "needs_subject": card.action.get("subject_from") == "context"}


def note_statement(text, now):
    """POZNÁMKA — explicitní příkaz paměti bez rozpoznatelné struktury
    („Ulož si, co se škádlívá, to se rádo má."): text se uchová DOSLOVNĚ
    jako výrok uživatele s časovou kotvou. Příkaz řekl „pamatuj" —
    persistence náleží i příslovím."""
    interval = resolve_temporal("dnes", now)
    return {"kind": "note", "predicate": current()["note_predicate"],
            "objects": [text], "places": [], "time": _date_label(interval),
            "card": "memory-note", "needs_subject": False}


def remember(graph, statement, user_entity):
    """Uloží rozložené konstatování do grafu jako časově ukotvený fakt.

    Epizoda: subj = uživatel, předměty obj. Pozorování: subj = první obsahové
    slovo, zbytek pred (sponová sémantika), uživatel jako pozorovatel (theme).

    Args:
        graph (FactGraph): Cílový graf (týž jako korpusový).
        statement (dict): Výstup `parse_statement`.
        user_entity (str): Id uzlu identity uživatele.

    Returns:
        str: Lidský popis uloženého faktu (pro potvrzení v dialogu).
    """
    objects = statement["objects"]
    places = set(statement.get("places", ()))

    def _part(role, node, typ):
        if node in places:
            return Participant("loc", node, "geo")
        return Participant(role, node, typ)
    if statement["kind"] == "note":
        participants = [Participant("subj", user_entity, "person"),
                        Participant("pred", objects[0], "výrok")]
    elif statement["kind"] == "episode":
        participants = [Participant("subj", user_entity, "person")]
        participants += [_part("obj", obj, "concept") for obj in objects]
    elif statement["kind"] == "attributed":
        # fakt PŘIPSANÝ korpusové osobě („ano, měl rád knedlíky" → subjekt
        # z těžiště/výroku doplnil automat); uživatel je zdroj (theme)
        participants = [Participant("subj", statement["subject"], "person")]
        participants += [_part("obj", obj, "concept") for obj in objects]
        participants.append(Participant("theme", user_entity, "person"))
    elif statement["kind"] == "event":
        # prézentní děj („Venku prší"): účastníci jsou obsahová slova,
        # uživatel je pozorovatel — zjišťovací „Prší venku?" pak sedí
        participants = [_part("obj", obj, "concept") for obj in objects]
        participants.append(Participant("theme", user_entity, "person"))
    else:
        participants = [Participant("subj", objects[0], "concept")]
        participants += [_part("pred", obj, "concept")
                         for obj in objects[1:]]
        participants.append(Participant("theme", user_entity, "person"))
    participants.append(Participant("time", statement["time"], "time"))
    graph.add_fact(make_fact(statement["predicate"], participants))
    detail = f"{statement['predicate']}: {', '.join(objects)} ({statement['time']})"
    if statement["kind"] == "attributed":
        detail = f"{statement['subject']} — {detail}"
    return detail


def forget(graph, path, predicate, objects, user_entity):
    """ZAPOMENUTÍ: odstraní fakta uživatele z GRAFU i DENÍKU podle
    předlohy — predikát + objekty jako podmnožina, PŘESNOU shodou tvarů
    („Pavel" nesmaže „Pavlu"; složený pokyn „odstraň X, ponech Y" je tak
    bezpečný sám od sebe). Konzistence: co zmizí z grafu, zmizí i z deníku.

    Returns:
        list[str]: Popisy odstraněných faktů (prázdné = nic nesedělo).
    """
    target = set(objects)
    removed, kept = [], {}
    for key, fact in graph.facts.items():
        nodes = {part.node for part in fact.participants}
        if fact.predicate == predicate and target <= nodes \
                and user_entity in nodes:
            removed.append(f"{fact.predicate}: "
                           f"{', '.join(sorted(target))}")
            continue
        kept[key] = fact
    if removed:
        graph.replace_facts(kept)
    if path and os.path.exists(path):
        lines = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("predicate") == predicate \
                        and target <= set(row.get("objects", [])):
                    continue
                lines.append(line)
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    return removed


def forget_interval(graph, path, interval, user_entity):
    """Zapomenutí OBDOBÍ („zapomeň, co jsem dnes/včera řekl"): smaže
    fakta uživatele s časovou kotvou uvnitř intervalu — z grafu i deníku
    (kombinace forget × recall: Chronos vybírá, Mnemos maže).

    Returns:
        list[str]: Popisy odstraněných faktů.
    """
    from jellyai.graph.graph import parse_date
    removed, kept = [], {}
    for key, fact in graph.facts.items():
        nodes = {part.node for part in fact.participants}
        times = [part.node for part in fact.participants
                 if part.role == "time"]
        if user_entity in nodes and times \
                and any(interval.contains_date(parse_date(t))
                        for t in times):
            others = [part.node for part in fact.participants
                      if part.node != user_entity
                      and part.role != "time"]
            removed.append(f"{times[0]} — {fact.predicate}: "
                           f"{', '.join(others[:5])}")
            continue
        kept[key] = fact
    if removed:
        graph.replace_facts(kept)
    if path and os.path.exists(path):
        lines = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if interval.contains_date(parse_date(row.get("time", ""))):
                    continue
                lines.append(line)
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    return removed


def persist(statement, path):
    """Připíše konstatování do DENÍKU paměti (JSONL, append-only).

    Deník je zdroj pravdy paměti uživatele: přežije restart služby i
    přestavbu korpusového grafu (fakta uživatele v anotacích nejsou) a je
    auditovatelný — každý řádek = jedno zapamatované konstatování.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(statement, ensure_ascii=False) + "\n")


def replay(graph, path, user_entity):
    """Přehraje deník paměti do (čerstvě načteného) grafu.

    Args:
        graph (FactGraph): Graf, do kterého se vzpomínky obnoví.
        path (str): Cesta k deníku (memory.jsonl); chybějící = prázdná paměť.
        user_entity (str): Id uzlu identity uživatele.

    Returns:
        set[str]: Predikáty obnovených faktů (doplní slovník parseru).
    """
    predicates = set()
    if not path or not os.path.exists(path):
        return predicates
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            statement = json.loads(line)
            remember(graph, statement, user_entity)
            predicates.add(statement["predicate"])
    return predicates
