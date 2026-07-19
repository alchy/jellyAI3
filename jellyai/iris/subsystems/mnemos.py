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

from jellyai.graph.canon import deaccent
from jellyai.graph.extract import make_fact, Participant
from jellyai.iris.subsystems.chronos import resolve_temporal
from jellyai.lang import current
from jellyai.lang.lexer import EMAIL_RE, classify

# Kmen jména pro shodu napříč pády ("Jindra"≈"Jindrovi" → "jindr"). POZOR:
# záměrně NE cluster_key — ten slučuje i různé lidi (Pavel≡Pavla → jeden kmen),
# což by u adresáta/forgetu smazalo/poslalo špatné osobě. Konzervativní ořez
# koncovek drží Pavel≠Pavla (Pavel končí na -l, neubírá se).
_NAME_SUFFIXES = ("ovi", "ovy", "ove", "ova", "em", "um", "y", "u", "a", "e", "i", "o")


def name_stem(name):
    """Deakcentovaný kmen jména bez pádové koncovky (shoda adresáta v pádech)."""
    stem = deaccent(str(name)).lower().strip()
    for suffix in _NAME_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 2:
            return stem[:-len(suffix)]
    return stem


def _date_label(interval):
    """Interval → povrch časového uzlu grafu („17. července 2026").

    Genitivy měsíců jsou jazyková data; formát odpovídá datům z korpusu,
    takže `parse_date` i dekompozice fungují nad pamětí stejně jako nad texty.
    """
    start = interval.start
    month = current()["temporal"]["month_genitives"][start.month - 1]
    return f"{start.day}. {month} {start.year}"


def _finite(tagged):
    """První kandidát prézentního slovesa (třída lexeru), nebo None."""
    return next((t for t in tagged if "sloveso_fin" in t.classes), None)


def utterance_features(tagged):
    """RYSY výroku pro kartové triggery — kód nerozhoduje, jen měří.

    Args:
        tagged (list[TaggedToken]): Výstup lexeru (`classify`).

    Returns:
        set[str]: Podmnožina {"first_person", "copula", "l_verb",
        "finite_verb", "email"}.
    """
    features = set()
    if any("prvni_osoba" in t.classes for t in tagged):
        features.add("first_person")
    if any("email" in t.classes for t in tagged):
        # Hodnota MIMO přirozený jazyk (e-mail): výrok je přiřazení atributu,
        # NE slovesné/sponové konstatování. Slovní rysy se proto POTLAČÍ, aby
        # spurious signály (slovo „email" končí na -l → vypadá jako l-příčestí)
        # nepřebily kartu atributu. Rys email pak zbývá jediná pasující cesta.
        features.add("email")
        return features
    if any("spona" in t.classes for t in tagged):
        features.add("copula")
    finite = _finite(tagged)
    if finite is not None:
        features.add("finite_verb")
    # l-příčestí: jméno „Marcela" po ořezu vypadá jako l-tvar — KAPITALIZOVANÝ
    # začátek věty se počítá, jen když věta nemá prézentní sloveso („Pršelo…")
    if any(t.l_stem and (finite is None or t.form != finite.form)
           and (t.form[:1].islower() or (i == 0 and finite is None))
           for i, t in enumerate(tagged)):
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
    tagged = classify(text, is_node=is_node)
    # otázka bez otazníku („Kdo je Roník.") NENÍ konstatování — tázací
    # slovo na začátku vetuje zápis, dotaz jde dotazovou cestou
    if tagged and "otaz" in tagged[0].classes:
        return None
    # VZOROVÉ karty výroků mají přednost (#46 fáze 3): sekvence tříd
    # rozhodne tam, kde ploché rysy nevidí („Roník JÍ granule" — sloveso
    # pod délkovým guardem pozná jen stavba věty); rysové karty zůstávají
    from jellyai.lang.matcher import expand_pattern, match_sequence
    aliases = lang.get("pattern_aliases", {})
    card, binding = None, None
    for pattern_card in deck.cards:
        if pattern_card.trigger.get("event") != "utterance.statement":
            continue
        sequence = pattern_card.trigger.get("pattern")
        if not sequence:
            continue
        found = match_sequence(expand_pattern(sequence, aliases), tagged)
        if found is not None:
            card, binding = pattern_card, found
            break
    # vzorová a rysová karta soutěží JEDNOU prioritou (deck ji už nese):
    # e-mailový atribut (12) přebije krátké sloveso „má" (8)
    feature_card = deck.best("utterance.statement",
                             {"features": utterance_features(tagged)})
    if card is None or (feature_card is not None
                        and feature_card.trigger.get("priority", 0)
                        > card.trigger.get("priority", 0)):
        card, binding = feature_card, None
    if card is None or "memorize" not in card.action:
        return None
    kind = card.action["memorize"]
    source = card.action.get("predicate_from")
    source_token = None      # povrchový tvar, ze kterého predikát vznikl
    if source == "pattern":
        # predikát ze SLOTU vzoru (katalog krátkých sloves na kartě)
        slot = binding.get(int(card.action["predicate_slot"][1:])) \
            if binding is not None else None
        predicate = slot.form if slot is not None else None
        source_token = predicate
    elif source == "l_verb":
        # l-ové příčestí je uvnitř věty MALÝMI písmeny; kapitalizovaný kandidát
        # („Emil", „Marcela" po ořezu) je zpravidla jméno — predikátem se stává,
        # jen když jiný l-tvar není („Pršelo v Praze")
        pick = next((t for t in tagged
                     if t.l_stem and t.form[:1].islower()),
                    next((t for t in tagged if t.l_stem), None))
        if pick is not None:
            source_token, predicate = pick.form, pick.l_stem
        else:
            predicate = None
    elif source == "finite_verb":
        finite = _finite(tagged)
        predicate = finite.form if finite is not None else None
        source_token = predicate
    else:
        predicate = card.action.get("predicate")
    if predicate is None:
        return None
    # KATALOG oprav zmršených predikátů (bezdiakritické „bydli"→ořez→„bydl"):
    # cílená ruční tabulka pro konkrétní tvary, ne univerzální algoritmus
    # (ten rozbíjel klasifikaci). Platí i pro dotaz — týž parser → shoda write/query.
    predicate = lang.get("predicate_catalog", {}).get(predicate, predicate)
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
    if kind == "attribute":
        # ATRIBUT s hodnotou mimo jazyk (e-mail): hodnota = adresa (regexp),
        # podmět = explicitní jméno ve výroku (nový person uzel — nemusí být
        # v korpusu), jinak identita uživatele. Predikát nese karta.
        value = next((t.form for t in tagged if "email" in t.classes), None)
        if value is None:
            return None
        stop = {"prvni_osoba", "spona", "funkcni", "potvrzeni",
                "privlastnovaci"}
        name = next((t.form for t in tagged
                     if t.form[:1].isupper() and "email" not in t.classes
                     and not (stop & t.classes) and len(t.form) > 1), None)
        return {"kind": "attribute", "predicate": predicate, "objects": [value],
                "subject": name or lang["user_entity"], "places": [],
                "time": time_label, "card": card.name, "needs_subject": False,
                "predicate_surface": source_token or predicate}
    exclude_l = card.action.get("exclude_l_forms", False)
    # účastníci = obsahová slova: bez 1. osoby, spony, funkčních slov,
    # potvrzení, ČASOVÝCH výrazů (interval už zlomil Chronos), ČÁSTIC
    # (už/však/občas — deník míval „neprší (Už)" jako zmršený zápis, #24)
    # a TÁZACÍCH slov (vztažné „co" ve vsuvce „To co X jí…" není účastník)
    skip = {"prvni_osoba", "spona", "funkcni", "potvrzeni", "cas", "castice",
            "otaz"}
    # KONCOVÁ spona neexistuje (spec §5, homograf „byt"≡„být" po deakcentaci):
    # spona stojí uprostřed věty — má-li výrok predikát ze VZORU („V Plzni
    # MÁ pronajatý byt."), je závěrečný spona-tvar podstatné jméno, účastník
    final_homograph = (tagged[-1] if tagged and source == "pattern"
                       and "spona" in tagged[-1].classes else None)
    objects = [t.form for t in tagged
               if (not (skip & t.classes) or t is final_homograph)
               # l-příčestí je uvnitř věty malými; KAPITALIZOVANÝ tvar, který
               # po ořezu vypadá jako l-tvar („Karla", „Emil"), je jméno a
               # z objektů vypadnout nesmí (výrok by se ztratil / přišel o podmět)
               and not (exclude_l and t.form[:1].islower() and t.l_stem)
               # sloveso není účastník — vylučuje se i POVRCHOVÝ zdroj
               # predikátu („Potkal"→potkal, „bydlel"→bydlet po katalogu)
               and t.form != predicate and t.form != source_token
               and len(t.form) > 1]
    if (not objects and not card.action.get("allow_no_objects")) \
            or (kind == "observation" and len(objects) < 2):
        return None
    # MÍSTA (brána E Toposu v dialogu): objekt za předložkou „v/ve/na"
    # je místo — dostane roli loc/geo („Marcela bydlí V PETROVICÍCH"),
    # aby „Kde bydlí…?" i kontejnment („v Čechách?") měly za co vzít
    places = []
    for i, tok in enumerate(tagged[:-1]):
        if tok.norm in ("v", "ve", "na") \
                and tagged[i + 1].form in objects:
            places.append(tagged[i + 1].form)
    return {"kind": kind, "predicate": predicate, "objects": objects,
            "places": places, "time": time_label, "card": card.name,
            "needs_subject": card.action.get("subject_from") == "context",
            "predicate_surface": source_token or predicate}


def parse_clauses(text, now, deck=None, is_node=None):
    """Souvětí → fakt na KLAUZULI (#46 fáze 4 v2, spec §3).

    „Jedna věta = jeden fakt" (docs/JAK-PSAT-FAKTA.md) platí i uvnitř
    souvětí: „Roník jí stravu, má však rád i maso." jsou DVA fakty.
    Rozpad na klauzule vyhrává jen když KAŽDÁ klauzule parsuje ČISTĚ —
    zdroj predikátu malými písmeny (kapitalizovaný l-lookalike „Pavla"
    ve výčtu „Potkal jsem Karla, Pavla a Marii." klauzuli netvoří);
    jinak platí parse celku. Když celek selže, uloží se čisté klauzule
    (zobecněná záchrana v1 — částečný zápis > ztráta). Klauzule bez
    kapitalizovaného účastníka dědí podmět předchozí klauzule.

    Returns:
        list[dict]: Parsy klauzulí (i jednoprvkový); [] = nic.
    """
    whole = parse_statement(text, now, deck, is_node)
    if "," not in text:
        return [whole] if whole is not None else []
    parses, subject, all_clean = [], None, True
    for raw in text.split(","):
        clause = raw.strip().rstrip(".")
        parsed = (parse_statement(clause + ".", now, deck, is_node)
                  if clause else None)
        if parsed is None \
                or parsed["predicate_surface"][:1].isupper():
            all_clean = False
            continue
        lead = next((o for o in parsed["objects"] if o[:1].isupper()), None)
        if lead is not None:
            subject = lead
        elif subject is not None and not parsed["needs_subject"]:
            parsed["objects"].insert(0, subject)    # dědění podmětu
        parses.append(parsed)
    if all_clean and len(parses) >= 2:
        return parses
    if whole is not None:
        return [whole]
    return parses


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
    elif statement["kind"] == "attribute":
        # ATRIBUT osoby (e-mail): hodnota dostane TYP `email` (regexp), aby
        # ji šlo z grafu dohledat mimo přirozený jazyk; uživatel je zdroj.
        participants = [Participant("subj", statement["subject"], "person")]
        participants += [Participant("obj", obj,
                                     "email" if EMAIL_RE.match(obj) else "concept")
                         for obj in objects]
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
    if statement["kind"] in ("attributed", "attribute"):
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


def forget_entity(graph, path, name, user_entity):
    """Zapomene CELOU entitu („zapomeň na Ronika"): odstraní z grafu VŠECHNY
    fakty, kde uzel vystupuje (shoda KMENE jména napříč pády — Ronika→Ronik),
    a odpovídající řádky deníku (entita jako subjekt nebo v objektech).

    Args:
        graph (FactGraph): Cílový graf.
        path (str | None): Deník paměti (None = jen graf, bez zápisu).
        name (str): Jméno entity (v libovolném pádu).
        user_entity (str): Id uzlu uživatele (nikdy se nemaže tímto povelem).

    Returns:
        list[str]: Popisy odstraněných faktů (prázdné = entita nenalezena).
    """
    target = name_stem(name)
    node_id = next((nid for nid in graph.nodes
                    if nid != user_entity and name_stem(nid) == target), None)
    if node_id is None:
        return []
    removed, kept = [], {}
    for key, fact in graph.facts.items():
        if any(part.node == node_id for part in fact.participants):
            others = [p.node for p in fact.participants
                      if p.node != node_id and p.role != "time"]
            removed.append(f"{fact.predicate}: {node_id}"
                           + (f" ({', '.join(others[:4])})" if others else ""))
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
                if name_stem(row.get("subject", "")) == target \
                        or any(name_stem(o) == target
                               for o in row.get("objects", [])):
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
