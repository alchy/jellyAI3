"""Hygiena grafu — KORPUSOVÁ EVIDENCE lemmat proti mis-tagům.

Jedna věta občas projde se špatnou značkou: „Vezmi hůl a hoď ji" udělá
z „hoď" (NOUN dle taggeru) účastníka „hodit"; „Izaiáš" označený jako VERB
se stane predikátem. CELÝ korpus ale ví lépe — hlasování upos přes všechny
výskyty lemmatu řekne, čím slovo převážně JE:

* účastník-koncept s převahou slovesných hlasů není entita → z faktu pryč;
* predikát s převahou jmenných hlasů (PROPN/NOUN bez sloves) není děj →
  celý fakt pryč;
* bez hlasů se NEsoudí (paměť Mnemos, řídká slova) a víceslovné uzly se
  nechávají (řeší je jiné mechanismy).

Fakt, kterému po čistce zbude jediný účastník, padá — bez protistrany nic
nenese. Prahy jsou konzervativní (převaha ≥ 80 %, aspoň 3 hlasy), takže
„koleno"/„rameno" (NOUN hlasy) zůstávají nedotčené.
"""

from collections import Counter, defaultdict

from jellyai.graph.graph import FactNode
from jellyai.lang import current

_MIN_VOTES = 3        # méně výskytů = nesoudit (řídké slovo)
_DOMINANCE = 0.8      # podíl hlasů, od kterého je verdikt „převážně"


def lemma_upos_votes(annotations):
    """Spočítá hlasy upos pro každé lemma přes celý korpus.

    Args:
        annotations (dict): Anotace vět (`{"sentences": [[token, …]]}`).

    Returns:
        dict[str, Counter]: lemma → Counter({upos: počet}).
    """
    votes = defaultdict(Counter)
    for annotation in annotations.values():
        for sent in annotation.get("sentences", []):
            for token in sent:
                lemma = token.get("lemma")
                upos = token.get("upos")
                if lemma and upos:
                    votes[lemma][upos] += 1
    return votes


def form_case_votes(annotations):
    """Hlasy o VELIKOSTI PÍSMEN tvarů: slovo → (malé, velké) počty.

    Jméno se v textu píše s velkým písmenem; slovo psané převážně malými
    („chléb") součástí jména není — NameTag ho občas splete s příjmením
    a vyrobí kontejner „Abraham chléb".

    Returns:
        dict[str, list[int, int]]: slovo (lower) → [malých, velkých].
    """
    votes = defaultdict(lambda: [0, 0])
    for annotation in annotations.values():
        for sent in annotation.get("sentences", []):
            for token in sent:
                form = token.get("form") or ""
                if not form or not form[:1].isalpha():
                    continue
                votes[form.lower()][0 if form[:1].islower() else 1] += 1
    return votes


def _lowercase_word(case_votes, word):
    """Slovo psané v korpusu PŘEVÁŽNĚ malými písmeny (≥3 hlasy, ≥80 %)."""
    lower, upper = case_votes.get(word.lower(), (0, 0))
    total = lower + upper
    return total >= _MIN_VOTES and lower / total >= _DOMINANCE


def name_case_votes(annotations):
    """Hlasy o PÁDU jmenných tvarů přes korpus: tvar → Counter({pád: počet}).

    Tvar českého jména nese pád morfologicky („Ježíš" je nominativ, „Martu"
    akuzativ) — hlasuje se jen z jistých výskytů (PROPN s jednoznačným
    Case). Homografy („Petra": ženský Nom × mužský Gen) mají hlasy
    rozštěpené, takže pod dominancí zůstanou nesouzené.
    """
    votes = defaultdict(Counter)
    for annotation in annotations.values():
        for sent in annotation.get("sentences", []):
            for token in sent:
                case = (token.get("feats") or {}).get("Case", "")
                if token.get("upos") == "PROPN" and case and "," not in case:
                    votes[(token.get("form") or "").lower()][case] += 1
    return votes


def _dominant_case(votes, form):
    """Dominantní pád tvaru z korpusu (≥3 hlasy, ≥80 %); jinak None."""
    counter = votes.get(form.lower())
    if not counter:
        return None
    total = sum(counter.values())
    if total < _MIN_VOTES:
        return None
    case, count = counter.most_common(1)[0]
    return case if count / total >= _DOMINANCE else None


def _span_cases(tokens, start, end, votes):
    """Pády slov jmenného rozpětí: lokální jistý pád tokenu (PROPN,
    jednoznačný Case), jinak dominantní pád TVARU z korpusu.

    Korpusový fallback kryje mis-tagy uvnitř věty — v „Ježíš Martu ...
    miloval" (Jan 11,5) je „Ježíš" otagován jako VERB bez pádu, ale 400
    nominativních výskytů jinde ví lépe. Bez jistoty (víceznačný pád,
    řídký tvar) se slovo nesoudí.
    """
    cases = []
    for token in tokens:
        t_start, t_end = token.get("start"), token.get("end")
        if t_start is None or t_end is None \
                or t_start < start or t_end > end:
            continue
        form = token.get("form") or ""
        if not form[:1].isalpha():
            continue
        case = (token.get("feats") or {}).get("Case", "")
        if token.get("upos") == "PROPN" and case and "," not in case:
            cases.append(case)
            continue
        fallback = _dominant_case(votes, form)
        if fallback is not None:
            cases.append(fallback)
    return cases


def scrub_entities(annotations, case_votes):
    """Vyřadí OSOBNÍ entity slepené omylem NER (in-place, před buildem).

    Dva nezávislé signály, oba DŘÍV, než z kontejneru kanonizace dokumentu
    udělá „nejdelší jméno" a přemapuje na něj celou osobu:

    * **velikost písmen** — entita (typ p/P vč. CNEC kontejnerů), jejíž
      KTERÉKOLI slovo je v korpusu převážně malé, není jméno („Abraham
      chléb": obecné slovo mis-tagnuté jako příjmení);
    * **pádová shoda** (jazykové pravidlo `name_case_agreement`) — české
      víceslovné jméno se skloňuje VE SHODĚ („Karla Čapka" Gen+Gen);
      pádově neshodné PROPN členy („Ježíš Martu" Nom+Acc, Jan 11,5) jsou
      dva větní účastníci, ne jedno jméno. Členy kontejneru (pf „Ježíš",
      ps „Martu") zůstávají — jsou to skutečné osoby té věty.

    Returns:
        int: Počet vyřazených entit.
    """
    agreement = current()["name_case_agreement"]
    grammar_votes = name_case_votes(annotations) if agreement else {}
    dropped = 0
    for annotation in annotations.values():
        entities = annotation.get("entities")
        if not entities:
            continue
        tokens = [token for sent in annotation.get("sentences", [])
                  for token in sent] if agreement else []
        kept = []
        for entity in entities:
            person = entity.get("type", "")[:1].lower() == "p"
            if person and any(_lowercase_word(case_votes, word)
                              for word in entity.get("text", "").split()):
                dropped += 1
                continue
            if person and agreement and " " in entity.get("text", "") \
                    and entity.get("start") is not None:
                cases = _span_cases(tokens, entity["start"], entity["end"],
                                    grammar_votes)
                if len(set(cases)) > 1:
                    dropped += 1
                    continue
            kept.append(entity)
        annotation["entities"] = kept
    return dropped


def _dominant(votes, lemma, kinds):
    """True, když má lemma dost hlasů a `kinds` v nich převažují."""
    counter = votes.get(lemma)
    if not counter:
        return False
    total = sum(counter.values())
    if total < _MIN_VOTES:
        return False
    return sum(counter[k] for k in kinds) / total >= _DOMINANCE


def scrub(graph, votes):
    """Vyčistí graf podle hlasů lemmat (in-place, přes `replace_facts`).

    Args:
        graph (FactGraph): Graf k čistce.
        votes (dict): Výstup `lemma_upos_votes`.

    Returns:
        tuple[int, int]: (vyřazených účastníků, vyřazených faktů).
    """
    # STRUKTURNÍ predikáty jsou jmenné ZÁMĚRNĚ: reifikované vztahy (bratr),
    # identita/zařazení (být/druh), asociace (kontext), dekompozice dat
    # (rok/měsíc/den) — hlasování se na ně nevztahuje
    lang = current()
    structural = (frozenset(lang["relational_nouns"])
                  | {"být", "druh", "kontext"}
                  | set(lang["date_part_forms"].values()))
    dropped_participants = 0
    dropped_facts = 0
    kept = {}
    for key, fact in graph.facts.items():
        # predikát s převahou jmenných hlasů není děj („Izaiáš")
        if fact.predicate not in structural \
                and _dominant(votes, fact.predicate, ("PROPN", "NOUN")):
            dropped_facts += 1
            continue
        participants = []
        for p in fact.participants:
            # entitní role (podmět/předmět/téma) s převahou slovesných či
            # adjektivních hlasů („hodit", „dovoleno") nejsou entity;
            # v rolích pred/attr adjektiva PATŘÍ (vlastnosti) — nesahat
            if p.type == "concept" and " " not in p.node \
                    and p.role in ("subj", "obj", "theme") \
                    and _dominant(votes, p.node, ("VERB", "AUX", "ADJ")):
                dropped_participants += 1
                continue
            participants.append(p)
        if len(participants) < 2:
            dropped_facts += 1                # bez protistrany fakt nenese nic
            continue
        if len(participants) == len(fact.participants):
            kept[key] = fact
        else:
            new_key = (fact.predicate, tuple(participants))
            existing = kept.get(new_key)
            if existing is None:          # ořez = nový klíč i id (drill čte id)
                kept[new_key] = FactNode(new_key, fact.predicate, fact.weight,
                                         tuple(participants), set(fact.source))
            else:                         # ořezy splynuly → agregace vah
                existing.weight += fact.weight
                existing.source |= fact.source
    graph.replace_facts(kept)
    return dropped_participants, dropped_facts
