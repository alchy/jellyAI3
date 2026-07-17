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

from jellyai.answerer.selection import _clean_lemma
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


_MIN_CASE_VOTES = 2   # pád tvaru je morfologie (skoro deterministická) —
#                       stačí 2 jednomyslné hlasy („Masaryka" Gen×2 proti
#                       mis-tagům NOUN); homografy rozštěpí dominanci samy


def _dominant_case(votes, form):
    """Dominantní pád tvaru z korpusu (≥2 hlasy, ≥80 %); jinak None."""
    counter = votes.get(form.lower())
    if not counter:
        return None
    total = sum(counter.values())
    if total < _MIN_CASE_VOTES:
        return None
    case, count = counter.most_common(1)[0]
    return case if count / total >= _DOMINANCE else None


def _span_cases(tokens, start, end, votes):
    """Pády slov jmenného rozpětí: KORPUS PRVNÍ (dominantní pád tvaru),
    lokální jistý pád tokenu jen pro řídké tvary bez korpusového verdiktu.

    Lokální tag jediné věty je nejslabší evidence — slepenec „Izák Jákoba"
    prošel na lokálním mis-tagu Nom+Nom, ač „Jákoba" je korpusově Gen;
    „Ježíš Martu" měl „Ježíš" jako VERB bez pádu, ale 400 nominativů jinde
    ví lépe. Bez jistoty (víceznačný pád, řídký tvar) se slovo nesoudí.
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
        corpus = _dominant_case(votes, form)
        if corpus is not None:
            cases.append(corpus)
            continue
        case = (token.get("feats") or {}).get("Case", "")
        if token.get("upos") == "PROPN" and case and "," not in case:
            cases.append(case)
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


def noun_animacy_votes(annotations):
    """Hlasy o ŽIVOTNOSTI substantiv: slovo (tvar i lemma) → Counter.

    Česká maskulina nesou Animacy přímo ve feats; feminina osoby designují
    rodem — hlasuje se proto jen nad maskuliny a feminina se nesoudí.
    Klíčem je tvar I lemma (uzly grafu nesou obojí).
    """
    votes = defaultdict(Counter)
    for annotation in annotations.values():
        for sent in annotation.get("sentences", []):
            for token in sent:
                feats = token.get("feats") or {}
                if token.get("upos") != "NOUN" \
                        or feats.get("Gender") != "Masc" \
                        or not feats.get("Animacy"):
                    continue
                for key in {(token.get("form") or "").lower(),
                            (token.get("lemma") or "").lower()}:
                    if key:
                        votes[key][feats["Animacy"]] += 1
    return votes


def _inanimate(votes, node_id):
    """Uzel je v korpusu PŘEVÁŽNĚ neživotné substantivum (≥2 hlasy, ≥80 %)."""
    counter = votes.get(node_id.lower())
    if not counter:
        return False
    total = sum(counter.values())
    return total >= _MIN_CASE_VOTES \
        and counter.get("Inan", 0) / total >= _DOMINANCE


def scrub_semantics(graph, animacy_votes):
    """Sémantické guardy faktů podle korpusové morfologie (in-place).

    * **druh-fakt s osobou pod neživotným druhem** — osoba nemůže být
      instancí neživotné věci: „druh(Dorothea, vztah)" (z „Ze vztahu
      Dorothey…" — nesklonné jméno bez pádu proklouzlo pádovým guardům)
      či „druh(Abraham, chléb)". Subjekt, který sám hlasuje neživotně
      („Týdenník" mylně typovaný person), fakt drží — obě strany neživotné
      jsou konzistentní zařazení.
    * **reifikovaný vztah bez protistrany** — vztahové jméno jako predikát
      („žena") vyžaduje obj účastníka; fakt jen s časy je troska parseru.

    Returns:
        int: Počet vyřazených faktů.
    """
    relational = frozenset(current()["relational_nouns"])
    kept, dropped = {}, 0
    for key, fact in graph.facts.items():
        roles = {p.role for p in fact.participants}
        if fact.predicate in relational and "obj" not in roles:
            dropped += 1
            continue
        if fact.predicate == "druh":
            subj = next((p for p in fact.participants
                         if p.role == "subj"), None)
            pred = next((p for p in fact.participants
                         if p.role == "pred"), None)
            if subj is not None and pred is not None \
                    and subj.type == "person" \
                    and _inanimate(animacy_votes, pred.node) \
                    and not _inanimate(animacy_votes, subj.node):
                dropped += 1
                continue
        kept[key] = fact
    graph.replace_facts(kept)
    return dropped


def propn_lemma_votes(annotations):
    """Hlasy o LEMMATU jmenných tvarů: tvar (lower) → Counter({lemma}).

    Lemma PROPN tokenu JE nominativ jména („Betlémě"→„Betlém", „Boha"→
    „Bůh") — korpus tak nese nominativizaci zdarma, bez morfo služby.
    Hlasuje se jen při shodě kapitalizace tvaru a lemmatu (lemma malými
    by z vlastního jména udělalo obecné slovo); mis-lemmata rozštěpí
    dominanci a tvar zůstane nesouzený.
    """
    votes = defaultdict(Counter)
    for annotation in annotations.values():
        for sent in annotation.get("sentences", []):
            for i, token in enumerate(sent):
                upos = token.get("upos")
                form = token.get("form") or ""
                if upos == "NOUN":
                    # jméno v lexikonu taggeru („Boha" → NOUN, lemma bůh)
                    # hlasuje jen KAPITALIZOVANÉ uvnitř věty — začátek věty
                    # kapitalizuje i obecná slova
                    if i == 0 or not form[:1].isupper():
                        continue
                elif upos != "PROPN":
                    continue
                lemma = _clean_lemma(token.get("lemma") or "")
                if not form or not lemma:
                    continue
                if form[:1].isupper() and lemma[:1].islower():
                    # lemma jména malými („Boha"→„bůh") — jméno zůstává
                    # jménem, nominativ se kapitalizuje („Bůh")
                    lemma = lemma[0].upper() + lemma[1:]
                elif form[:1].islower() and lemma[:1].isupper():
                    continue          # malý tvar s velkým lemmatem — nesoudit
                votes[form.lower()][lemma] += 1
    return votes


def _dominant_lemma(votes, word):
    """Dominantní lemma tvaru z korpusu (≥2 hlasy, ≥80 %); jinak None."""
    counter = votes.get(word.lower())
    if not counter:
        return None
    total = sum(counter.values())
    if total < _MIN_CASE_VOTES:
        return None
    lemma, count = counter.most_common(1)[0]
    if len(lemma) < 3:
        return None       # zmrzačené lemma cizího jména („Lea" → „Le")
    return lemma if count / total >= _DOMINANCE else None


_NOMINATIVE_TYPES = ("person", "geo", "dílo", "institution")


def nominativize(graph, lemma_votes):
    """Id pojmenovaných uzlů NOMINATIVEM (in-place; backlog #1v2).

    Skloněný povrch v id fragmentuje fakty („Betlémě" 6 vedle „Betlém" 2;
    „Boha" 429 vedle „Bůh" 1175) a prosakuje do odpovědí („Kdo je Ježíš?"
    → „Boha"). Každé slovo id se nahradí dominantním lemmatem z korpusu;
    slovo bez verdiktu zůstává (cizí nesklonná jména). Kolize s existujícím
    nominativním uzlem = žádoucí SLOUČENÍ (součet vah, aliasy); typy se
    nemění. Časové uzly se nedotýkají (genitivní formát dat čte parse_date).

    Returns:
        int: Počet přemapovaných uzlů.
    """
    node_map = {}
    for node in graph.nodes.values():
        if node.type not in _NOMINATIVE_TYPES:
            continue
        words = node.id.split()
        renamed = [(_dominant_lemma(lemma_votes, w) or w) for w in words]
        new_id = " ".join(renamed)
        if new_id and new_id != node.id:
            node_map[node.id] = new_id
    if node_map:
        from jellyai.graph.graph import remap_nodes
        remap_nodes(graph, node_map, force_person=False)
    return len(node_map)


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
