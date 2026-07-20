"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozeberou ŠABLONY (`build_query`: vzorové karty + pseudo-QL — bez
UDPipe, řez #14), najde se uzel tématu, z něj fakty (dle role a predikátu)
a z faktu s **nejvyšší vahou** vezme účastník cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, odpověď je
poctivé „nenašel" — nehádat je zákon.
"""

from dataclasses import dataclass, field as dc_field
from datetime import datetime

from jellyai.answerer.base import Answer, Answerer
from jellyai.graph.graph import parse_date
from jellyai.iris.subsystems.chronos import resolve_temporal
from jellyai.iris.subsystems.topos import (_key as _topos_key,
                                           area_keys, load_gazetteer,
                                           place_within)
from jellyai.lang import current
from jellyai.answerer.pattern import SubQuery, Pattern
from jellyai.answerer.query import build_query, Query
from jellyai.answerer.template import _to_nominative
from jellyai.graph.activation import ActivationField
from jellyai.graph.graph import instance_lit
from jellyai.graph.canon import _stem, name_gender, deaccent

_DATE_PARTS = {"rok", "měsíc", "den"}   # drill: „v kterém roce/měsíci…"
_MAX_ENUM = 5                           # strop výčtové odpovědi (čitelnost)


def _loose(word):
    """Nejvolnější porovnávací klíč: kmen → bez diakritiky → bez koncové
    samohlásky (floor 2). „hru"≡„hra"≡„hr", „jezis"≡„Ježíš", „Čapka"≡„Čapek"
    (epenteze kmene je pro jména nutná). Daň: krátká slova se občas potkají
    se šumem („měl"→„ml" ≡ vadný uzel „mle") — to ale řeší upřímný dialog
    (nízká assurance), ne ticho; šum patří vyčistit v datech. Nejslabší
    patro — nikdy nepřebije přesnější shodu."""
    s = deaccent(_stem(word))
    return s[:-1] if len(s) > 2 and s[-1] in "aeiouy" else s


def _movement_ring(predicate):
    """SMĚROVÁ třída přesunu (putovat≈přijít≈odejít) — kruh JEN pro
    místní díry a afinitu: u osobních děr obrací děj („Kdo přišel
    k X?" nesmí odpovědět fakt odejít — nález „Pan")."""
    movement = current().get("movement_predicates", ())
    low = predicate.lower() if predicate else ""
    if predicate in movement or low in movement:
        # kapitalizované mis-lemma („Putovat") — týž fallback jako
        # _synonym_ring
        return [(p, False) for p in movement if p not in (predicate, low)]
    return []


def _synonym_ring(predicate):
    """(predikát, exact) + jeho synonyma z jazykových dat — „Kde žili?" najde
    bydlet-fakt; přesný predikát drží přednost bonusem. Kapitalizovaný
    predikát grafu („Putovat" — mis-lemma začátku věty) sáhne po
    synonymech přes lowercase, jinak by remíza kapitalizace o ring
    připravila (nedeterminismus pořadí setu)."""
    synonyms = current()["predicate_synonyms"]
    group = synonyms.get(predicate) or synonyms.get(predicate.lower(), ())
    ring = [(predicate, True)]
    for lemma in group:
        if lemma != predicate:
            ring.append((lemma, False))
    return ring


def _evidence_date(fact):
    """Řadicí klíč datované evidence (rok, měsíc, den); None = nedatováno.

    Denní přesnost stačí — jemnější remízy rozhoduje pořadí zápisu
    (deník paměti je chronologický)."""
    months = current()["temporal"]["months"]
    for part in fact.participants:
        if part.role != "time":
            continue
        parts = parse_date(part.node)
        if "rok" not in parts:
            continue
        month = months.get(deaccent(parts.get("měsíc", "")), 0)
        return (int(parts["rok"]), month, int(parts.get("den", 0)))
    return None


def _event_text(fact, exclude=()):
    """Děj jako odpověď: SLOVESO první, pak účastníci. „Co se stalo?" se ptá
    na děj — faktový uzel je děj reifikovaný, holý podmět odpovědí není."""
    others = [p.node for p in fact.participants if p.node not in exclude]
    return fact.predicate + (": " + ", ".join(others[:5]) if others else "")


@dataclass
class TurnResult:
    """Výsledek JEDNOHO tahu answereru (postřeh 2.1).

    Dřív devět side-channel atributů s křehkým protokolem nulování
    („nesmí přežít z minula" na třech místech) — teď jeden objekt,
    který `begin_turn()` vymění celý. Automat i harness čtou
    `answerer.turn.*`; konverzační stav (těžiště, `_prev_trace`,
    `_last_values`) zůstává na answereru — přežívá tahy záměrně.
    """
    trace: dict = None        # trasa odpovědi (téma → fakt → hodnota)
    pattern: object = None    # vykonaný pseudo-QL Pattern (API/viz)
    query_card: str = None    # vzorová karta dotazu (telemetrie #38)
    resolution: dict = None   # evidence rozlišení (vstup QueryAssurance)
    overflow: list = dc_field(default_factory=list)   # oblasti přetečení
    empty_role: tuple = None  # verdikt prázdné díry (#57 E3) — jistota
    empty_topic: tuple = None  # (téma, kandidáti) — nabídka s volbou (B4)
    theme_bound: set = dc_field(default_factory=set)  # adresáti (#55)


class GraphAnswerer(Answerer):
    """Odpovídá z globálního faktového grafu; jinak fallback.

    Drží **konverzační těžiště** (`ActivationField` nad id uzlů): každý tah rozsvítí
    téma a odpověď, každý tah pohasíná. Když navazující otázka nemá vlastní téma
    („Kdy se narodila?"), vezme se nejteplejší uzel z rozhovoru — dialog tak plyne.
    """

    def __init__(self, graph, client, fallback, *, context_decay=0.55,
                 spread_depth=2, spread_falloff=0.35, clock=None,
                 context_hub_limit=50):
        """Vytvoří answerer.

        Args:
            graph (FactGraph): Postavený faktový graf.
            client: ÚFAL klient (jen nominativizace odpovědí — rozbor
                otázky jde šablonami, řez #14).
            fallback (Answerer): Answerer pro neúspěch (extraktivní/template).
            context_decay (float): Pohasínání konverzačního těžiště na dotaz
                (viz `ActivationField.decay`; nižší = kratší paměť kontextu).
        """
        self.graph = graph
        self.client = client
        self.fallback = fallback
        self.context_decay = context_decay
        self.spread_depth = spread_depth
        self.spread_falloff = spread_falloff
        self.turn = TurnResult()   # výsledek AKTUÁLNÍHO tahu (postřeh 2.1)
        self.pick_focus = None   # zvolená oblast overflow dialogu (#5) — 1 tah
        self.context_hub_limit = context_hub_limit   # hub asociací (B4)
        self._last_values = []   # hodnoty minulé odpovědi („Kdo další…?", #53a)
        self._prev_trace = None  # trasa PŘEDCHOZÍHO tahu (drill „Kdy?")
        self.visited = []        # uzly protnuté (rekurzivním) matchem → rozsvícení
        self.context = ActivationField(decay=context_decay)   # těžiště (id uzlu → jas)
        self.source_context = ActivationField(decay=context_decay)  # attention nad ZDROJI
        self.domain_docs = frozenset()   # explicitní doména (focus-shift „v kontextu X")
        self._predicates = {f.predicate for f in graph.facts.values()}  # slovník QL
        self.history = []        # trajektorie konverzace (tahy s trasou a těžištěm)
        self._word_set = None    # doslovná slova uzlů (veto neznámého slovesa)
        self._resolved_knowns = set()   # entity otázky rozřešené tímto tahem
        self.clock = clock or datetime.now   # „teď" pro časová primitiva otázky
        self.time_filter = None  # tvrdý interval otázky (Chronos, brána Q→A)
        self.place_filter = None  # oblast otázky (Topos kontejnment)
        self._gazetteer_path = "data/sub_topos_gazetteer.jsonl"
        self._gazetteer = load_gazetteer(self._gazetteer_path)
        self._area_keys = area_keys(self._gazetteer)

    def _node_word(self, token):
        """DOSLOVNÉ slovo některého ENTITNÍHO uzlu grafu — bez kmenových
        pater (volná shoda by vetovala i slovesa: „Měl"≈šumový uzel „mle")
        a bez výrokových uzlů (obsah řeči jsou celé věty, ne entity). Cache
        se přestaví, když paměť Mnemos přidá uzly (fingerprint = počet)."""
        if self._word_set is None or self._word_set[0] != len(self.graph.nodes):
            self._word_set = (len(self.graph.nodes),
                              {word for node in self.graph.nodes.values()
                               if node.type != "výrok"
                               for word in node.id.lower().split()})
        return token.lower() in self._word_set[1]

    def reset(self):
        """Začne nový rozhovor — vymaže těžiště, provenienci i historii."""
        self.context = ActivationField(decay=self.context_decay)
        self.source_context = ActivationField(decay=self.context_decay)
        self.history = []
        self.turn = TurnResult()
        self.pick_focus = None
        self._last_values = []
        self.domain_docs = frozenset()

    def begin_turn(self):
        """Nový tah: trasa minulého tahu → `_prev_trace` (drill „Kde?"),
        výsledek tahu se vymění CELÝ — nulování je jeden řádek, žádný
        atribut „nesmí přežít z minula" (třída chyb postřehu 2.1)."""
        self._prev_trace = self.turn.trace or self._prev_trace
        self.turn = TurnResult()

    def _span_is_node(self, span):
        """Přísný test rozpětí (spec 4.3): rozřeší se na uzel A jeho obsahová
        slova jsou podmnožinou slov uzlu (kmenově/bezdiakriticky) — slepenec
        dvou entit ani cizí titul neprojde."""
        terms = [t for t in span.split()
                 if len(t) > 1
                 and deaccent(t.lower()) not in current()["query_skip_words"]]
        if not terms:
            return False
        node_id = self._resolve_topic(terms, warm=False)
        if node_id is None:
            return False
        node_keys = {_loose(w) for w in node_id.split()}
        return all(_loose(t) in node_keys for t in terms)

    def _resolve_topic(self, topic_terms, predicate=None, warm=True):
        """Najde uzel tématu otázky — nejlepší shodu s obsahovými lemmaty.

        **Přesná shoda velikosti má přednost**, case-insensitive je fallback:
        UDPipe někdy velikost zachová (PROPN „Babička" → lemma „Babička" — odliší
        knihu od obecné „babička"), jindy lemmatizuje na malá („Vějíř"→„vějíř" —
        pak by kapitalizovaný uzel byl jinak jménem nedostupný). Třetí patro je
        **kmenová shoda** (`canon._stem` — týž mechanismus jako build-side
        `resolve_entities`): skloněný termín („Galéna") tak dosáhne na kanonický
        uzel („Galén"), aniž by kdy přebil přesnější shodu. Dál preferuje uzel
        pokrývající **víc témat** (aby „Božena Němcová" přebila „Němcová"), delší
        (víceslovnou) entitu a nakonec vyšší frekvenci.

        Args:
            topic_terms (list[str]): Obsahová lemmata otázky.

        Returns:
            str | None: Id uzlu tématu, nebo None když nic nesedí.
        """
        lang = current()
        terms = [t for t in topic_terms
                 if t and len(t) > 1
                 and deaccent(t.lower()) not in lang["query_skip_words"]]
        low_terms = [t.lower() for t in terms]
        stems = [_stem(t) for t in terms]
        da_stems = [deaccent(s) for s in stems]   # bezdiakritický alias dotazu
        loose = [_loose(t) for t in terms]
        best_id, best_score = None, None
        candidates = []   # (id, stem_hits, váha, loose klíč, afinita) — vějíř
        ring = (_synonym_ring(predicate)
                + _movement_ring(predicate)) if predicate else ()
        for node in self.graph.nodes.values():
            if node.type == "výrok":
                continue                  # obsah řeči je hodnota, ne téma
            low_id = node.id.lower()
            low_words = low_id.split()
            node_stems = {_stem(w) for w in low_words}
            da_node = {deaccent(s) for s in node_stems}
            loose_node = frozenset(_loose(w) for w in low_words)
            # ALIASY (sloučené tvary z kanonizace) platí jako povrch uzlu:
            # „Ježíše" je táž osoba jako uzel „Ježíš"
            alias_forms = set(self.graph.aliases.get(node.id, ()))
            alias_low = {a.lower() for a in alias_forms}
            per_term = [(t == low_id or t in low_words or t in alias_low,
                         s in node_stems, d in da_node, l in loose_node)
                        for t, s, d, l in zip(low_terms, stems, da_stems, loose)]
            # POKRYTÍ TERMŮ je primární patro: uzel trefený víc slovy dotazu
            # (jakýmkoli patrem) přebije jediný povrchový hit („Karla Čapka"
            # → „Karel Antonín Čapek", ne zbytkový uzel „Antonína Čapka")
            coverage = sum(1 for hit in per_term if any(hit))
            if coverage == 0:
                continue
            ins_hits = sum(1 for hit in per_term if hit[0])
            stem_hits = sum(1 for hit in per_term if hit[1])
            da_hits = sum(1 for hit in per_term if hit[2])
            loose_hits = sum(1 for hit in per_term if hit[3])
            # přesná shoda velikosti má smysl jen u termů NESOUCÍCH velké
            # písmeno (lowercase lemma „vějíř" nerozliší pojem od titulu)
            exact_hits = sum(1 for t in terms
                             if any(ch.isupper() for ch in t)
                             and (t == node.id or t in node.id.split()
                                  or t in alias_forms))
            # PREDIKÁTOVÁ AFINITA: mezi rovnocennými jmennými shodami vyhrává
            # uzel, o němž se predikát otázky dá vypovědět („Vějíř" s napsat
            # faktem) — ale jmenné shody nikdy nepřebije (Ludvík Němec ≠
            # „Němec" s být-faktem jiné osoby)
            affinity = int(any(self.graph.facts_of(node.id, predicate=pred)
                               for pred, _ in ring))
            # bezdiakritická a volná shoda jsou nejslabší patra (pod kmenovou),
            # aby „cestina"/„hru" dosáhly na uzel a nikdy nepřebily přesnější;
            # POKRYTÍ stojí i nad exact — skloněný povrchový uzel („Čapka")
            # nesmí jedinou přesnou shodou přebít plnější jméno
            # PLNÉ POKRYTÍ uzlu (žádné nezakryté slovo) až ZA afinitou:
            # exact „Ježíš" přebije slepenec „Ježíš Martu", ale fragment
            # nikdy nepřebije plné jméno dřív, než promluví jmenná patra
            full = int(coverage == len(low_words))
            score = (coverage, exact_hits, ins_hits, stem_hits, da_hits,
                     loose_hits, affinity, full, len(low_words), node.weight)
            candidates.append((node.id, stem_hits, node.weight,
                               loose_node, affinity, score[:6]))
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
                best_terms, best_exact = per_term, exact_hits
        if best_id is not None and ring:
            # CLUSTER-AFINITA: varianty TÉHOŽ jména (stejný volný klíč) arbitruje
            # predikát — zbytkový skloněný uzel („Babičku", exact hit) nepřebije
            # variantu s faktem otázky („Babička" s napsat); cizí jména nemění
            best = next(c for c in candidates if c[0] == best_id)
            if not best[4]:
                same = [c for c in candidates
                        if c[3] == best[3] and c[4] and c[0] != best_id]
                if same:
                    # mezi afinitními variantami rozhodne shoda velikosti
                    # písmen s termem („Babičku"→„Babička", ne „babička"),
                    # pak váha
                    term_upper = any(t[:1].isupper() for t in terms)
                    best_id = max(same, key=lambda c: (
                        c[0][:1].isupper() == term_upper, c[2]))[0]
        domain_lit = None
        if best_id is not None and self.domain_docs:
            # DOMÉNOVÉ PATRO: explicitní doména (focus-shift „v kontextu
            # Bible" → ostrá množina dokumentů) arbitruje rovnocenné jmenné
            # kandidáty PROVENIENCÍ: vyhrává kandidát s fakty z domény,
            # soupeři se zúží na doménové. (Ostrá příslušnost, ne glow —
            # vyzařování po doc_links by doménu rozmazalo.)
            best_cand = next(c for c in candidates if c[0] == best_id)
            group = [c for c in candidates if c[5] == best_cand[5]]
            if len(group) > 1:
                def _in_domain(node_id):
                    return any(set(getattr(f, "source", ()) or ())
                               & self.domain_docs
                               for f in self.graph.facts_of(node_id))
                lit = [c for c in group if _in_domain(c[0])]
                if lit:
                    domain_lit = {c[0] for c in lit}
                    if best_id not in domain_lit:
                        best_id = max(lit, key=lambda c: c[2])[0]
        if best_id is not None and warm:
            # EVIDENCE PRO QueryAssurance: kvalita = průměrná váha nejlepšího
            # patra na term (exact ⊂ ins → bonus +0.2/exact term); soupeři =
            # kandidáti s TOUTÉŽ jmennou evidencí z JINÉHO kmenového clusteru
            # („Kdo je Čapek?" → Karel i Josef). Sondy is_node nezapisují.
            weights = (0.8, 0.6, 0.4, 0.25)          # ins/stem/da/loose
            base = sum(max((w for w, hit in zip(weights, t) if hit),
                           default=0.0) for t in best_terms)
            quality = min(1.0, (base + 0.2 * best_exact) / max(1, len(terms)))
            best_cand = next(c for c in candidates if c[0] == best_id)
            # AFINITA FILTRUJE SOUPEŘE: má-li vítěz fakt predikátu otázky
            # a soupeř ne, soupeř není skutečná alternativa („Kde se narodil
            # Ježíš?" — narodit-fakt má jen jeden) → volba není hádání.
            # U identity (fakt mají oba Čapkové) soupeři zůstávají → dialog.
            rivals = [c[0] for c in candidates
                      if c[0] != best_id and c[5] == best_cand[5]
                      and c[3] != best_cand[3]
                      and (domain_lit is None or c[0] in domain_lit)
                      and (not best_cand[4] or c[4])]
            self.turn.resolution = {"term": " ".join(terms), "winner": best_id,
                                    "quality": quality, "rivals": rivals}
            # nejednoznačné jméno rozsvítí VŠECHNY kandidáty se stejnou
            # kmenovou shodou (homonymní vějíř „Marie" → i biblická Maria) —
            # attention nese nejistotu rozřešení, vítěz svítí nejvíc; při
            # sondě is_node (zamítaná rozpětí) se vějíř nešíří (warm=False).
            # AKTIVNÍ DOMÉNA vějíř drží ve svých dokumentech — holá křestní
            # jména jinak lepí světy (bible ↔ wiki) a rozsvěcují cizí satelity
            top_stem = best_score[3]
            fan = sorted((c for c in candidates
                          if c[0] != best_id and c[1] == top_stem and c[1] > 0),
                         key=lambda c: -c[2])[:5]
            for node_id, _, _, _, _, _ in fan:
                if self.domain_docs and not any(
                        set(getattr(f, "source", ()) or ()) & self.domain_docs
                        for f in self.graph.facts_of(node_id)):
                    continue
                self.context.warm(node_id, 0.3)
        return best_id

    def _context_candidates(self):
        """Uzly kontextu seřazené podle jasu (nejteplejší první)."""
        return sorted(self.context.scores, key=self.context.scores.get, reverse=True)

    def _gender_ok(self, qa, node_id):
        """Shoda rodu otázky s (heuristickým) rodem osoby. U ne-osob vždy True."""
        node = self.graph.nodes.get(node_id)
        if qa.gender is None or node is None or node.type != "person":
            return True
        return name_gender(node_id) == qa.gender

    def _pattern_answer(self, question, pat, qa):  # pylint: disable=too-many-return-statements,too-many-branches
        """Univerzální match: neúplný fakt (pattern) → najdi shodný v grafu → díra.

        Nahrazuje ruční qtype/relační pravidla i attention: predikát + známé role
        musí sedět, vrátí se účastník v roli díry (ranking váhou + aktivací). Bez
        explicitní entity (**navazující dotaz**) se téma vezme z **kontextu** —
        projdou se kandidáti od nejteplejšího (gender-filtr) a vezme první, co odpoví.

        Args:
            question (str): Původní dotaz (jen pro datum v generické větvi).
            pat (Pattern): Neúplný fakt z otázky (šablony nebo UDPipe).
            qa (Query | QuestionAnalysis): Rozbor (rod, qtype, témata).

        Returns:
            tuple: (téma | None, hodnota | None, fakt | None).
        """
        self.visited = []                    # uzly protnuté (i)rekurzí → rozsvítit
        self.turn.theme_bound = set()            # explicitní adresáti dotazu (#55)
        if pat.hole_role == "relation":
            return self._relation_answer(pat)
        if pat.known:
            known_set, first_res = set(), None
            for role, known in pat.known:
                node = self._solve(known, pat.predicate)   # rekurzivně (i vnořené)
                if node is None:
                    return None, [], None    # pojmenované, ale neznámé → nehádat
                known_set.add(node)
                if role == "theme":
                    # ADRESÁT z dativu („Co řekl Ježíšovi?") — match ho
                    # vyžaduje v roli theme faktu, ne jako mluvčího (#55)
                    self.turn.theme_bound.add(node)
                if first_res is None:
                    first_res = self.turn.resolution
            if first_res is not None:
                # evidence tahu = rozlišení PODMĚTU (první known) — pozdější
                # předměty („rád") nesmí přepsat, o kom otázka je
                self.turn.resolution = first_res
            self._resolved_knowns = set(known_set)
            filled = self._fill_subject(pat, qa)
            if filled is not None:
                # QUERY-SIDE PRO-DROP: elidovaný podmět otázky („Jakou hru
                # napsal?") zdědí rodově shodnou osobu z těžiště — týž princip
                # jako pro-drop v extrakci; při neúspěchu se zkusí bez něj
                topic, values, fact = self._answer_from(pat, known_set | {filled})
                if values:
                    return topic, values, fact
            if pat.predicate in current()["generic_event_verbs"] \
                    and "rok" not in parse_date(question):
                # „Co se stalo s X?" = děje tématu (hloubka 1); „Co víme o X?"
                # (vědět) = okolní agregace do hloubky 2 (pseudo-n-gramy).
                # Otázky s datem nechává reverznímu lookupu (přesné párování).
                depth = 2 if pat.predicate == "vědět" else 1
                topic, values, fact = self._event_answer(known_set, depth)
                if values:
                    return topic, values, fact
            if qa.qtype is None and pat.hole_role is None:
                # zjišťovací otázka („Napsal X Y?") — existence, ne díra
                return self._existence(pat.predicate, known_set)
            return self._answer_from(pat, known_set)
        if qa.qtype is None and pat.hole_role is None \
                and pat.predicate is not None:
            # zjišťovací otázka BEZ účastníků („Pršelo dnes?") — existence
            # samotného predikátu; časový interval už rozsvítil Chronos
            return self._existence(pat.predicate, set())
        if pat.predicate is None and pat.hole_role and self._prev_trace:
            # holé tázací navázání („Kdy?", „Kde?") = drill do POSLEDNÍHO
            # faktu — jeho účastník v roli/typu díry
            fact = self.graph.facts.get(self._prev_trace["fact"])
            if fact is not None:
                values = [p.node for p in fact.participants
                          if p.role == pat.hole_role or p.type == pat.hole_type
                          or (pat.hole_type == "time" and p.role in ("time", "num"))]
                if values:
                    self.visited.extend(p.node for p in fact.participants)
                    return self._prev_trace["topic"], values[:_MAX_ENUM], fact
        if pat.predicate is None or qa.topic_terms:
            # bez predikátu nelze; pojmenoval-li něco, co pattern nezachytil (RUR),
            # NEhádat z kontextu — kontext je jen pro skutečně navazující dotaz
            return None, [], None
        for candidate in self._context_candidates():   # navazující dotaz → z těžiště
            if not self._gender_ok(qa, candidate):
                continue                     # „narodil?" ≠ ženská entita (a naopak)
            topic, values, fact = self._answer_from(pat, {candidate})
            if values:
                return topic, values, fact
        return None, [], None

    def run_pattern(self, pat):
        """Vykoná pseudo-QL `Pattern` přímo (API `/graphql` — jazyk je
        testovatelný bez parseru). Sémantika = jádro `_pattern_answer`:
        rozřeš known → díra/existence; bez kontextových pater.

        Returns:
            tuple: (téma | None, list hodnot, fakt | None).
        """
        self.visited = []
        known_set = set()
        for _, known in pat.known:
            node = self._solve(known, pat.predicate)
            if node is None:
                return None, [], None
            known_set.add(node)
        if not known_set:
            return None, [], None
        if pat.hole_role is None and pat.date_part is None:
            return self._existence(pat.predicate, known_set)
        return self._answer_from(pat, known_set)

    def _identity_vote(self, known_set, hole_role, hole_type):
        """HLASOVÁNÍ PŘES PATRA (#25): kandidát identity skóruje počtem
        NEZÁVISLÝCH pater evidence (spona být / druh / jmenovat).

        Bible je plná řečových spon („Ty jsi Bůh/Šimon/David…" v přímé
        řeči) — jeden sponový fakt identitu neurčuje. Kandidát doložený
        VÍCE patry (druh(Mesiáš) + jmenovat(Mesiáš)) přebije jednopatrový
        šum; bez vícepatrového vítěze platí dnešní pořadí (spona první,
        pak slabší patra) — zákon 5: korpusová evidence > jeden fakt.

        Returns:
            tuple: (list hodnot, fakt | None) — jako `_match`.
        """
        layers = ("být", "druh", current()["name_predicate"])
        votes, first, facts = {}, {}, {}
        for layer in layers:
            values, fact = self._match(layer, known_set, hole_role, hole_type)
            for order, value in enumerate(values):
                votes.setdefault(value, set()).add(layer)
                first.setdefault(value, (layers.index(layer), order))
                facts.setdefault(value, fact)
        multi = [v for v, layer_set in votes.items() if len(layer_set) >= 2]
        if multi:
            multi.sort(key=lambda v: first[v])
            return multi, facts[multi[0]]
        being = [v for v in votes if "být" in votes[v]]
        being.sort(key=lambda v: first[v])
        return being, facts[being[0]] if being else None

    def _answer_from(self, pat, known_set):
        """Z množiny známých uzlů dořeší díru patternu (vč. 2-skokového drillu).

        Returns:
            tuple: (téma | None, list hodnot, fakt | None).
        """
        node0 = next(iter(known_set))
        if pat.date_part:
            # 2-SKOK (rekurze): událost → datum (uzel) → jeho pod-fakt rok/měsíc/den
            date_nodes, _ = self._match(pat.predicate, known_set, "time", "time")
            if not date_nodes:
                return None, [], None
            values, fact = self._match(pat.date_part, {date_nodes[0]}, "val", "number")
            return (date_nodes[0], values, fact) if values else (None, [], None)
        if pat.predicate is None:
            return None, [], None            # bez predikátu se nehádá (žádný wildcard)
        if pat.predicate == "být" and pat.hole_role in ("pred", "attr"):
            # HLASOVÁNÍ PŘES PATRA (#25): identita se neřeší jedním faktem
            values, fact = self._identity_vote(known_set, pat.hole_role,
                                               pat.hole_type)
        else:
            values, fact = self._match(pat.predicate, known_set,
                                       pat.hole_role, pat.hole_type)
        if not values and pat.hole_role in ("pred", "attr"):
            # druhové zařazení (apozice) je slabší evidence než spona „být" —
            # čte se, až když spona mlčí („Co je R.U.R.?" → druh drama);
            # a jméno je také identita („Ježíš řečený Kristus" — jmenovat)
            for weaker in ("druh", current()["name_predicate"]):
                values, fact = self._match(weaker, known_set, pat.hole_role,
                                           pat.hole_type)
                if values:
                    break
        if not values and pat.hole_role in ("subj", "obj") \
                and self.graph.nodes.get(pat.predicate) is not None \
                and self.graph.nodes[pat.predicate].type == "concept":
            # relační jméno bez vlastního faktu („matka Karla Čapka") → osoba
            # z okolí, která tím druhem JE (apoziční identita + kontext join)
            # — má přednost před holým nejtěžším asociátem
            values, fact = self._typed_match(pat.predicate,
                                             known_set | {pat.predicate})
        if not values and pat.hole_role == "num" and len(known_set) > 1:
            # POČETNÍ díra (#11): počet je vlastnost SITUACE, ne děje —
            # „Kolik měla dětí BN?" nese fakt přestěhovat(num čtyři,
            # theme dítě, subj BN); hledá se fakt se VŠEMI známými
            # napříč predikáty a bere se jeho číselný účastník
            for fact in self.graph.facts_of(node0):
                if not known_set <= {p.node for p in fact.participants}:
                    continue
                nums = [p.node for p in fact.participants
                        if p.role == "num"]
                if nums:
                    values, fact_hit = nums, fact
                    return (node0, values, fact_hit)
        if not values and pat.hole_role in ("subj", "obj"):
            # KONTEXTOVÉ patro (asociace jako odpověď) — NÁLEZ B4:
            # vlajkové „Kdo napsal R.U.R.?" je odpověď asociace (fakt
            # napsat s R.U.R. v korpusu není!) — patro je nosné. Hranice
            # figlů: u HUBOVÉHO tématu je asociace mnohoznačná (potkal/
            # přišel/zradil… — druh vztahu nevíme) → neodpovídat, PTÁME
            # SE (kaskáda: částečná odpověď / nabídka kandidátů).
            # HRANICE = rozvětvenost: asociační HUB (Ježíš 209 sousedů)
            # konkrétní vztah nenese, řídké téma (R.U.R. 24) ano —
            # změřeno; práh context_hub_limit v configu. Verdikt False
            # (role neexistuje) řeší chytrá clarifikace (E3).
            verdict = instance_lit(getattr(pat, "predicate", None),
                                   pat.hole_role, self._ring_roles)
            fanout = sum(1 for f in self.graph.facts_of(
                node0, predicate="kontext") for p in f.participants
                if p.node != node0)
            if verdict is None or (verdict is True
                                   and fanout <= self.context_hub_limit):
                values, fact = self._match("kontext", known_set,
                                           pat.hole_role, pat.hole_type)
        if not values and len(known_set) > 1:
            # výběrová otázka: konceptový known bez místa ve faktu = TYPOVÝ
            # filtr díry (join: napsat(X, ?) ∧ být(?, hra) → „Jakou hru…")
            values, fact = self._typed_match(pat.predicate, known_set)
        if not values and pat.hole_role in ("pred", "attr") \
                and self.context.scores.get(node0, 0.0) > 0:
            # NAVAZUJÍCÍ identitní otázka („Jaká rodina?" po odpovědi „rodina")
            # smí čerpat souvislosti — téma svítí z konverzace; svěží identitní
            # dotaz zůstává poctivě bez hádání
            values, fact = self._match("kontext", known_set, "subj", None)
        return (node0, values, fact) if values else (None, [], None)

    def _solve(self, known, predicate=None):
        """Rekurzivně vyřeší `known` na uzel: list = přímá entita; `SubQuery` =
        vnořený pod-dotaz (vyřeš jeho známé rekurzivně, pak `_match`). **Samo-
        rozbalování/zabalování**: hloubka = zanoření otázky, auto-trigger struktura.

        Args:
            known (str | SubQuery): Termín entity nebo vnořený pod-dotaz.

        Returns:
            str | None: Id uzlu, nebo None když nejde vyřešit.
        """
        if not isinstance(known, SubQuery):
            return self._resolve_topic(known.split(), predicate)
        sub = set()
        for _, inner in known.known:
            node = self._solve(inner, known.predicate)   # rekurze do hloubky
            if node is None:
                return None
            sub.add(node)
        values, _ = self._match(known.predicate, sub, known.hole_role, None)
        if not values and known.hole_role in ("subj", "obj"):
            # i vnořený skok smí spadnout do kontextové asociace (rekurze
            # „autor, který napsal X" bez explicitního napsat-faktu)
            values, _ = self._match("kontext", sub, known.hole_role, None)
        return values[0] if values else None

    def _match(self, predicate, known_set, hole_role, hole_type):
        """Jeden skok matche: najdi fakty s `predicate` obsahující všechny
        `known_set` a vrať **všechny rovnocenné díry** (jiné účastníky s top
        skóre) — role/typ díry je preference (ne filtr), ranking dělá váha
        faktu + aktivace. Víc rovnocenných faktů = výčtová odpověď („Co napsal
        X?"); jednohodnotová otázka vrátí jednoprvkový seznam. Pro rekurzi.

        Returns:
            tuple: (list[str] hodnoty s top skóre, fakt nejlepší | None).
        """
        if not known_set:
            return [], None
        node0 = next(iter(known_set))
        scored = []
        ring = list(_synonym_ring(predicate))
        if hole_role == "loc" or hole_type == "geo":
            ring += _movement_ring(predicate)
        for pred, exact in ring:
            for fact in self.graph.facts_of(node0, predicate=pred):
                if not known_set <= {p.node for p in fact.participants}:
                    continue
                if any(not any(p.node == n and p.role == "theme"
                               for p in fact.participants)
                       for n in self.turn.theme_bound):
                    # ROLOVĚ VÁZANÝ známý (#55): explicitní adresát musí
                    # být v roli theme — dativ nesmí matchnout mluvčího
                    continue
                if self._time_excluded(fact):
                    continue
                if self.place_filter is not None \
                        and hole_role in ("subj", "obj") \
                        and not any(p.role == "loc"
                                    for p in fact.participants):
                    # MÍSTNÍ filtr u obsahové díry (B7/T7): fakt bez
                    # místa nesmí vyrábět falešné „v Kafarnaum" odpovědi
                    continue
                for part in fact.participants:
                    if part.node in known_set or len(part.node) < 2:
                        continue             # díra ≠ známé; 1-znak = artefakt NER
                    if part.role == "theme" \
                            and part.node == current()["user_entity"]:
                        # POZOROVATEL není odpověď: uživatel v roli theme je
                        # metadata zápisu Mnemos, ne účastník děje (#34)
                        continue
                    if hole_type == "person" \
                            and part.role in ("theme", "loc", "time", "num"):
                        # OSOBNÍ díra nebere okolnosti (T4 „den", T11
                        # „Boží Hora"): kdo/koho plní jen subj/obj/pred;
                        # obsahové díry bez typu (řeč) theme odpovídat
                        # SMÍ (143 výroků)
                        continue
                    if part.role == "time" and hole_role != "time" \
                            and hole_type != "time":
                        # časová kotva odpovídá jen časové díře
                        continue
                    if hole_type == "geo" and not exact \
                            and (node := self.graph.nodes.get(part.node)) \
                            and node.type == "person":
                        # OSOBA jako místo („přišel K JOSEFOVI") jen u
                        # PŘESNÉHO predikátu — synonymní kruh („přespal"
                        # ≈ přijít) ji na „kde" půjčovat nesmí; rozhoduje
                        # typ UZLU (typ účastníka je u geo jmen šumový)
                        continue
                    if pred in ("být", "druh", "kontext") \
                            and any(part.node.lower() in k.lower().split()
                                    for k in known_set):
                        # identitní echo slova tématu („Adam stvořitel" →
                        # „stvořitel") = tautologie; dekompoziční predikáty
                        # (rok z „13. ledna 1890") echo naopak CHTĚJÍ
                        continue
                    if hole_role in ("pred", "attr") \
                            and ((pred == "být"
                                  and part.node in current()["relational_nouns"])
                                 or part.node in current()["interrogative_pronouns"]):
                        # sponové „byl bratrem" je vztah k protistraně, ne
                        # identita; apoziční druh („matka Maria") pojmenování JE
                        # vztahové jméno („bratr") není identita osoby bez
                        # protistrany — profese/druh ano
                        continue
                    base = fact.weight + (10 if exact else 0) \
                        + self._source_bonus(fact)
                    if self.pick_focus is not None \
                            and any(q.node == self.pick_focus
                                    for q in fact.participants):
                        # VOLBA OBLASTI DOMINUJE (#5): uživatel oblast
                        # VYBRAL — fakt s ní přebíjí váhu i role ostatních
                        base += 10000
                    # aktivace VŠECH účastníků faktu (poloviční) — volba
                    # oblasti („učedníkům") pak zvedne výroky JEJÍHO faktu
                    glow = self.context.scores.get(part.node, 0.0) \
                        + 0.5 * sum(self.context.scores.get(q.node, 0.0)
                                    for q in fact.participants)
                    # „kdy" bere čas i rok-jako-číslo; „jaký" bere i druh
                    # (pred) — druh je vlastnost; jinak přesná role díry
                    matched = False
                    if hole_role and (part.role == hole_role
                                      or (hole_type == "time"
                                          and part.role in ("time", "num"))
                                      or (hole_role == "attr"
                                          and part.role == "pred")):
                        base += 1000
                        matched = True
                    if hole_type and part.type == hole_type:
                        base += 100
                        matched = True
                    if hole_role in ("loc", "time", "num", "pred") and not matched:
                        continue    # sémantická díra bez shody role/typu mlčí
                        #               („Kdy zemřel?" nesmí vrátit téma faktu)
                    scored.append((base, glow, part.node, fact))
        if not scored:
            return [], None
        # remíza se určuje ZÁKLADNÍM skóre (váha+role/typ) — aktivace remízu
        # jen řadí, nerozbíjí (výčet subj/obj je stabilní napříč konverzací);
        # u UPŘESŇUJÍCÍ díry (attr/pred — „Jakou hru?") ale svítící část
        # remízy vypíná nezasvícené členy: konverzace už zaostřila
        scored.sort(key=lambda t: (-t[0], -t[1]))
        top = scored[0][0]
        ties = [(act, v) for b, act, v, _ in scored if b == top]
        if hole_role in ("attr", "pred") and any(act > 0 for act, _ in ties):
            ties = [(act, v) for act, v in ties if act > 0]
        distinct = list(dict.fromkeys(v for _, v in ties))
        values = distinct[:_MAX_ENUM]
        everything = list(dict.fromkeys(v for _, _, v, _ in scored))
        if len(everything) > _MAX_ENUM:
            # PŘETÉKAJÍCÍ VÝČET: víc kandidátních hodnot, než odpověď unese
            # („Co řekl Ježíš?" — 143 výroků) — zapamatuj OBLASTI (theme
            # účastníky kandidátních faktů), automat z nich nabídne zaostření
            self.turn.overflow = sorted(
                {p.node for _, _, _, f in scored
                 for p in f.participants if p.role == "theme"
                 and p.node not in known_set})[:8]
        # zapamatuj všechny účastníky použitého faktu → rozsvítí se celá cesta
        self.visited.extend(p.node for p in scored[0][3].participants)
        return values, scored[0][3]

    def _fill_subject(self, pat, qa):
        """Kandidát na elidovaný podmět otázky: nejteplejší rodově shodná OSOBA
        konverzačního těžiště. Jen když otázka podmět nemá, neptá se na něj
        (díra ≠ subj) a predikát není spona (identita má vlastní sémantiku).

        Returns:
            str | None: Id osoby, nebo None.
        """
        if pat.hole_role == "subj" or pat.predicate == "být" \
                or any(role == "subj" for role, _ in pat.known):
            return None
        for candidate in self._context_candidates():
            node = self.graph.nodes.get(candidate)
            if node is not None and node.type == "person" \
                    and self._gender_ok(qa, candidate):
                return candidate
        return None

    def _event_answer(self, known_set, depth=1):
        """Salientní DĚJE z OKOLÍ tématu do `depth` skoků (pseudo-n-gramy
        aktivace): bounded BFS po faktech, ranking váha × decay^hloubka.
        Hloubka 1 = přímé fakty („Co se stalo s X?"); hloubka 2 = i sousedé
        („Co víme o X?" — Maria → Ježíš → pokřtěn). Identita/asociace nejsou děj.

        Returns:
            tuple: (téma | None, [děje sloveso-první] | [], nejlepší fakt | None).
        """
        node0 = next(iter(known_set))
        seen_nodes, seen_facts, scored = {node0}, set(), []
        frontier = [(node0, 0)]
        while frontier:
            node, dist = frontier.pop(0)
            for fact in self.graph.facts_of(node):
                if fact.predicate in ("být", "druh", "kontext"):
                    continue
                if fact.id not in seen_facts:
                    seen_facts.add(fact.id)
                    weight = fact.weight * (0.4 ** dist)
                    # aktivace VŠECH účastníků faktu disambiguuje sdílený uzel
                    # („rodina" Bible × Boženy): v navazujícím tahu vyhraje
                    # fakt, jehož okolí svítí z minulé odpovědi
                    warmth = sum(self.context.scores.get(p.node, 0.0)
                                 for p in fact.participants)
                    scored.append((weight + 3.0 * warmth, dist, fact))
                if dist < depth:
                    for p in fact.participants:
                        if p.node not in seen_nodes and p.type == "person":
                            seen_nodes.add(p.node)
                            frontier.append((p.node, dist + 1))
        if not scored:
            return None, [], None
        scored.sort(key=lambda t: (t[1], -t[0]))   # blízké a silné první
        best_fact = scored[0][2]
        self.visited.extend(p.node for p in best_fact.participants)
        values = list(dict.fromkeys(_event_text(f, (node0,))
                                    for _, _, f in scored))[:_MAX_ENUM]
        return node0, values, best_fact

    def _relation_answer(self, pat):
        """Vztahová otázka: fakty SDÍLENÉ dvěma uzly („Jaký měl vztah
        k Janovi?"). Slovesné děje jsou silnější evidence než asociace
        (kontext) a zařazení (být/druh) — čtou se první; elidovaného
        druhého účastníka („Jaký MĚL vztah…") doplní nejteplejší osoba
        těžiště různá od explicitní entity.

        Returns:
            tuple: (téma | None, texty dějů, fakt nejsilnější | None).
        """
        known_set, first_res = set(), None
        for _, known in pat.known:
            node = self._solve(known)
            if node is None:
                return None, [], None
            known_set.add(node)
            if first_res is None:
                first_res = self.turn.resolution
        if first_res is not None:
            self.turn.resolution = first_res
        self._resolved_knowns = set(known_set)
        if not known_set:
            # PLURÁLNÍ ANAFORA („Jaký byl MEZI NIMI vztah?" — #55):
            # „nimi" jsou DVĚ nejteplejší osoby konverzačního těžiště
            persons = [c for c in self._context_candidates()
                       if self.graph.nodes.get(c) is not None
                       and self.graph.nodes[c].type == "person"]
            if len(persons) < 2:
                return None, [], None        # bez dvou žhavých osob nehádat
            known_set = set(persons[:2])
        if len(known_set) == 1:
            anchor = next(iter(known_set))
            other = next((c for c in self._context_candidates()
                          if c != anchor
                          and self.graph.nodes.get(c) is not None
                          and self.graph.nodes[c].type == "person"), None)
            if other is None:
                return None, [], None
            known_set.add(other)
        node0 = next(iter(known_set))
        shared = [f for f in self.graph.facts_of(node0)
                  if known_set <= {p.node for p in f.participants}]
        verbal = [f for f in shared
                  if f.predicate not in ("kontext", "být", "druh")]
        ranked = sorted(verbal or shared, key=lambda f: -f.weight)
        chosen, seen = [], set()
        for fact in ranked:                    # subj/obj zrcadla = týž vztah
            key = (fact.predicate,
                   frozenset(p.node for p in fact.participants))
            if key not in seen:
                seen.add(key)
                chosen.append(fact)
            if len(chosen) == 3:
                break
        if not chosen:
            return None, [], None
        self.visited.extend(p.node for f in chosen for p in f.participants)
        return node0, [_event_text(f) for f in chosen], chosen[0]

    def _is_area(self, token):
        """Slovo otázky je OBLASTÍ gazetteeru (kmenově) — nárok Topos."""
        return _topos_key(token) in self._area_keys

    def _place_excluded(self, fact):
        """Fakt bez účastníka UVNITŘ oblasti otázky (kontejnment Topos).

        Na rozdíl od času je nepřítomnost místa vylučující: fakt, o němž
        nevíme KDE, není evidence pro „v Čechách".
        """
        if self.place_filter is None:
            return False
        return not any(place_within(p.node, self.place_filter,
                                    self._gazetteer)
                       for p in fact.participants)

    def _time_excluded(self, fact):
        """Fakt s časovým účastníkem MIMO interval otázky (tvrdý filtr).

        Mechanismus brány A: interval dodal Chronos z otázky; fakt bez
        časového účastníka se nevylučuje — nedatované časem soudit nelze.
        """
        if self.time_filter is None:
            return False
        times = [p.node for p in fact.participants if p.role == "time"]
        if not times:
            return False
        return not any(self.time_filter.contains_date(parse_date(t))
                       for t in times)

    def _existence_candidates(self, predicate, known_set):
        """Fakty svědčící pro existenci: predikát v synonymním ringu, známí
        účastníci podmnožinou, Chronos/Topos filtry respektovány.

        Yields:
            tuple: (pořadí v grafu, fakt) — pořadí je rozhodčí remízy datace
            (deník paměti je chronologický, korpus předchází paměť).
        """
        rings = {pred for pred, _ in _synonym_ring(predicate)}
        for seq, fact in enumerate(self.graph.facts.values()):
            if fact.predicate not in rings or self._time_excluded(fact) \
                    or self._place_excluded(fact):
                continue
            if known_set and not known_set <= {p.node
                                               for p in fact.participants}:
                continue
            yield seq, fact

    def _existence(self, predicate, known_set):
        """Zjišťovací (ano/ne) otázka: existuje fakt s predikátem a všemi
        známými účastníky? → „Ano"; jinak nic (fallback „nenašel" čte se jako
        nevím). Kontextem se u zjišťovací otázky NEhádá.

        NEGACE (#24): negovaný predikát (`negation_prefix` + predikát) je
        evidence OPAKU — „Prší?" s faktem `neprší(T)` → „Ne, od T neprší."
        Rozhoduje NEJNOVĚJŠÍ evidence: datovaná > nedatovaná (korpus),
        remíza datace → pozdější zápis. Text záporu nese šablona
        `negative_existence_answer` v jazykových datech.

        Returns:
            tuple: (téma | None, [text] | [], fakt | None).
        """
        if predicate is None:
            return None, [], None
        lang = current()
        prefix = lang.get("negation_prefix", "")
        negated = (prefix + predicate) if prefix \
            and not predicate.lower().startswith(prefix) else None
        best = None            # ((datum, pořadí), fakt, je_negace)
        for is_negative, root in ((False, predicate), (True, negated)):
            if root is None:
                continue
            for seq, fact in self._existence_candidates(root, known_set):
                key = (_evidence_date(fact) or (0, 0, 0), seq)
                if best is None or key > best[0]:
                    best = (key, fact, is_negative)
        if best is None:
            return None, [], None
        _, fact, is_negative = best
        self.visited.extend(p.node for p in fact.participants)
        topic = next(iter(known_set)) if known_set else next(
            (p.node for p in fact.participants
             if p.role in ("subj", "obj")), None)
        if not is_negative:
            return topic, ["Ano"], fact
        time_label = next((p.node for p in fact.participants
                           if p.role == "time"), None)
        template = lang.get("negative_existence_answer", "Ne")
        text = template.format(time=time_label, predicate=fact.predicate) \
            if time_label else "Ne"
        return topic, [text], fact

    def _typed_match(self, predicate, known_set):
        """Join výběrové otázky: koncepty z `known_set` se stanou typovým
        filtrem díry — kandidát musí být instancí druhu (`_is_a` přes
        identitní fakty). Kontextové patro je tu povolené: typ dodává
        přesnost, kterou samotná asociace nemá.

        Returns:
            tuple: (list hodnot [nejlepší kandidát], fakt | None).
        """
        concepts = {k for k in known_set
                    if self.graph.nodes.get(k) is not None
                    and self.graph.nodes[k].type == "concept"}
        knowns = known_set - concepts
        if not concepts or not knowns:
            return [], None
        node0 = next(iter(knowns))
        for pred in (predicate, "kontext"):    # přesný predikát před asociací
            best = None
            for fact in self.graph.facts_of(node0, predicate=pred):
                if not knowns <= {p.node for p in fact.participants}:
                    continue
                for part in fact.participants:
                    if part.node in knowns or part.node in concepts \
                            or not self._is_a(part.node, concepts):
                        continue
                    score = fact.weight + self.context.scores.get(part.node, 0.0)
                    if best is None or score > best[0]:  # pylint: disable=unsubscriptable-object
                        best = (score, part.node, fact)
            if best is not None:
                self.visited.extend(p.node for p in best[2].participants)
                return [best[1]], best[2]
        return [], None

    def _is_a(self, node_id, kinds):
        """Instance ↔ druh přes identitní fakty (spona „být" i apoziční „druh")."""
        for predicate in ("být", "druh"):
            for fact in self.graph.facts_of(node_id, role="subj",
                                            predicate=predicate):
                if any(p.role == "pred" and p.node in kinds
                       for p in fact.participants):
                    return True
        return False

    def _reverse_lookup(self, question):
        """Reverzní dotaz: z data v otázce najde událost (datum → podmět faktu).

        Poslední záchrana pro „Co se stalo <datum>?" — datum otázky se rozloží
        (`parse_date`) a páruje se VŠEMI složkami s časovým uzlem: „v listopadu
        1848" nesmí trefit uzel „července 1848" jen kvůli shodě roku.

        Args:
            question (str): Původní dotaz (hledá se v něm rok/měsíc/den).

        Returns:
            tuple: (uzel data | None, podmět | None, fakt | None).
        """
        wanted = parse_date(question)
        if "rok" not in wanted:
            return None, None, None
        for node in self.graph.nodes.values():
            if node.type != "time":
                continue
            have = parse_date(node.id)
            if any(have.get(part) != value for part, value in wanted.items()):
                continue
            value, fact = self._pick(self.graph.facts_of(node.id, role="time"), "subj")
            if value is not None:
                return node.id, value, fact
        return None, None, None

    def _pick(self, facts, role):
        """Z faktů vrátí (hodnotu cílové role, fakt) z faktu s nejvyšší vahou."""
        best_weight, best_value, best_fact = -1, None, None
        for fact in facts:
            values = self.graph.participants(fact, role)
            if values and fact.weight > best_weight:
                best_weight, best_value, best_fact = fact.weight, values[0], fact
        return best_value, best_fact

    def _candidate_facts_roles(self, qa, topic):
        """Vrátí (fakty, cílové role) pro daný typ otázky (zdroj pro alternativy)."""
        g, verb = self.graph, qa.verb_lemma
        if qa.qtype in ("Jaký", "Který"):
            return g.facts_of(topic, role="subj", predicate="být"), ["attr"]
        if qa.is_copula:
            return g.facts_of(topic, role="subj", predicate="být"), ["pred"]
        if qa.qtype in ("Kdy", "Kde", "Kolik"):
            facts = (g.facts_of(topic, role="subj", predicate=verb) if verb
                     else g.facts_of(topic, role="subj"))
            roles = {"Kdy": ["time", "num"], "Kde": ["loc"], "Kolik": ["num"]}[qa.qtype]
            return facts, roles
        if qa.qtype in ("Kdo", "Co"):
            facts = g.facts_of(topic, role="obj", predicate=verb)
            return (facts, ["subj"]) if facts else \
                (g.facts_of(topic, role="subj", predicate=verb), ["obj"])
        return [], []

    def _neighbor_context(self, node_id, exclude, limit=10):
        """Širší kontext kolem uzlu: hodnoty faktů, kde uzel vystupuje jako podmět
        (co ho graf spojuje — pro „Co se stalo <datum>?" tak z Boženy vyplyne i
        „Babička" přes napsat). Bere **top-K podle váhy** (ne úzký práh), aby se
        do kontextu dostaly i řidší, ale věcné vazby."""
        weight_by_value = {}
        for fact in self.graph.facts_of(node_id, role="subj"):
            for participant in fact.participants:
                if participant.node != node_id and participant.node not in exclude:
                    if fact.weight > weight_by_value.get(participant.node, 0):
                        weight_by_value[participant.node] = fact.weight
        ranked = sorted(weight_by_value.items(), key=lambda kv: -kv[1])
        return [v for v, _ in ranked[:limit]]

    def _rank_values(self, facts, roles, temperature):
        """Seřadí hodnoty cílových rolí dle váhy faktu; nechá ty nad prahem teploty."""
        weight_by_value = {}
        for fact in facts:
            for role in roles:
                for value in self.graph.participants(fact, role):
                    if fact.weight > weight_by_value.get(value, 0):
                        weight_by_value[value] = fact.weight
        ranked = sorted(weight_by_value.items(), key=lambda kv: -kv[1])
        if not ranked:
            return []
        top = ranked[0][1]
        return [v for v, w in ranked if w >= top * (1.0 - temperature)]

    def answer(self, question, retrieved, *, temperature=0.0,
               pick_focus=None):
        """Odpoví 2-skokem grafu; při neúspěchu deleguje na fallback.

        Uloží i `last_trace` (téma → fakt → hodnota). **Teplota** `> 0` navíc vrátí
        v `Answer.alternatives` další kandidáty (nad prahem `top × (1 − temperature)`)
        — vhodné, když z nich kompozitor skládá delší text.

        Args:
            question (str): Dotaz uživatele.
            retrieved (list): Pasáže (jen pro fallback).
            temperature (float): Teplota shody (0 = jen primární, 1 = široce).

        Returns:
            Answer: Odpověď z grafu (zdroj „graf"), nebo výsledek fallbacku.
        """
        self.begin_turn()             # čistý výsledek tahu (postřeh 2.1)
        self.pick_focus = pick_focus  # volba oblasti (#5) — VSTUP tahu,
                                      # platí jen tento jeden tah
        # TVRDÝ ČASOVÝ FILTR: časové primitivum otázky („v 19. století",
        # „letos", „21.1.1900") VYŘADÍ fakty s časem mimo interval;
        # nedatované fakty filtr nechává (nelze je vyloučit časem)
        self.time_filter = resolve_temporal(question, self.clock())
        self.place_filter = None   # nastaví brána Q (oblast v otázce)
        # UNIVERZÁLNÍ princip: otázka→neúplný fakt→match→díra (nahrazuje qtype pravidla
        # i attention — kontextově navazující dotaz řeší tatáž cesta). Rozbor dodají
        # VÝHRADNĚ šablony (vzorové karty + pseudo-QL) — řez #14: dotazová
        # strana UDPipe nevolá; šablony nic → nehádat, poctivé „nenašel"
        qa, pat = None, None
        query = build_query(question, self._predicates, self._span_is_node,
                            self._node_word, self._is_area)
        if query is not None:
            qa, pat = query, query.pattern
            self.place_filter = getattr(query, "place", None)
            self.turn.query_card = query.card
        if qa is None:
            qa, pat = Query(), Pattern()
        self.turn.pattern = pat
        self._resolved_knowns = set()
        if getattr(pat, "predicate_class", None):
            # TŘÍDA DĚJŮ (A2): agregace faktů členů třídy pro podmět
            return self._class_answer(pat, question, retrieved)
        topic, values, fact = self._pattern_answer(question, pat, qa)
        for node in self._resolved_knowns:
            # ROZŘEŠENÍ JE ZAOSTŘENÍ: pojmenované entity otázky svítí i bez
            # odpovědi — navazující tah („ano, měl rád knedlíky.") potřebuje
            # vědět, o kom byla řeč. Až PO matchi: vnitřní guardy „je téma
            # žhavé z MINULÉHO tahu?" nesmí vidět jas právě probíhajícího
            # rozřešení (svěží identitní otázka by hádala z kontextu)
            self.context.warm(node, 0.5)
        if getattr(qa, "novelty", False) and values:
            # „Kdo DALŠÍ…?" (#53a): už jmenovaní z minulé odpovědi vypadnou;
            # vyčerpaný výčet odpoví poctivě šablonou (jazyk jako data)
            fresh = [v for v in values if v not in self._last_values]
            if not fresh:
                text = current()["novelty_exhausted_answer"]
                self.turn.trace = {"topic": topic, "predicate": None,
                                   "fact": None, "answer": text}
                self._log_turn(question, topic, None, text)
                return Answer(text=text, sources=["graf"], score=1.0,
                              trace=self.turn.trace)
            values = fresh
        reverse = False
        focus = None                # UZEL odpovědi (paměť/okolí ≠ složený text)
        if not values:
            # poslední záchrana: „Co se stalo <datum>?" — datum → DĚJ
            topic, single, fact = self._reverse_lookup(question)
            if single is not None:
                values, focus, reverse = [_event_text(fact, (topic,))], single, True
        if values:
            if qa.qtype == "Kde":
                values = [(_to_nominative(v, self.client) or v)
                          for v in values]                    # „Slezsku"→„Slezsko"
            # výčtová odpověď: víc rovnocenných děr se vyjmenuje („Co napsal X?")
            self._last_values = list(values)   # pro „Kdo další…?" (#53a)
            text = ", ".join(values)
            self.turn.trace = {"topic": topic, "predicate": fact.predicate,
                               "fact": fact.id, "answer": text}
            focus = focus or values[0]
            for node in self.visited:        # rozsvítí celou (rekurzivní) cestu, ne jen konce
                self.context.warm(node, 0.7)
            for value in values[1:]:         # výčet svítí celý (stabilita opakování)
                self.context.warm(value, 0.7)
            self._warm_sources(fact)         # attention nad soubory (+ jejich graf)
            self._remember(qa, topic, focus)
            self._log_turn(question, topic, fact.predicate, text)
            return Answer(text=text, sources=["graf"], score=1.0,
                          alternatives=self._alternatives(qa, topic, focus,
                                                          reverse, temperature),
                          trace=self.turn.trace)
        # neúspěch NErozmělňuje kontext: attention nesmí vyhasnout jen proto, že
        # jsme odpověď nenašli (jinak by po pár marných dotazech spadla k nule).
        # Pohasíná se jen při úspěchu (v _remember spolu s rozsvícením nového tématu).
        self._log_turn(question, topic, None, None)
        if getattr(pat, "predicate", None) is None and pat.hole_role \
                and self._prev_trace:
            drill_fact = self.graph.facts.get(self._prev_trace.get("fact"))
            if drill_fact is not None:
                # DRILL bez role (T9): fakt roli nenese — vysvětli
                # (empty-role šablona nad predikátem faktu)
                labels = current().get("role_labels", {})
                known = ", ".join(labels[r] for r in labels
                                  if any(p.role == r
                                         for p in drill_fact.participants))
                missing = labels.get(pat.hole_role)
                template = current().get("empty_role_answer")
                if known and missing and template:
                    self.turn.empty_role = (drill_fact.predicate,
                                            pat.hole_role)
                    return Answer(text=template.format(
                        predicate=drill_fact.predicate, known=known,
                        missing=missing), sources=["graf"], score=1.0,
                        trace=None)
        cascade_topic = topic or (self.turn.resolution or {}).get("winner")
        empty = (self._empty_role_answer(pat)
                 or self._partial_fact_answer(pat, cascade_topic)
                 or self._empty_topic_answer(pat, cascade_topic))
        if empty is not None:
            # verdikt je JISTOTA (schéma i fakta prohledána) — automat
            # nesmí odpověď přebít clarify dialogem (assurance-fail)
            self.turn.empty_role = (getattr(pat, "predicate", None),
                                    getattr(pat, "hole_role", None))
            return empty
        return self.fallback.answer(question, retrieved)

    def _empty_role_answer(self, pat):
        """Chytrá clarifikace prázdné díry (#57 E3, šablona empty_role_answer).

        Fakty predikátu roli díry NIKDY nenesou → hledání je marné;
        místo generického terminálu vyjmenovat role, které děj zná.
        Vakuové guardy (past 2): bez predikátu, díry nebo rolí se nesoudí.
        """
        predicate = getattr(pat, "predicate", None)
        hole_role = getattr(pat, "hole_role", None)
        if instance_lit(predicate, hole_role,
                        self._ring_roles) is not False:
            return None
        lang = current()
        labels = lang.get("role_labels", {})
        roles = self._ring_roles(predicate)
        # pořadí výčtu = pořadí tabulky role_labels (kdo, co, kde, kdy,
        # kolik), ne abeceda (T-nález A3)
        known = ", ".join(labels[r] for r in labels if r in roles)
        missing = labels.get(hole_role)
        template = lang.get("empty_role_answer")
        if not known or missing is None or template is None:
            return None
        return Answer(text=template.format(predicate=predicate, known=known,
                                           missing=missing),
                      sources=["graf"], score=1.0, trace=None)

    def _class_answer(self, pat, question, retrieved):
        """Otázka na TŘÍDU dějů (A2): „Jaké zázraky činil Ježíš?" —
        členové třídy z tabulky predicate_classes, odpověď agreguje
        fakty podmětu po predikátech. Zárodek oceánu #41 nad ději."""
        lang = current()
        members = set(lang.get("predicate_classes", {})
                      .get(pat.predicate_class, ()))
        known = next((term for _, term in getattr(pat, "known", ()) or ()
                      if term), None)
        topic = self._resolve_topic(known.split()) if known else None
        groups = {}
        if topic is not None:
            for fact in self.graph.facts_of(topic):
                if fact.predicate not in members:
                    continue
                for p in fact.participants:
                    if p.node == topic or p.role in ("theme", "time"):
                        continue
                    groups.setdefault(fact.predicate, [])
                    if p.node not in groups[fact.predicate]:
                        groups[fact.predicate].append(p.node)
        template = lang.get("class_answer")
        if not groups or template is None:
            return self.fallback.answer(question, retrieved)
        items = "; ".join(f"{pred} — {', '.join(vs[:4])}"
                          for pred, vs in groups.items())
        self.turn.trace = {"topic": topic, "predicate": None,
                           "fact": None, "answer": items}
        self.context.warm(topic, 1.0)
        return Answer(text=template.format(cls=pat.predicate_class,
                                           topic=topic, items=items),
                      sources=["graf"], score=1.0, trace=self.turn.trace)

    def _ring_roles(self, predicate):
        """Role schématu přes CELÝ synonymní/vidový kruh — normalizace
        smí vybrat člen bez faktů (potkal→potkávat) a verdikt nesmí
        spadnout do vakua (B4 nález)."""
        roles = set()
        for pred, _ in (_synonym_ring(predicate or "")
                        + _movement_ring(predicate or "")):
            roles |= self.graph.predicate_roles(pred)
        return frozenset(roles)

    def _partial_fact_answer(self, pat, topic):
        """ČÁSTEČNÁ odpověď (princip user, T11): fakty predikátu s tématem
        EXISTUJÍ, jen díru neumí naplnit (typový guard) — graf se
        k výsledku dostat MUSÍ: vyjmenuje role, které fakty nesou."""
        if self.place_filter is not None:
            # místní filtr (B7): kaskáda nesmí obejít místo otázky
            return None
        predicate = getattr(pat, "predicate", None)
        hole_role = getattr(pat, "hole_role", None)
        if not predicate or not topic or hole_role is None:
            return None
        if predicate in current().get("cascade_skip_predicates", ()):
            # sponové/dekompoziční predikáty nejsou děje — identita má
            # vlastní patra a poctivý terminál (jádro poctivosti!)
            return None
        lang = current()
        labels = lang.get("role_labels", {})
        ring = {p for p, _ in _synonym_ring(predicate)}
        found = {}
        for fact in self.graph.facts.values():
            if fact.predicate not in ring \
                    or all(p.node != topic for p in fact.participants):
                continue
            for p in fact.participants:
                if p.node == topic or p.role == hole_role \
                        or p.role == "theme" or p.role not in labels:
                    continue
                found.setdefault(p.role, [])
                if p.node not in found[p.role]:
                    found[p.role].append(p.node)
        missing = labels.get(hole_role)
        template = lang.get("partial_fact_answer")
        if not found or missing is None or template is None:
            return None
        parts = "; ".join(f"{labels[r]}: {', '.join(vs[:4])}"
                          for r, vs in found.items())
        best = next((f for f in self.graph.facts.values()
                     if f.predicate in ring
                     and any(p.node == topic for p in f.participants)),
                    None)
        if best is not None:
            # trasa: drill („Kde?") po částečné odpovědi funguje (T9/T12)
            self.turn.trace = {"topic": topic, "predicate": best.predicate,
                               "fact": best.id, "answer": parts}
        return Answer(text=template.format(predicate=predicate, topic=topic,
                                           missing=missing, found=parts),
                      sources=["graf"], score=1.0, trace=self.turn.trace)

    def _empty_topic_answer(self, pat, topic):
        """PRÁZDNÉ TÉMA (B4, princip user): predikát roli má, ale žádný
        jeho fakt téma nenese — kontext NEhádá (figl); místo toho SE
        PTÁME: kandidáti = nejtěžší podměty faktů predikátu (nabídku
        s volbou z nich staví automat, volba přehraje otázku)."""
        if self.place_filter is not None:
            # místní filtr (B7): kaskáda nesmí obejít místo otázky
            return None
        predicate = getattr(pat, "predicate", None)
        if not predicate or not topic \
                or getattr(pat, "hole_role", None) is None:
            return None
        if predicate in current().get("cascade_skip_predicates", ()):
            # sponové/dekompoziční predikáty nejsou děje — identita má
            # vlastní patra a poctivý terminál (jádro poctivosti!)
            return None
        ring = {p for p, _ in _synonym_ring(predicate)}
        topic_node = self.graph.nodes.get(topic)
        topic_type = topic_node.type if topic_node is not None else None
        carriers = {}
        for fact in self.graph.facts.values():
            if fact.predicate not in ring:
                continue
            if any(p.node == topic for p in fact.participants):
                return None              # téma fakty nese → jiná cesta
            for p in fact.participants:
                if p.role != "subj":
                    continue
                node = self.graph.nodes.get(p.node)
                if topic_type and node is not None \
                        and node.type != topic_type:
                    # kandidát nabídky má TYP tématu („koho" = osoba) —
                    # pozn.: person-slepence (Tyč, T14) propustí jen
                    # vadný typ v datech, to řeší hygiena (dávka D)
                    continue
                carriers[p.node] = carriers.get(p.node, 0) + fact.weight
        template = current().get("empty_topic_answer")
        if not carriers or template is None:
            return None
        top = [n for n, _ in sorted(carriers.items(),
                                    key=lambda kv: -kv[1])][:5]
        self.turn.empty_topic = (topic, top)   # kandidáti pro nabídku
        return Answer(text=template.format(predicate=predicate, topic=topic,
                                           known=", ".join(top)),
                      sources=["graf"], score=1.0, trace=None)

    def _alternatives(self, qa, topic, value, reverse, temperature):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Alternativy odpovědi pro teplotu > 0 — fuzzy kandidáti (další hodnoty
        s menší vahou) + širší kontext (okolí odpovědi v grafu). Krmivo pro
        kompozitor/NN; při teplotě 0 prázdné."""
        if not temperature:
            return []
        if reverse:
            facts, roles = self.graph.facts_of(topic, role="time"), ["subj"]
        else:
            facts, roles = self._candidate_facts_roles(qa, topic)
        alternatives = [v for v in self._rank_values(facts, roles, temperature)
                        if v != value]
        for near in self._neighbor_context(value, {value, topic}):
            if near not in alternatives:
                alternatives.append(near)
        return alternatives

    def _log_turn(self, question, topic, predicate, answer):
        """Zapíše tah do historie (trajektorie těžiště přes graf).

        Args:
            question (str): Otázka tahu.
            topic (str | None): Rozřešené téma (i z těžiště).
            predicate (str | None): Predikát použitého faktu (None u fallbacku).
            answer (str | None): Odpověď z grafu (None u fallbacku).
        """
        self.history.append({"question": question, "topic": topic,
                             "predicate": predicate, "answer": answer,
                             "gravity": self.context.hottest()})

    def _warm_sources(self, fact):
        """Rozsvítí zdrojové dokumenty faktu a VYZÁŘÍ po grafu dokumentů
        (doc_links) — týž princip jako spread nad termíny: sousední soubory
        (sdílejí entity) dostanou zlomek jasu. Pak pohasne (× decay)."""
        links = getattr(self.graph, "doc_links", {})
        for doc in getattr(fact, "source", ()):
            self.source_context.warm(doc, 3.0)
            neighbors = sorted(links.get(doc, {}).items(),
                               key=lambda kv: -kv[1])[:8]
            for near, _ in neighbors:
                self.source_context.warm(near, 3.0 * self.spread_falloff)
        self.source_context.step()

    def _source_bonus(self, fact):
        """Attention nad SOUBORY: bonus faktu podle jasu jeho zdrojových
        dokumentů. Sloučený uzel (dvě Marie z různých korpusů) tak upřednostní
        fakt z domény, o které se právě mluví — provenience místo rozpadu uzlu."""
        return 2.0 * sum(self.source_context.scores.get(doc, 0.0)
                         for doc in getattr(fact, "source", ()))

    def _spread(self, sources, base=0.6, fanout=8):
        """SPREADING ACTIVATION do HLOUBKY: jas vyzařuje z uzlů trasy (`sources`)
        po role-hranách do `spread_depth` skoků, útlum `spread_falloff^dist`.
        Silná hrana (těžší fakt) nese jas dál; opakované protnutí uzel akumuluje
        (warm je aditivní), takže napříč tahy okolí obecně sílí a lépe ukotvuje
        téma. Výroky (obsah řeči) se nešíří. Depth 0 = jen trasa (žádné okolí).

        Args:
            sources (iterable): Uzly trasy (téma i odpověď — okolní n-gramy obou).
            base (float): Výchozí jas zdroje pro vyzařování (na skok se tlumí).
            fanout (int): Nejvíc sousedů na uzel a skok (top-K dle váhy hrany).
        """
        frontier = {node: base for node in sources if node}
        visited = set(frontier)
        for dist in range(self.spread_depth):
            nxt = {}
            for node, energy in frontier.items():
                neighbors = {}
                for fact in self.graph.facts_of(node):
                    for p in fact.participants:
                        if p.node != node and p.type != "výrok":
                            neighbors[p.node] = max(neighbors.get(p.node, 0.0),
                                                    float(fact.weight))
                top = sorted(neighbors, key=neighbors.get, reverse=True)[:fanout]
                give = energy * self.spread_falloff
                for near in top:
                    self.context.warm(near, give)
                    if near not in visited:
                        visited.add(near)
                        nxt[near] = max(nxt.get(near, 0.0), give)
            frontier = nxt

    def _remember(self, qa, topic, value):
        """Zapíše tah do konverzačního těžiště a pohasí ho.

        Odpověď-entita (Kdo/Co) se rozsvítí nejsilněji — bývá tématem navazující
        otázky („…kdo napsal X? Y. …kdy se narodila?"). U hodnotových odpovědí
        (datum/místo) drží těžiště téma. Pak se vše pohasí (× decay).

        Args:
            qa (QuestionAnalysis): Rozbor otázky.
            topic (str): Id uzlu tématu.
            value (str): Vrácená odpověď (id uzlu).
        """
        if qa.qtype in ("Kdo", "Co"):
            self.context.warm(value, 2.0)
            self.context.warm(topic, 1.0)
        else:
            self.context.warm(topic, 2.0)
        # rozlij z celé trasy: téma, odpověď i uzly protnuté (rekurzí) —
        # okolní n-gramy ukotvují téma pro navazující dotaz
        self._spread(set(self.visited) | {topic, value})
        self.context.step()
