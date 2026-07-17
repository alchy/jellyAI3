"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozebere `analyze_question`, najde uzel tématu, z něj fakty (dle role a
predikátu) a z faktu s **nejvyšší vahou** vezme účastníka cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, deleguje na fallback.
"""

from jellyai.answerer.base import Answer, Answerer
from jellyai.graph.graph import parse_date
from jellyai.lang import current
from jellyai.answerer.question import analyze_question
from jellyai.answerer.pattern import question_pattern, SubQuery
from jellyai.answerer.template import _to_nominative
from jellyai.graph.activation import ActivationField
from jellyai.graph.canon import _stem, name_gender

_DATE_PARTS = {"rok", "měsíc", "den"}   # drill: „v kterém roce/měsíci…"
_MAX_ENUM = 5                           # strop výčtové odpovědi (čitelnost)


def _synonym_ring(predicate):
    """(predikát, exact) + jeho synonyma z jazykových dat — „Kde žili?" najde
    bydlet-fakt; přesný predikát drží přednost bonusem."""
    ring = [(predicate, True)]
    for lemma in current()["predicate_synonyms"].get(predicate, ()):
        if lemma != predicate:
            ring.append((lemma, False))
    return ring


def _event_text(fact, exclude=()):
    """Děj jako odpověď: SLOVESO první, pak účastníci. „Co se stalo?" se ptá
    na děj — faktový uzel je děj reifikovaný, holý podmět odpovědí není."""
    others = [p.node for p in fact.participants if p.node not in exclude]
    return fact.predicate + (": " + ", ".join(others[:5]) if others else "")


class GraphAnswerer(Answerer):
    """Odpovídá z globálního faktového grafu; jinak fallback.

    Drží **konverzační těžiště** (`ActivationField` nad id uzlů): každý tah rozsvítí
    téma a odpověď, každý tah pohasíná. Když navazující otázka nemá vlastní téma
    („Kdy se narodila?"), vezme se nejteplejší uzel z rozhovoru — dialog tak plyne.
    """

    def __init__(self, graph, client, fallback, *, context_decay=0.55):
        """Vytvoří answerer.

        Args:
            graph (FactGraph): Postavený faktový graf.
            client: ÚFAL klient (rozbor otázky).
            fallback (Answerer): Answerer pro neúspěch (extraktivní/template).
            context_decay (float): Pohasínání konverzačního těžiště na dotaz
                (viz `ActivationField.decay`; nižší = kratší paměť kontextu).
        """
        self.graph = graph
        self.client = client
        self.fallback = fallback
        self.context_decay = context_decay
        self.last_trace = None   # trasa poslední odpovědi (téma → fakt → hodnota)
        self._prev_trace = None  # trasa PŘEDCHOZÍHO tahu (drill „Kdy?")
        self.visited = []        # uzly protnuté (rekurzivním) matchem → rozsvícení
        self.context = ActivationField(decay=context_decay)   # těžiště (id uzlu → jas)
        self.history = []        # trajektorie konverzace (tahy s trasou a těžištěm)

    def reset(self):
        """Začne nový rozhovor — vymaže těžiště i historii."""
        self.context = ActivationField(decay=self.context_decay)
        self.history = []
        self.last_trace = None

    def _resolve_topic(self, topic_terms, predicate=None):
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
        terms = [t for t in topic_terms if t]
        low_terms = [t.lower() for t in terms]
        stems = [_stem(t) for t in terms]
        best_id, best_score = None, None
        ring = _synonym_ring(predicate) if predicate else ()
        for node in self.graph.nodes.values():
            if node.type == "výrok":
                continue                  # obsah řeči je hodnota, ne téma
            low_id = node.id.lower()
            low_words = low_id.split()
            ins_hits = sum(1 for t in low_terms if t == low_id or t in low_words)
            node_stems = {_stem(w) for w in low_words}
            stem_hits = sum(1 for s in stems if s in node_stems)
            if ins_hits == 0 and stem_hits == 0:
                continue
            # přesná shoda velikosti má smysl jen u termů NESOUCÍCH velké
            # písmeno (lowercase lemma „vějíř" nerozliší pojem od titulu)
            exact_hits = sum(1 for t in terms
                             if any(ch.isupper() for ch in t)
                             and (t == node.id or t in node.id.split()))
            # PREDIKÁTOVÁ AFINITA: mezi rovnocennými jmennými shodami vyhrává
            # uzel, o němž se predikát otázky dá vypovědět („Vějíř" s napsat
            # faktem) — ale jmenné shody nikdy nepřebije (Ludvík Němec ≠
            # „Němec" s být-faktem jiné osoby)
            affinity = int(any(self.graph.facts_of(node.id, predicate=pred)
                               for pred, _ in ring))
            score = (exact_hits, ins_hits, stem_hits, affinity,
                     len(low_words), node.weight)
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
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

    def _pattern_answer(self, question, qa):  # pylint: disable=too-many-return-statements,too-many-branches
        """Univerzální match: otázka → neúplný fakt → najdi shodný v grafu → díra.

        Nahrazuje ruční qtype/relační pravidla i attention: predikát + známé role
        musí sedět, vrátí se účastník v roli díry (ranking váhou + aktivací). Bez
        explicitní entity (**navazující dotaz**) se téma vezme z **kontextu** —
        projdou se kandidáti od nejteplejšího (gender-filtr) a vezme první, co odpoví.

        Args:
            question (str): Dotaz uživatele.
            qa (QuestionAnalysis): Rozbor (rod pro shodu kontextového tématu).

        Returns:
            tuple: (téma | None, hodnota | None, fakt | None).
        """
        self.visited = []                    # uzly protnuté (i)rekurzí → rozsvítit
        pat = question_pattern(question, self.client)
        if pat.known:
            known_set = set()
            for _, known in pat.known:
                node = self._solve(known, pat.predicate)   # rekurzivně (i vnořené)
                if node is None:
                    return None, [], None    # pojmenované, ale neznámé → nehádat
                known_set.add(node)
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
                # „Co se stalo s X?" — lehké sloveso: odpověď je DĚJ tématu.
                # Otázky s datem nechává reverznímu lookupu (přesné párování
                # složek data > fuzzy rozřešení termínu „května")
                topic, values, fact = self._event_answer(known_set)
                if values:
                    return topic, values, fact
            if qa.qtype is None and pat.hole_role is None:
                # zjišťovací otázka („Napsal X Y?") — existence, ne díra
                return self._existence(pat.predicate, known_set)
            return self._answer_from(pat, known_set)
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
        values, fact = self._match(pat.predicate, known_set, pat.hole_role, pat.hole_type)
        if not values and pat.hole_role in ("pred", "attr"):
            # druhové zařazení (apozice) je slabší evidence než spona „být" —
            # čte se, až když spona mlčí („Co je R.U.R.?" → druh drama)
            values, fact = self._match("druh", known_set, pat.hole_role,
                                       pat.hole_type)
        if not values and pat.hole_role in ("subj", "obj") \
                and self.graph.nodes.get(pat.predicate) is not None \
                and self.graph.nodes[pat.predicate].type == "concept":
            # relační jméno bez vlastního faktu („matka Karla Čapka") → osoba
            # z okolí, která tím druhem JE (apoziční identita + kontext join)
            # — má přednost před holým nejtěžším asociátem
            values, fact = self._typed_match(pat.predicate,
                                             known_set | {pat.predicate})
        if not values and pat.hole_role in ("subj", "obj"):
            # kontextové patro jen pro ENTITNÍ díry (kdo napsal X…); identita/
            # vlastnost (pred/attr) ani zjišťovací otázka (díra None) kontextem
            # nehádají — poctivé „nenašel" je lepší než nejtěžší soused
            values, fact = self._match("kontext", known_set, pat.hole_role, pat.hole_type)
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
        for pred, exact in _synonym_ring(predicate):
            for fact in self.graph.facts_of(node0, predicate=pred):
                if not known_set <= {p.node for p in fact.participants}:
                    continue
                for part in fact.participants:
                    if part.node in known_set or len(part.node) < 2:
                        continue             # díra ≠ známé; 1-znak = artefakt NER
                    if pred in ("být", "druh", "kontext") \
                            and any(part.node.lower() in k.lower().split()
                                    for k in known_set):
                        # identitní echo slova tématu („Adam stvořitel" →
                        # „stvořitel") = tautologie; dekompoziční predikáty
                        # (rok z „13. ledna 1890") echo naopak CHTĚJÍ
                        continue
                    if hole_role in ("pred", "attr") and pred in ("být", "druh") \
                            and (part.node in current()["relational_nouns"]
                                 or part.node in current()["interrogative_pronouns"]):
                        # vztahové jméno („bratr") není identita osoby bez
                        # protistrany — profese/druh ano
                        continue
                    base = fact.weight + (10 if exact else 0)
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
                    scored.append((base, self.context.scores.get(part.node, 0.0),
                                   part.node, fact))
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
        values = list(dict.fromkeys(v for _, v in ties))[:_MAX_ENUM]
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

    def _event_answer(self, known_set):
        """Nejsilnější UDÁLOST tématu (váha + aktivace účastníků) jako děj —
        slovesné fakty; identita (být) a asociace (kontext) děj nejsou.

        Returns:
            tuple: (téma | None, [děj sloveso-první] | [], fakt | None).
        """
        node0 = next(iter(known_set))
        best = None
        for fact in self.graph.facts_of(node0):
            if fact.predicate in ("být", "druh", "kontext"):
                continue
            score = fact.weight + sum(self.context.scores.get(p.node, 0.0)
                                      for p in fact.participants)
            if best is None or score > best[0]:  # pylint: disable=unsubscriptable-object
                best = (score, fact)
        if best is None:
            return None, [], None
        self.visited.extend(p.node for p in best[1].participants)
        return node0, [_event_text(best[1])], best[1]

    def _existence(self, predicate, known_set):
        """Zjišťovací (ano/ne) otázka: existuje fakt s predikátem a všemi
        známými účastníky? → „Ano"; jinak nic (fallback „nenašel" čte se jako
        nevím). Kontextem se u zjišťovací otázky NEhádá.

        Returns:
            tuple: (téma | None, ["Ano"] | [], fakt | None).
        """
        if predicate is None:
            return None, [], None
        node0 = next(iter(known_set))
        for pred, _ in _synonym_ring(predicate):
            for fact in self.graph.facts_of(node0, predicate=pred):
                if known_set <= {p.node for p in fact.participants}:
                    self.visited.extend(p.node for p in fact.participants)
                    return node0, ["Ano"], fact
        return None, [], None

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

    def answer(self, question, retrieved, *, temperature=0.0):
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
        self._prev_trace = self.last_trace or self._prev_trace
        self.last_trace = None
        qa = analyze_question(question, self.client)
        # UNIVERZÁLNÍ princip: otázka→neúplný fakt→match→díra (nahrazuje qtype pravidla
        # i attention — kontextově navazující dotaz řeší tatáž cesta)
        topic, values, fact = self._pattern_answer(question, qa)
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
            text = ", ".join(values)
            self.last_trace = {"topic": topic, "predicate": fact.predicate,
                               "fact": fact.id, "answer": text}
            focus = focus or values[0]
            for node in self.visited:        # rozsvítí celou (rekurzivní) cestu, ne jen konce
                self.context.warm(node, 0.7)
            for value in values[1:]:         # výčet svítí celý (stabilita opakování)
                self.context.warm(value, 0.7)
            self._remember(qa, topic, focus)
            self._log_turn(question, topic, fact.predicate, text)
            return Answer(text=text, sources=["graf"], score=1.0,
                          alternatives=self._alternatives(qa, topic, focus,
                                                          reverse, temperature),
                          trace=self.last_trace)
        # neúspěch NErozmělňuje kontext: attention nesmí vyhasnout jen proto, že
        # jsme odpověď nenašli (jinak by po pár marných dotazech spadla k nule).
        # Pohasíná se jen při úspěchu (v _remember spolu s rozsvícením nového tématu).
        self._log_turn(question, topic, None, None)
        return self.fallback.answer(question, retrieved)

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
        self.context.step()
