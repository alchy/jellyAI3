"""Answerer odpovídající 2-skokovým průchodem reifikovaného faktového grafu.

Otázku rozebere `analyze_question`, najde uzel tématu, z něj fakty (dle role a
predikátu) a z faktu s **nejvyšší vahou** vezme účastníka cílové role. N-arita: „kde"
i „kdy" čerpají z téhož narozovacího faktu. Když nic nesedí, deleguje na fallback.
"""

from jellyai.answerer.base import Answer, Answerer
from jellyai.answerer.question import analyze_question
from jellyai.answerer.template import _to_nominative
from jellyai.graph.activation import ActivationField

_DATE_PARTS = {"rok", "měsíc", "den"}   # drill: „v kterém roce/měsíci…"


class GraphAnswerer(Answerer):
    """Odpovídá z globálního faktového grafu; jinak fallback.

    Drží **konverzační těžiště** (`ActivationField` nad id uzlů): každý tah rozsvítí
    téma a odpověď, každý tah pohasíná. Když navazující otázka nemá vlastní téma
    („Kdy se narodila?"), vezme se nejteplejší uzel z rozhovoru — dialog tak plyne.
    """

    def __init__(self, graph, client, fallback):
        """Vytvoří answerer.

        Args:
            graph (FactGraph): Postavený faktový graf.
            client: ÚFAL klient (rozbor otázky).
            fallback (Answerer): Answerer pro neúspěch (extraktivní/template).
        """
        self.graph = graph
        self.client = client
        self.fallback = fallback
        self.last_trace = None   # trasa poslední odpovědi (téma → fakt → hodnota)
        self.context = ActivationField()   # konverzační těžiště (id uzlu → jas)
        self.history = []        # trajektorie konverzace (tahy s trasou a těžištěm)

    def reset(self):
        """Začne nový rozhovor — vymaže těžiště i historii."""
        self.context = ActivationField()
        self.history = []
        self.last_trace = None

    def _resolve_topic(self, topic_terms):
        """Najde uzel tématu otázky — nejlepší shodu s obsahovými lemmaty.

        Shoda je **case-sensitive** (vlastní jméno „Babička" se nesmí splést
        s obecným „babička"). Preferuje uzel, který pokrývá **víc témat** (aby
        „Božena Němcová" přebila samotnou „Němcová"), pak delší (víceslovnou)
        entitu, a teprve nakonec vyšší frekvenci.

        Args:
            topic_terms (list[str]): Obsahová lemmata otázky.

        Returns:
            str | None: Id uzlu tématu, nebo None když nic nesedí.
        """
        terms = [t for t in topic_terms if t]
        best_id, best_score = None, None
        for node in self.graph.nodes.values():
            words = node.id.split()
            hits = sum(1 for t in terms if t == node.id or t in words)
            if hits == 0:
                continue
            score = (hits, len(words), node.weight)
            if best_score is None or score > best_score:
                best_id, best_score = node.id, score
        return best_id

    def _pick(self, facts, role):
        """Z faktů vrátí (hodnotu cílové role, fakt) z faktu s nejvyšší vahou."""
        best_weight, best_value, best_fact = -1, None, None
        for fact in facts:
            values = self.graph.participants(fact, role)
            if values and fact.weight > best_weight:
                best_weight, best_value, best_fact = fact.weight, values[0], fact
        return best_value, best_fact

    def _traverse(self, qa, topic):
        """Projde graf podle typu otázky; vrátí (hodnota, fakt) nebo (None, None).

        U „kdy/kde/kolik" **trvá na shodě slovesa** (žádná náhrada nesouvisející
        událostí — na „kdy se narodil" nesmí odpovědět datem svatby). N-arita: „kdy"
        i „kde" čerpají z téhož faktu.

        Args:
            qa (QuestionAnalysis): Rozbor otázky.
            topic (str): Id uzlu tématu.

        Returns:
            tuple: (hodnota | None, fakt | None).
        """
        g, verb = self.graph, qa.verb_lemma
        # drill „v kterém roce se narodil X": událost → datum (uzel) → pod-fakt rok
        date_part = next((t for t in qa.topic_terms if t in _DATE_PARTS), None)
        if date_part:
            facts = (g.facts_of(topic, role="subj", predicate=verb) if verb
                     else g.facts_of(topic, role="subj"))
            time_value, _ = self._pick(facts, "time")
            if time_value is None:
                return None, None
            return self._pick(g.facts_of(time_value, role="subj", predicate=date_part), "val")
        if qa.is_copula or qa.qtype in ("Jaký", "Který"):
            return self._pick(g.facts_of(topic, role="subj", predicate="být"), "pred")
        if qa.qtype in ("Kdy", "Kde", "Kolik"):
            facts = (g.facts_of(topic, role="subj", predicate=verb) if verb
                     else g.facts_of(topic, role="subj"))
            if qa.qtype == "Kdy":
                value, fact = self._pick(facts, "time")
                return (value, fact) if value is not None else self._pick(facts, "num")
            if qa.qtype == "Kde":
                return self._pick(facts, "loc")
            return self._pick(facts, "num")
        if qa.qtype in ("Kdo", "Co"):
            value, fact = self._pick(g.facts_of(topic, role="obj", predicate=verb), "subj")
            if value is not None:
                return value, fact
            return self._pick(g.facts_of(topic, role="subj", predicate=verb), "obj")
        return None, None

    def _candidate_facts_roles(self, qa, topic):
        """Vrátí (fakty, cílové role) pro daný typ otázky (zdroj pro alternativy)."""
        g, verb = self.graph, qa.verb_lemma
        if qa.is_copula or qa.qtype in ("Jaký", "Který"):
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
        self.last_trace = None
        qa = analyze_question(question, self.client)
        # téma z otázky, jinak konverzační těžiště (navazující dotaz bez tématu)
        topic = self._resolve_topic(qa.topic_terms) or self.context.hottest()
        if topic is not None:
            value, fact = self._traverse(qa, topic)
            if value is not None:
                if qa.qtype == "Kde":
                    value = _to_nominative(value, self.client) or value   # „Slezsku"→„Slezsko"
                alternatives = []
                if temperature:
                    facts, roles = self._candidate_facts_roles(qa, topic)
                    alternatives = [v for v in self._rank_values(facts, roles, temperature)
                                    if v != value]
                self.last_trace = {"topic": topic, "predicate": fact.predicate,
                                   "fact": fact.id, "answer": value}
                self._remember(qa, topic, value)
                self._log_turn(question, topic, fact.predicate, value)
                return Answer(text=value, sources=["graf"], score=1.0,
                              alternatives=alternatives)
        self.context.step()
        self._log_turn(question, topic, None, None)
        return self.fallback.answer(question, retrieved)

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
