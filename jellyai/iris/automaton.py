"""Jádro automatu Iris: tah uživatele → odpověď, NEBO dialogové doostření.

Jeden tah (`turn`) projde vždy stejnou cestou:

1. **Rozpracovaný dialog?** Když minulý tah nabídl kandidáty a text vypadá
   jako volba, otázka se PŘEHRAJE zaostřená — nejednoznačný termín se nahradí
   vybraným kandidátem („Kdo je Čapek?" + volba „Josef Čapek" → „Kdo je Josef
   Čapek?"). Volba se navíc rozsvítí v poli (aktivace nese zaostření dál).
2. **Odpověď answereru** (plugin pseudo-QL) + **QueryAssurance** z evidence
   rozlišení: kvalita jmenných pater / rovnocenní soupeři / jas vítěze.
3. **Rozhodnutí**: jistá odpověď → vrať ji („answer"); odpověď postavená na
   hádání mezi rovnocennými kandidáty → událost `resolve.ambiguous` → karta
   s nabídkou zaostření („dialog"); žádná odpověď, ale rozlišený kandidát →
   událost `focus.low` → upřímný terminál s nejbližšími kandidáty; jinak
   poctivé „nenašel" fallbacku.

Chování NENÍ v kódu — o textech a akcích rozhodují pattern-karty
(`patterns/<jazyk>/*.json`); automat jen hlásí události a vykonává akce
karet (dialog > figly).
"""

import json
import os
import re
import threading

from dataclasses import dataclass, field
from datetime import datetime

from jellyai.graph.canon import deaccent
from jellyai.graph.graph import parse_date
from jellyai.iris.assurance import assurance
from jellyai.iris.chronos import (clock_answer, format_due, resolve_due,
                                  resolve_temporal)
from jellyai.iris.mnemos import parse_statement, persist, remember, replay
from jellyai.iris.patterns import PatternDeck
from jellyai.iris.presenter import activation_window, docs_window
from jellyai.iris.state import FocusState, PendingFocus
from jellyai.lang import current

_PICK_WARMTH = 2.0        # jas vybraného kandidáta (volba = silné zaostření)


@dataclass
class IrisResponse:
    """Výsledek jednoho tahu automatu.

    Atributy:
        text (str): Věta pro dialogové okno (odpověď NEBO řeč automatu).
        kind (str): „answer" (odpověď z grafu/fallbacku) | „dialog" (řeč
            automatu — otázka na doostření či upřímný terminál).
        assurance (float): Jistota zaostření tahu (0–1).
        activation_window (list): [(uzel, jas)] sestupně — UI okno 2.
        docs_window (list): [(dokument, jas)] sestupně — UI okno 3.
        used (dict): Metadata: {"components": [...], "patterns": [...]} —
            které komponenty a pattern-karty se na tahu podílely.
        clarify (dict | None): U dialogu {"prompt", "candidates"}.
        trace (dict | None): Trasa grafu (téma → fakt → hodnota).
        sources (list): Zdroje odpovědi.
        alternatives (list): Fuzzy souvislosti (teplota > 0).
    """
    text: str
    kind: str
    assurance: float
    activation_window: list = field(default_factory=list)
    docs_window: list = field(default_factory=list)
    used: dict = field(default_factory=dict)
    clarify: dict = None
    trace: dict = None
    sources: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)


class IrisAutomaton:
    """Stavový automat zaostření nad jedním answererem (jedna konverzace)."""

    def __init__(self, answerer, deck=None, clock=None, memory_path=None,
                 reminders_path=None):
        """Vytvoří automat.

        ZÁKON: žádné prahy ani rozhodnutí v kódu — kdy se vede dialog a kdy
        odpovídá, určují VÝHRADNĚ karty (triggery `assurance_below`,
        `min_candidates`…). Automat jen hlásí události a vykonává akce.

        Args:
            answerer (GraphAnswerer): Odpovídací plugin (drží graf i pole).
            deck (PatternDeck | None): Balíček pattern-karet; None = vestavěné
                karty češtiny (`patterns/cs/`).
            clock (callable | None): Zdroj „teď" (Chronos — časová kotva);
                None = systémové hodiny, testy vstřikují fixní čas.
            memory_path (str | None): Deník paměti Mnemos (memory.jsonl) —
                při startu se PŘEHRAJE do grafu (paměť přežívá restart);
                None = paměť jen v RAM (testy).
            reminders_path (str | None): Sklad připomínek (JSONL) — KRÁTKODOBÁ
                paměť Chronos: připomínka po odpálení mizí; None = jen RAM.
        """
        self.answerer = answerer
        if deck is None:
            deck = PatternDeck.for_language("cs")
            deck.load()
        self.deck = deck
        self.clock = clock or datetime.now
        self.memory_path = memory_path
        self.state = FocusState()
        self._words = None       # cache doslovných slov uzlů (veto sloves)
        self.telemetry = {}      # jméno karty → {"used", "gain"} (měřený zisk)
        if memory_path:
            restored = replay(answerer.graph, memory_path,
                              current()["user_entity"])
            answerer._predicates |= restored  # pylint: disable=protected-access
        self.reminders_path = reminders_path
        self._reminder_lock = threading.Lock()   # Chronos tep běží v jiném vlákně
        self.reminders = self._load_reminders()

    def reset(self):
        """Nový rozhovor: vymaže rozpracovaný dialog i pole answereru."""
        self.state = FocusState()
        self.answerer.reset()

    def turn(self, text, temperature=0.0):
        """Jeden tah konverzace (viz docstring modulu).

        Před tahem se odpálí DOZRÁLÉ připomínky (Chronos) — jejich texty
        se předřadí odpovědi (plynutí hodin vidí i uživatel bez tikeru).

        Args:
            text (str): Vstup uživatele (otázka NEBO volba kandidáta).
            temperature (float): Teplota answereru (fuzzy souvislosti).

        Returns:
            IrisResponse: Odpověď nebo dialogové doostření + metadata.
        """
        fired = self.fire_due()
        response = self._turn(text, temperature=temperature)
        if fired:
            response.text = "\n".join(fired + [response.text])
        return response

    def _turn(self, text, temperature=0.0):
        """Vlastní tah (bez odpalu připomínek — replay/focus-shift rekurze)."""
        direct = clock_answer(text, self.clock())
        if direct is not None:
            # hodinová otázka — odpovídá Chronos sám (časová kotva), graf se
            # nedotkne; aktivační pole zůstává, jak bylo
            response = IrisResponse(
                text=direct, kind="answer", assurance=1.0,
                activation_window=activation_window(self.answerer),
                docs_window=docs_window(self.answerer),
                used={"components": ["chronos"], "patterns": []})
            self.state.remember(text, response)
            return response
        used_patterns = []
        pick = self._resume_pick(text)
        if pick is not None:
            chosen, text = pick
            used_patterns.append(self.state.pending.card)
            self.state.pending = None
            self.answerer.context.warm(chosen, _PICK_WARMTH)
        elif "?" not in text:
            # PŘIPOMÍNKY (Chronos): doplnění času rozpracované žádosti,
            # nebo nová žádost frází z jazykové tabulky; scénáře nesou
            # karty reminder-set/when/due
            pending_task = self.state.pending_reminder
            if pending_task is not None:
                self.state.pending_reminder = None
                due = resolve_due(text, self.clock())
                if due is not None:
                    return self._set_reminder(pending_task, due, text)
            low = deaccent(text.lower())
            phrase = next((p for p in current()["reminder_phrases"]
                           if p in low), None)
            if phrase is not None:
                reminder = self._reminder(text, phrase)
                if reminder is not None:
                    return reminder
            # POKYN K ZAOSTŘENÍ („v kontextu Bible") má přednost — posvítí
            # na doménu a přehraje předchozí otázku (karta, spec §3e)
            shift = self._focus_shift(text)
            if shift is not None:
                return shift
            # KONSTATOVÁNÍ (ne dotaz, ne volba) → Mnemos: timestamp + graf;
            # DRUH výroku rozhodují karty (utterance.statement), ne kód
            statement = parse_statement(text, self.clock(), self.deck,
                                        is_node=self._known_word)
            if statement is not None and statement.get("needs_subject"):
                subject, rest = self._statement_subject(statement["objects"])
                if subject is None:
                    statement = None     # není komu připsat → nepřipisovat
                else:
                    statement["subject"] = subject
                    statement["objects"] = rest
            if statement is not None:
                return self._memorize(text, statement)
        # ČASOVÁ OSA ZAOSTŘENÍ: primitivum v otázce („tento měsíc", „včera")
        # rozsvítí časové uzly grafu ležící v intervalu JEŠTĚ PŘED odpovědí —
        # ranking answereru pak remízy řadí i podle této aktivace
        interval = resolve_temporal(text, self.clock())
        if interval is not None:
            self._warm_interval(interval)
        # jas PŘED tahem: „zaostřil už uživatel?" — tah sám si téma rozsvítí,
        # takže čtení po odpovědi by jistotu falešně zvedlo
        before = dict(self.answerer.context.scores)
        answer = self.answerer.answer(text, [], temperature=temperature)
        res = self.answerer.last_resolution
        if res is not None:
            glow = before.get(res["winner"], 0.0)
            assur = assurance(res["quality"], len(res["rivals"]), glow)
            candidates = [res["winner"]] + list(res["rivals"])
        else:
            assur = 1.0 if answer.trace else 0.0
            candidates = []
        context = {"assurance": assur, "candidates": candidates,
                   "term": res["term"] if res else text}

        # ROZHODUJÍ KARTY (výběr benefitem — deck.best): automat jen ohlásí
        # událost tahu (odpověď na hádané volbě → resolve.ambiguous;
        # přetékající výčet → data.overflow; bez odpovědi → focus.low)
        if answer.trace:
            card = self.deck.best("resolve.ambiguous", context)
            if card is not None:
                return self._dialog(card, context, text, used_patterns)
            overflow = self.answerer.last_overflow
            if overflow:
                # svítila-li oblast už PŘED tahem (uživatel ZAOSTŘIL — volba
                # hřeje 2.0), výčet se prostě zodpoví; mimoděčné teplo
                # z minulých odpovědí (≤0.7) oblast nezamyká
                over_ctx = {"assurance": assur, "candidates": overflow,
                            "term": context["term"],
                            "features": ({"area_lit"} if any(
                                before.get(area, 0.0) > 1.0
                                for area in overflow) else set())}
                card = self.deck.best("data.overflow", over_ctx)
                if card is not None:
                    return self._dialog(card, over_ctx, text, used_patterns)
            return self._respond(answer, "answer", assur, used_patterns, text)
        card = self.deck.best("focus.low", context)
        if card is not None:
            return self._dialog(card, context, text, used_patterns,
                                await_pick=False)
        return self._respond(answer, "answer", assur, used_patterns, text)

    def _load_reminders(self):
        """Načte sklad připomínek (JSONL {due, task}); chybějící = prázdný."""
        items = []
        if self.reminders_path and os.path.exists(self.reminders_path):
            with open(self.reminders_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        items.append(json.loads(line))
        return items

    def _save_reminders(self):
        """Přepíše sklad připomínek (krátkodobá paměť — odpálené mizí)."""
        if not self.reminders_path:
            return
        directory = os.path.dirname(self.reminders_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.reminders_path, "w", encoding="utf-8") as fh:
            for item in self.reminders:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    def fire_due(self, now=None):
        """Odpálí DOZRÁLÉ připomínky (tep Chronos i začátek tahu).

        Mechanismus: termín ≤ teď → připomínka se vyřadí ze skladu a vrátí
        se její text z karty `reminder.due`. Volá se z vlákna časovače
        i z tahů — sklad chrání zámek.

        Returns:
            list[str]: Texty odpálených připomínek (prázdné = nic nedozrálo).
        """
        now = now or self.clock()
        with self._reminder_lock:
            due = [r for r in self.reminders
                   if datetime.fromisoformat(r["due"]) <= now]
            if not due:
                return []
            self.reminders = [r for r in self.reminders if r not in due]
            self._save_reminders()
        card = self.deck.best("reminder.due", {})
        if card is not None:
            for _ in due:
                self._note_card(card.name)
        return [card.dialog.format(task=item["task"]) if card
                else item["task"] for item in due]

    def _reminder(self, text, phrase):
        """Žádost o připomenutí: úkol + termín (Chronos `resolve_due`).

        Bez termínu se automat ZEPTÁ (karta `reminder.when`, dialog > figly)
        a úkol čeká na doplnění v příštím tahu. Bez karet se nic neděje
        (mechanismus bez karet nemá chování — ZÁKON).

        Returns:
            IrisResponse | None: Potvrzení/dotaz, nebo None (karty mlčí).
        """
        task = self._reminder_task(text, phrase)
        due = resolve_due(text, self.clock())
        if due is not None:
            return self._set_reminder(task, due, text)
        card = self.deck.best("reminder.when", {})
        if card is None or not task:
            return None
        self.state.pending_reminder = task
        self._note_card(card.name)
        message = card.dialog.format(task=task)
        response = IrisResponse(
            text=message, kind="dialog", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["chronos"], "patterns": [card.name]},
            clarify={"prompt": message, "candidates": []})
        self.state.remember(text, response)
        return response

    def _set_reminder(self, task, due, text):
        """Uloží připomínku do skladu a potvrdí kartou `reminder.set`."""
        card = self.deck.best("reminder.set", {})
        if card is None:
            return None
        with self._reminder_lock:
            self.reminders.append({"due": due.isoformat(), "task": task})
            self.reminders.sort(key=lambda item: item["due"])
            self._save_reminders()
        self._note_card(card.name)
        message = card.dialog.format(time=format_due(due, self.clock()),
                                     task=task)
        response = IrisResponse(
            text=message, kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["chronos"], "patterns": [card.name]})
        self.state.remember(text, response)
        return response

    def _reminder_task(self, text, phrase):
        """Úkol připomínky: text bez fráze, časových výrazů a úvodních spojek.

        Časové běhy se mažou i s předložkou před nimi („v 18:30"); vnitřní
        předložky úkolu zůstávají („vybrat maso Z trouby").
        """
        lang = current()
        temporal = lang["temporal"]
        drop = (frozenset(temporal.get("units", {}))
                | frozenset(temporal.get("numerals", {}))
                | frozenset(temporal.get("month_forms", {}))
                | frozenset(temporal.get("day_words", {}))
                | frozenset(temporal.get("forward_words", ()))
                | frozenset(temporal.get("advance_words", ())))
        tokens = re.findall(r"[\w:.]+", text)
        lows = [deaccent(t.lower()) for t in tokens]
        keep = [True] * len(tokens)
        parts = phrase.split()
        for i in range(len(lows) - len(parts) + 1):    # fráze pryč
            if lows[i:i + len(parts)] == parts:
                for j in range(i, i + len(parts)):
                    keep[j] = False
                break
        for i, low in enumerate(lows):                 # časové výrazy pryč
            if low in drop or re.fullmatch(r"\d[\d:.]*", low):
                keep[i] = False
                if i and lows[i - 1] in ("v", "ve"):
                    keep[i - 1] = False
        words = [t for t, k in zip(tokens, keep) if k]
        strip = lang["reminder_strip"]
        while words and deaccent(words[0].lower()) in strip:
            words.pop(0)
        return " ".join(words)

    def _warm_interval(self, interval, warmth=0.5):
        """Rozsvítí časové uzly grafu spadající do intervalu (Chronos osa).

        Mechanismus, ne rozhodnutí: interval → `contains_date` nad
        `parse_date` každého časového uzlu; svítí uzel (spread answereru
        pak jas roznese na fakty kolem něj).
        """
        for node in self.answerer.graph.nodes.values():
            if node.type == "time" \
                    and interval.contains_date(parse_date(node.id)):
                self.answerer.context.warm(node.id, warmth)

    def _note_card(self, name, gain=0.0):
        """Telemetrie karty: počet použití + kumulovaný zisk aktivace."""
        entry = self.telemetry.setdefault(name, {"used": 0, "gain": 0.0})
        entry["used"] += 1
        entry["gain"] += round(gain, 4)

    def _statement_subject(self, objects):
        """Podmět připsaného tvrzení: EXPLICITNÍ osoba ve výroku (nejdelší
        rozpětí objektů rozřešitelné na person uzel) má přednost; jinak
        nejteplejší osoba konverzačního těžiště (o kom se právě mluvilo).

        Returns:
            tuple: (id osoby | None, objekty bez rozpětí podmětu).
        """
        for size in range(len(objects), 0, -1):
            for i in range(len(objects) - size + 1):
                span = " ".join(objects[i:i + size])
                if not self.answerer._span_is_node(span):  # pylint: disable=protected-access
                    continue
                node = self.answerer._resolve_topic(span.split(), warm=False)  # pylint: disable=protected-access
                found = self.answerer.graph.nodes.get(node)
                if found is not None and found.type == "person":
                    return node, objects[:i] + objects[i + size:]
        for candidate in self.answerer._context_candidates():  # pylint: disable=protected-access
            node = self.answerer.graph.nodes.get(candidate)
            if node is not None and node.type == "person":
                return candidate, objects
        return None, objects

    def _known_word(self, token):
        """DOSLOVNÉ slovo uzlu grafu? Veto pro detekci slovesa v Mnemos:
        „nádraží" (věc) sloveso není, „prší" (v grafu nefiguruje) ano.
        Záměrně bez kmenových pater — volná shoda by vetovala i slovesa
        („prší"≈„prsa"). Cache se doplňuje při zápisu paměti."""
        if self._words is None:
            self._words = {word for node_id in self.answerer.graph.nodes
                           for word in node_id.lower().split()}
        return token.lower() in self._words

    def _memorize(self, text, statement):
        """Uloží konstatování do grafu (Mnemos) a rozsvítí jeho uzly.

        Uložení JE zaostření: nové téma svítí, takže navazující otázka
        („Kdy jsem měl…?") jede po zahřáté aktivaci.
        """
        detail = remember(self.answerer.graph, statement,
                          current()["user_entity"])
        if self.memory_path:
            persist(statement, self.memory_path)   # paměť přežije restart
        # nový fakt = nový slovník: predikát musí znát i pseudo-QL parser
        self.answerer._predicates.add(statement["predicate"])  # pylint: disable=protected-access
        if self._words is not None:      # nové uzly paměti do veto cache
            for obj in statement["objects"]:
                self._words.update(obj.lower().split())
        for node in statement["objects"]:
            self.answerer.context.warm(node, 0.7)
        card = self.deck.best("memory.stored", {})
        message = card.dialog.format(fact=detail) if card else detail
        patterns = [statement["card"]] + ([card.name] if card else [])
        response = IrisResponse(
            text=message, kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["mnemos", "chronos"],
                  "patterns": patterns})
        self.state.remember(text, response)
        return response

    def _resume_pick(self, text):
        """Rozpozná volbu kandidáta z rozpracovaného dialogu.

        Volba se páruje volně (bez diakritiky, po slovech): „josef", „Josefa
        Čapka" i „ano" (= první nabídnutý). Nesedí-li nic, text je NOVÁ
        otázka a rozpracovaný dialog se zahodí.

        Returns:
            tuple[str, str] | None: (vybraný kandidát, zaostřená otázka).
        """
        pending = self.state.pending
        if pending is None:
            return None
        if "?" in text:
            # NOVÁ OTÁZKA místo volby — odpověď se netýká zaostření (i kdyby
            # sdílela slova s kandidáty: „Kdo je Ježíš Martu?" po nabídce
            # oblastí); nabídka končí a otázka jde běžnou cestou
            self.state.pending = None
            return None
        words = {deaccent(w.lower()) for w in re.findall(r"[\w.]+", text)}
        if not words:
            return None
        if words <= {"ano", "jo", "yes"}:
            chosen = pending.candidates[0]
        else:
            # vyhrává kandidát s NEJVĚTŠÍM průnikem slov („Josef Čapek"
            # sdílí příjmení s oběma bratry — rozhodne křestní jméno)
            overlaps = [(len(words & {deaccent(w.lower()) for w in c.split()}),
                         c) for c in pending.candidates]
            best = max(overlaps, key=lambda o: o[0])
            chosen = best[1] if best[0] > 0 else None
        if chosen is None:
            self.state.pending = None      # nové téma — dialog končí
            return None
        if pending.term is None:           # volba oblasti — otázka beze změny
            return chosen, pending.question
        # zaostřená otázka: nejednoznačný termín → vybraný kandidát
        question = re.sub(re.escape(pending.term), chosen,
                          pending.question, count=1, flags=re.IGNORECASE)
        if chosen not in question:
            question = pending.question    # termín v otázce nebyl — nech být
        return chosen, question

    def _dialog(self, card, context, question, used_patterns,
                await_pick=None):
        """Vykoná pattern-kartu: text z šablony + akce (warm, čekání na volbu).

        Zisk karty se MĚŘÍ (Δ jasu kandidátů po akci) — telemetrie řídí
        další vývoj karet daty, ne dojmem (spec §2.6c).
        """
        candidates = context["candidates"]
        text = card.dialog.format(candidates=", ".join(candidates),
                                  term=context["term"])
        warmth = card.action.get("warm_candidates", 0.0)
        scores = self.answerer.context.scores
        before = sum(scores.get(node, 0.0) for node in candidates)
        for node in candidates:
            self.answerer.context.warm(node, warmth)
        gain = sum(scores.get(node, 0.0) for node in candidates) - before
        self._note_card(card.name, gain)
        wants_pick = card.action.get("await") == "user-pick" \
            if await_pick is None else await_pick
        if wants_pick:
            # replace_term=false (overflow): volba jen rozsvítí oblast,
            # otázka se přehraje beze změny — zúží ji aktivace
            term = context["term"] if card.action.get("replace_term", True) \
                else None
            self.state.pending = PendingFocus(question, term,
                                              candidates, card.name)
        response = IrisResponse(
            text=text, kind="dialog", assurance=context["assurance"],
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used=self._used(used_patterns + [card.name]),
            clarify={"prompt": text, "candidates": candidates})
        self.state.remember(question, response)
        return response

    def _respond(self, answer, kind, assur, used_patterns, question=None):
        """Zabalí odpověď answereru do IrisResponse + metadata tahu.

        Do historie jde VSTUP uživatele (otázka) — replay pokynu k zaostření
        potřebuje, na co se ptal, ne co jsme odpověděli.
        """
        response = IrisResponse(
            text=answer.text, kind=kind, assurance=round(assur, 3),
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used=self._used(used_patterns), trace=answer.trace,
            sources=answer.sources, alternatives=answer.alternatives)
        self.state.remember(question or answer.text, response)
        return response

    def _focus_shift(self, text):
        """Pokyn k zaostření — karta `utterance.focus-shift` (spec §3e).

        Kód jen měří rysy (fráze z jazykové tabulky + rozřešitelná doména)
        a vykonává akce karty: posvítit na doménu a její dokumentové okolí,
        přehrát poslední otázku v novém světle.
        """
        lang = current()
        low = deaccent(text.lower())
        phrase = next((p for p in lang["focus_shift_phrases"] if p in low),
                      None)
        features, domain, domain_docs = set(), None, []
        if phrase is not None:
            features.add("focus_phrase")
            phrase_words = set(phrase.split())
            tokens = [t.rstrip(".") for t in re.findall(r"[\w.]+", text)]
            rest = [t for t in tokens
                    if deaccent(t.lower()) not in phrase_words
                    and deaccent(t.lower()) not in lang["query_skip_words"]]
            if rest:
                domain = self.answerer._resolve_topic(rest, warm=False)  # pylint: disable=protected-access
                # doména nemusí být uzel — „Bible" je RODINA DOKUMENTŮ
                # (bible_*): shoda jména s id dokumentu (attention nad zdroji)
                links = getattr(self.answerer.graph, "doc_links", {})
                keys = [deaccent(t.lower()) for t in rest]
                domain_docs = [doc for doc in sorted(links)
                               if any(k in deaccent(doc.lower())
                                      for k in keys)]
                if domain is not None or domain_docs:
                    features.add("domain")
        card = self.deck.best("utterance.focus-shift",
                              {"features": features})
        if card is None or (domain is None and not domain_docs):
            return None
        if domain is not None:
            self.answerer.context.warm(domain,
                                       card.action.get("warm_domain", 1.0))
        if card.action.get("warm_documents"):
            if domain_docs:
                # EXPLICITNÍ doména: ostrá množina dokumentů drží do další
                # změny zaostření (rozlišení jmen ji čte jako provenienci)
                self.answerer.domain_docs = frozenset(domain_docs)
            for doc in domain_docs:
                self.answerer.source_context.warm(doc, 3.0)
            if domain is not None:
                for fact in self.answerer.graph.facts_of(domain):
                    self.answerer._warm_sources(fact)  # pylint: disable=protected-access
        prefix = card.dialog.format(domain=domain or ", ".join(domain_docs))
        if card.action.get("replay") == "last-question":
            previous = next((h["text"] for h in reversed(self.state.history)
                             if "?" in h.get("text", "")), None)
            if previous is not None:
                replayed = self.turn(previous)
                replayed.text = f"{prefix} {replayed.text}"
                replayed.used["patterns"] = ([card.name]
                                             + replayed.used["patterns"])
                return replayed
        response = IrisResponse(
            text=prefix, kind="dialog", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["iris"], "patterns": [card.name]})
        self.state.remember(text, response)
        return response

    def _used(self, patterns):
        """Metadata použitých komponent (observabilita API — spec §5)."""
        pat = self.answerer.last_pattern
        components = []
        if pat is not None and pat.predicate is not None:
            components.append("pseudo-ql-parser")
        components.append("graph-answerer" if self.answerer.last_trace
                          else "fallback-extractive")
        return {"components": components, "patterns": patterns}
