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
from datetime import datetime, timedelta

from jellyai.graph.canon import deaccent
from jellyai.graph.graph import parse_date
from jellyai.iris.assurance import assurance
from jellyai.iris.subsystems.chronos import (TimeInterval, clock_answer,
                                             format_due, resolve_due,
                                             resolve_plan, resolve_temporal,
                                             resolve_weekday)
from jellyai.iris.subsystems.mnemos import (forget, forget_entity,
                                            forget_interval, name_stem,
                                            note_statement, parse_statement,
                                            persist, remember, replay)
from jellyai.iris.patterns import PatternDeck
from jellyai.iris.presenter import activation_window, docs_window
from jellyai.iris.state import FocusState, PendingFocus, PendingIdentity
from jellyai.lang import current

_PICK_WARMTH = 2.0        # jas vybraného kandidáta (volba = silné zaostření)


class ReminderMessage(str):
    """Text dozrálé připomínky, který navíc nese ADRESÁTA e-mailu (None =
    default). Je to podtřída `str` → všude jinde (konzole, okno, spojení
    v `turn()`, JSON) se chová jako obyčejný řetězec; jen e-mailový kanál
    si přečte `.recipient`. Tím se adresát dostane k odesílání bez zásahu
    do generického rozhraní kanálů."""

    def __new__(cls, text, recipient=None):
        obj = super().__new__(cls, text)
        obj.recipient = recipient
        return obj


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
    memorized: dict = None    # konstatování uložené v tomto tahu (Mnemos) —
                              # vizualizace ho přidá do grafu i s atributy
    forgotten: str = None     # jméno entity zapomenuté v tomto tahu —
                              # vizualizace ji (i její fakty) z plátna odebere


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
        if hasattr(answerer, "clock"):
            # jeden zdroj „teď": hodiny automatu (injektované v testech)
            # musí platit i pro tvrdý časový filtr answereru
            answerer.clock = clock or datetime.now
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
        # VOLBA IDENTITY podmětu rozpracovaného výroku (#43) má přednost —
        # PendingIdentity není otázka k přehrání, ale výrok k dopsání
        identity = self._resume_identity(text)
        if identity is not None:
            return identity
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
            pending = self.state.pending_reminder
            if pending is not None:
                self.state.pending_reminder = None
                low0 = deaccent(text.lower())
                phrase0 = next((p for p in current()["reminder_phrases"]
                                if p in low0), None)
                # „připomeň mi TO v 16:30" = odkaz na čekající (prázdný
                # úkol po očištění), ne nová připomínka
                takeover = (
                    (phrase0 is not None
                     and self._reminder_task(text, phrase0) not in ("", "to"))
                    or any(t.rstrip(".") in current()["plan_cancel_words"]
                           or t.rstrip(".") in current()["plan_move_words"]
                           for t in re.findall(r"[\w:.]+", low0)))
                if not takeover:            # nový příkaz nabídku ukončí
                    due = resolve_due(text, self.clock())
                    if due is not None:
                        if isinstance(pending, dict):   # PŘEPLÁNOVÁNÍ
                            return self._set_reminder(pending["task"], due,
                                                      text, record=pending)
                        return self._set_reminder(pending, due, text)
            forgot = self._forget_command(text)
            if forgot is not None:
                return forgot
            memo = next((p for p in sorted(current()["memorize_phrases"],
                                           key=len, reverse=True)
                         if p in deaccent(text.lower())), None)
            if memo is not None:
                memorized = self._memorize_command(text, memo)
                if memorized is not None:
                    return memorized
            managed = self._plan_manage(text)
            if managed is not None:
                return managed
            sent = self._send_command(text)   # „pošli Jindrovi…" (s adresátem)
            if sent is not None:
                return sent
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
                statement = self._subject_or_clarify(text, statement)
                if isinstance(statement, IrisResponse):
                    return statement     # dialog o identitě podmětu (#43)
            if statement is not None:
                return self._memorize(text, statement)
        # VZPOMÍNÁNÍ („Co jsem ti řekl včera?") — Chronos filtr nad
        # timestampy Mnemos; fráze z tabulky, texty karty memory.recall
        recalled = self._recall_query(text)
        if recalled is not None:
            return recalled
        # DOTAZ NA PLÁN („Mám nějaké naplánované úkoly?") — aktivace jde
        # Chronosu: čekající připomínky, zúžené intervalem otázky
        plans = self._plan_query(text)
        if plans is not None:
            return plans
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
        # ReminderMessage nese adresáta (record["recipient"]) k e-mailovému
        # kanálu; jinak se chová jako řetězec (konzole/okno/join beze změny).
        return [ReminderMessage(
                    card.dialog.format(task=item["task"]) if card
                    else item["task"], recipient=item.get("recipient"))
                for item in due]

    def _recall_query(self, text):
        """„Co jsem ti řekl (včera / dnes / minulý týden)?" — výpis toho,
        co uživatel svěřil paměti, zúžený časovým primitivem otázky
        (Chronos nad timestampy Mnemos). Fakta uživatele = fakta grafu
        s účastníkem `user_entity`; přísloví-poznámky se vypisují doslovně.

        Returns:
            IrisResponse | None: Výpis, nebo None (není dotaz na paměť).
        """
        low = deaccent(text.lower())
        if not any(p in low for p in current()["recall_phrases"]):
            return None
        interval = resolve_temporal(text, self.clock())
        user = current()["user_entity"]
        items = []
        for fact in self.answerer.graph.facts_of(user):
            times = [p.node for p in fact.participants if p.role == "time"]
            if interval is not None and not any(
                    interval.contains_date(parse_date(t)) for t in times):
                continue
            others = [p.node for p in fact.participants
                      if p.node != user and p.role != "time"]
            label = times[0] if times else ""
            items.append(f"{label} — {fact.predicate}: "
                         f"{', '.join(others[:5])}")
        items.sort()
        card = self.deck.best("memory.recall", {"candidates": items})
        if card is None:
            return None
        return self._plan_response(card,
                                   card.dialog.format(items="\n".join(items)),
                                   text)

    def _plan_query(self, text):
        """Dotaz na PLÁN („Mám nějaké naplánované úkoly?", „Nezapomněl jsem
        na něco?") — Iris předá aktivaci Chronosu: vypíší se čekající
        připomínky ze skladu, zúžené časovým primitivem otázky („zítra",
        „dnes večer" — denní část posouvá začátek intervalu). Fráze nese
        tabulka `plan_query_phrases`, texty karty `reminder.list`.

        Returns:
            IrisResponse | None: Výpis plánu, nebo None (není dotaz na plán).
        """
        low = deaccent(text.lower())
        if not any(p in low for p in current()["plan_query_phrases"]):
            return None
        now = self.clock()
        interval = resolve_temporal(text, now)
        day_parts = current()["temporal"].get("day_parts", {})
        part = next((day_parts[t] for t in re.findall(r"[\w:.]+", low)
                     if t in day_parts), None)
        if interval is not None and part is not None:
            hour, minute = (int(x) for x in part.split(":"))
            start = interval.start.replace(hour=hour, minute=minute)
            interval = TimeInterval(start, interval.end,
                                    interval.granularity)
        with self._reminder_lock:
            pending = [dict(item) for item in self.reminders]
        if interval is not None:
            pending = [r for r in pending
                       if interval.contains(datetime.fromisoformat(r["due"]))]
        tasks = []
        for r in pending:
            line = (f"{format_due(datetime.fromisoformat(r['due']), now)} — "
                    f"{r['task']}")
            if r.get("recipient"):        # adresovaná připomínka → komu půjde
                line += f" → {r['recipient']}"
            tasks.append(line)
        card = self.deck.best("reminder.list", {"candidates": tasks})
        if card is None:
            return None
        self._note_card(card.name)
        message = card.dialog.format(tasks="\n".join(tasks))
        response = IrisResponse(
            text=message, kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["chronos"], "patterns": [card.name]})
        self.state.remember(text, response)
        return response

    def _forget_command(self, text):
        """ZAPOMENUTÍ („zapomeň/odstraň/vymaž, že Pavel bydlí na
        Barrandově"): zbytek věty se klasifikuje jako konstatování a maže
        se PŘESNOU shodou tvarů z grafu i deníku (Pavel ≠ Pavla — složený
        pokyn s „ponech…" je bezpečný; klauzule za keep-slovem se ořeže).
        Fráze tokenově (slovo „zapomeň" ≠ „nezapomeň")."""
        lang = current()
        tokens = re.findall(r"[\w:.]+", text)
        lows = [deaccent(t.lower()).rstrip(".") for t in tokens]
        start = None
        for phrase in sorted(lang["forget_phrases"], key=len, reverse=True):
            words = phrase.split()
            for i in range(len(lows) - len(words) + 1):
                if lows[i:i + len(words)] == words:
                    start = i + len(words)
                    break
            if start is not None:
                break
        if start is None:
            return None
        rest = tokens[start:]
        keep = lang["keep_words"]
        cut = next((k for k, t in enumerate(rest)
                    if deaccent(t.lower()).rstrip(".") in keep), None)
        if cut is not None:
            rest = rest[:cut]              # „ponech…" klauzule = pojistka
        while rest and deaccent(rest[0].lower()).rstrip(".") in ("ze", "at"):
            rest.pop(0)
        remainder = " ".join(rest).rstrip(".,")
        if not remainder:
            return None
        forgotten_entity = None       # jméno pro živé odebrání z vizualizace
        if deaccent(rest[0].lower()).rstrip(".") == "co":
            # „zapomeň, CO jsem dnes/včera řekl" — období vybírá Chronos
            interval = resolve_temporal(remainder, self.clock())
            if interval is None:
                return None
            removed = forget_interval(self.answerer.graph, self.memory_path,
                                      interval, current()["user_entity"])
        else:
            statement = parse_statement(remainder, self.clock(), self.deck,
                                        is_node=self._known_word)
            if statement is None:
                # ne konstatování → ZAPOMENUTÍ CELÉ ENTITY („zapomeň na Ronika"):
                # smaž všechny fakty, kde jméno (kmen napříč pády) vystupuje
                removed = forget_entity(self.answerer.graph, self.memory_path,
                                        remainder, current()["user_entity"])
                if not removed:
                    return None
                forgotten_entity = remainder    # → vizualizace odebere entitu
            else:
                removed = forget(self.answerer.graph, self.memory_path,
                                 statement["predicate"], statement["objects"],
                                 current()["user_entity"])
        card = self.deck.best("memory.forget",
                              {"candidates": removed})
        if card is None:
            return None
        message = card.dialog.format(facts="; ".join(removed))
        resp = self._plan_response(card, message, text)
        resp.forgotten = forgotten_entity      # None nebo jméno (pro vizualizaci)
        return resp

    def _memorize_command(self, text, phrase):
        """EXPLICITNÍ příkaz paměti („zapamatuj si, že…", „nezapomeň, …",
        „zapiš si za uši…"): zbytek věty jde běžnou kartovou klasifikací
        konstatování (fakt s místem/časem); bez rozpoznatelné struktury
        (přísloví) se uloží DOSLOVNĚ jako poznámka (karta memory-note) —
        příkaz „pamatuj" znamená persistenci vždy."""
        words = phrase.split()
        # e-mail (@) jako CELEK — jinak by ho re-join přes mezery rozbil dřív,
        # než ho uvidí email-aware parse_statement (spadlo by to do poznámky)
        tokens = re.findall(r"[\w.+-]+@[\w.-]+\.\w+|[\w:.]+", text)
        lows = [deaccent(t.lower()).rstrip(".") for t in tokens]
        start = None
        for i in range(len(lows) - len(words) + 1):
            if lows[i:i + len(words)] == words:
                start = i + len(words)
                break
        if start is None:
            return None
        rest = tokens[start:]
        while rest and deaccent(rest[0].lower()).rstrip(".") in ("ze", "at"):
            rest.pop(0)
        remainder = " ".join(rest)
        if not remainder:
            return None
        statement = parse_statement(remainder, self.clock(), self.deck,
                                    is_node=self._known_word)
        if statement is not None and statement.get("needs_subject"):
            statement = self._subject_or_clarify(text, statement)
            if isinstance(statement, IrisResponse):
                return statement         # dialog o identitě podmětu (#43)
        if statement is None:
            card = self.deck.best("memory.note", {})
            if card is None:
                return None
            statement = note_statement(remainder.rstrip("."), self.clock())
        return self._memorize(text, statement)

    def _plan_manage(self, text):  # pylint: disable=too-many-locals,too-many-branches
        """SPRÁVA PLÁNU dialogem: „zruš všechno na zítra", „posuň všechny
        ze zítra na čtvrtek", „přeplánuj to ze 17 na 20".

        Mechanismus: ZDROJ vybírá interval (ze zítra), hodina (ze 17) nebo
        shoda textu úkolu; bez selektoru je nutné slovo typu „všechno".
        CÍL posunu je den v týdnu (drží se čas dne záznamu), hodina, nebo
        plný termín. Slova nesou tabulky plan_cancel/move/all_words,
        texty karty reminder.cancel/move/manage-miss.

        Returns:
            IrisResponse | None: Potvrzení/miss, nebo None (není správa).
        """
        lang = current()
        low = deaccent(text.lower())
        tokens = [t.rstrip(".") or "." for t in re.findall(r"[\w:.]+", low)]
        cancel = any(t in lang["plan_cancel_words"] for t in tokens)
        move = any(t in lang["plan_move_words"] for t in tokens)
        if not (cancel or move):
            return None
        now = self.clock()
        source_text = low.split(" na ")[0] if move and " na " in low else low
        interval = resolve_temporal(source_text, now)
        hour_match = re.search(r"\bze? (\d{1,2})\b", source_text)
        src_hour = int(hour_match.group(1)) if hour_match else None
        take_all = any(t in lang["plan_all_words"] for t in tokens)
        skip = (lang["plan_cancel_words"] | lang["plan_move_words"]
                | lang["plan_all_words"] | lang["query_skip_words"])
        content = [t for t in tokens
                   if t not in skip and not t.replace(":", "").isdigit()
                   and t not in lang["temporal"].get("day_words", {})
                   and t not in lang["temporal"].get("weekday_forms", {})
                   and t not in lang["temporal"].get("units", {})]

        def selected(record):
            due = datetime.fromisoformat(record["due"])
            event = datetime.fromisoformat(record["event"]) \
                if record.get("event") else None
            if interval is not None and not interval.contains(due) \
                    and not (event is not None and interval.contains(event)):
                return False
            if src_hour is not None and due.hour != src_hour \
                    and not (event is not None and event.hour == src_hour):
                return False
            if interval is None and src_hour is None and not take_all:
                task = deaccent(record["task"].lower())
                return any(w in task for w in content)
            return True

        with self._reminder_lock:
            targets = [r for r in self.reminders if selected(r)]
        if not targets:
            card = self.deck.best("reminder.manage-miss", {})
            if card is None:
                return None
            return self._plan_response(card, card.dialog, text)
        if cancel:
            with self._reminder_lock:
                self.reminders = [r for r in self.reminders
                                  if r not in targets]
                self._save_reminders()
            card = self.deck.best(
                "reminder.cancel", {"candidates": [r["task"] for r in targets]})
            if card is None:
                return None
            tasks = ", ".join(r["task"] for r in targets)
            return self._plan_response(card, card.dialog.format(tasks=tasks),
                                       text)
        target_text = low.split(" na ", 1)[1] if " na " in low else low
        day = resolve_weekday(target_text, now)
        hour_target = next((int(t.rstrip("."))
                            for t in re.findall(r"[\w:.]+", target_text)
                            if t.rstrip(".").isdigit()
                            and int(t.rstrip(".")) < 24), None)
        with self._reminder_lock:
            for record in targets:
                due = datetime.fromisoformat(record["due"])
                if day is not None:        # den v týdnu — čas dne se drží
                    due = day.replace(hour=due.hour, minute=due.minute)
                elif hour_target is not None:
                    due = due.replace(hour=hour_target, minute=0)
                    if due <= now:
                        due += timedelta(days=1)
                else:
                    resolved = resolve_due(target_text, now)
                    if resolved is None:
                        return None
                    due = resolved
                record["due"] = due.isoformat()
                record.pop("event", None)   # explicitní cíl ruší předstih
            self.reminders.sort(key=lambda item: item["due"])
            self._save_reminders()
        card = self.deck.best(
            "reminder.move", {"candidates": [r["task"] for r in targets]})
        if card is None:
            return None
        tasks = ", ".join(
            f"{format_due(datetime.fromisoformat(r['due']), now)} — "
            f"{r['task']}" for r in targets)
        return self._plan_response(card, card.dialog.format(tasks=tasks), text)

    def _plan_response(self, card, message, text):
        """Odpověď správy plánu (potvrzení/miss) + telemetrie karty."""
        self._note_card(card.name)
        response = IrisResponse(
            text=message, kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["chronos"], "patterns": [card.name]})
        self.state.remember(text, response)
        return response

    def _reminder(self, text, phrase):
        """Žádost o připomenutí: úkol + termín (Chronos `resolve_due`).

        Bez termínu se automat ZEPTÁ (karta `reminder.when`, dialog > figly)
        a úkol čeká na doplnění v příštím tahu. Bez karet se nic neděje
        (mechanismus bez karet nemá chování — ZÁKON).

        Returns:
            IrisResponse | None: Potvrzení/dotaz, nebo None (karty mlčí).
        """
        task = self._reminder_task(text, phrase)
        plan = resolve_plan(text, self.clock())
        if plan is not None:
            return self._set_reminder(task, plan["due"], text,
                                      event=plan.get("event"),
                                      offered=plan["offered"])
        if not task:
            return None
        card = self.deck.best("reminder.default", {})
        if card is not None:
            # BEZ TERMÍNU: naplánuj výchozí ofset (jazyková data) a nabídni
            # změnu — navazující určení času tuto připomínku PŘEPLÁNUJE
            minutes = current()["temporal"].get("default_reminder_minutes", 15)
            due = self.clock() + timedelta(minutes=minutes)
            record = {"due": due.isoformat(), "task": task}
            with self._reminder_lock:
                self.reminders.append(record)
                self.reminders.sort(key=lambda item: item["due"])
                self._save_reminders()
            self.state.pending_reminder = record
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
        card = self.deck.best("reminder.when", {})
        if card is None:
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

    def _set_reminder(self, task, due, text, record=None, event=None,
                      offered=False, recipient=None):
        """Uloží připomínku do skladu (nebo PŘEPLÁNUJE existující záznam)
        a potvrdí kartou — `reminder.set` (doslovný termín), NEBO
        `reminder.heads-up` (odvozený default s nabídkou upřesnění;
        navazující určení času záznam přeplánuje). Čas UDÁLOSTI se
        u předstihu ukládá (`event`) — výběr „ze 17" pak míří na událost."""
        card = self.deck.best("reminder.heads-up" if offered
                              else "reminder.set", {})
        if card is None:
            return None
        with self._reminder_lock:
            if record is not None and record in self.reminders:
                record["due"] = due.isoformat()
                record.pop("event", None)   # explicitní čas ruší předstih
            else:
                record = {"due": due.isoformat(), "task": task}
                if event is not None:
                    record["event"] = event.isoformat()
                if recipient is not None:      # adresát e-mailu (jinak default)
                    record["recipient"] = recipient
                self.reminders.append(record)
            self.reminders.sort(key=lambda item: item["due"])
            self._save_reminders()
        if offered:
            self.state.pending_reminder = record
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
                | frozenset(temporal.get("day_parts", {}))
                | frozenset(temporal.get("forward_words", ()))
                | frozenset(temporal.get("advance_words", ()))
                | frozenset(temporal.get("soon_minutes", {}))
                | frozenset(temporal.get("fraction_minutes", {})))
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
        return " ".join(words).rstrip(".")   # větná tečka do úkolu nepatří

    def _resolve_person_email(self, name):
        """E-mail osoby z GRAFU: fakt s účastníkem typu `email` a subjektem,
        jehož kmen odpovídá jménu (shoda napříč pády: Jindrovi→Jindra).
        E-mail žije v grafu jako typovaná hodnota (Mnemos), ne v kódu."""
        target = name_stem(name)
        for fact in self.answerer.graph.facts.values():
            email = next((p.node for p in fact.participants
                          if p.type == "email"), None)
            if email is None:
                continue
            if any(p.role == "subj" and name_stem(p.node) == target
                   for p in fact.participants):
                return email
        return None

    def _send_command(self, text):
        """„Pošli Jindrovi zítra ráno upozornění XYZ" → připomínka s ADRESÁTEM.

        Adresát = jméno za slovem „pošli" (ne zájmeno); e-mail se dohledá
        v grafu (typovaný fakt uložený přes Mnemos). BEZ pojmenovaného
        adresáta jde připomínka na DEFAULT (adresu doplní e-mailový kanál).
        Termín z Chronosu (resolve_plan/due), úkol = text bez fráze, adresáta
        a časových výrazů. Jméno bez známého e-mailu → upřímná výzva.

        Returns:
            IrisResponse | None: Potvrzení/výzva, nebo None (není příkaz pošli).
        """
        lang = current()
        low = deaccent(text.lower())
        tokens = [t.rstrip(".") for t in re.findall(r"[\w:@.+-]+", low)]
        send_words = set(lang.get("send_phrases", ()))
        idx = next((i for i, t in enumerate(tokens) if t in send_words), None)
        if idx is None:
            return None
        pronouns = set(lang.get("recipient_pronouns", ()))
        recipient_email = recipient_name = None
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if nxt is not None and nxt not in pronouns and not nxt.isdigit():
            recipient_name = nxt
            recipient_email = self._resolve_person_email(nxt)
            if recipient_email is None:      # jméno znám, e-mail ne → zeptej se
                return self._info(
                    f"Neznám e-mail pro {nxt}. Řekni mi třeba: zapamatuj si "
                    f"že {nxt} má email nekdo@nekde.cz", text)
        phrase = tokens[idx] + (f" {recipient_name}" if recipient_name else "")
        task = self._reminder_task(text, phrase) or "upozornění"
        plan = resolve_plan(text, self.clock())
        if plan is not None:
            resp = self._set_reminder(task, plan["due"], text,
                                      event=plan.get("event"),
                                      offered=plan["offered"],
                                      recipient=recipient_email)
        else:
            due = resolve_due(text, self.clock())
            if due is None:                  # bez termínu → výchozí ofset
                minutes = lang["temporal"].get("default_reminder_minutes", 15)
                due = self.clock() + timedelta(minutes=minutes)
            resp = self._set_reminder(task, due, text, recipient=recipient_email)
        # potvrzení ať PŘIZNÁ adresáta („pošli Jindrovi" ≠ „připomenu ti")
        if resp is not None and recipient_name:
            resp.text = resp.text.replace(
                "Připomenu", f"Pošlu {recipient_name.capitalize()}", 1)
        return resp

    def _info(self, message, text):
        """Prostá informační odpověď bez karty (výzva k doplnění kontaktu)."""
        response = IrisResponse(
            text=message, kind="answer", assurance=1.0,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used={"components": ["chronos"], "patterns": []})
        self.state.remember(text, response)
        return response

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

    def _statement_subject(self, objects, places=()):
        """Podmět připsaného tvrzení: EXPLICITNÍ osoba ve výroku má přednost;
        jinak nejteplejší osoba konverzačního těžiště (o kom se právě mluvilo).

        Pořadí: (1) rozpětí rozřešitelné na EXISTUJÍCÍ person uzel, (2) explicitní
        JMÉNO (velké písmeno) ještě NEznámé grafu — nová 3. osoba, kterou zápis
        teprve vytvoří (např. „Ronik bydlí v Petrovicích"), (3) fallback na
        aktivaci/těžiště — ten platí jen pro ELIDOVANÝ podmět („ano, měl rád
        knedlíky"). Bez kroku (2) by nová jmenovaná osoba spadla na fallback a
        výrok by se nesmyslně připsal uživateli/nejteplejší osobě.

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
                    return node, objects[:i] + objects[i + size:], span
        places = set(places)
        for i, obj in enumerate(objects):     # explicitní nové jméno > kontext
            if (obj not in places and obj[:1].isupper()
                    and obj.replace("-", "").isalpha() and len(obj) > 1
                    and self.answerer.graph.nodes.get(obj) is None):
                return obj, objects[:i] + objects[i + 1:], obj
        for candidate in self.answerer._context_candidates():  # pylint: disable=protected-access
            node = self.answerer.graph.nodes.get(candidate)
            if node is not None and node.type == "person":
                return candidate, objects, None
        return None, objects, None

    def _subject_or_clarify(self, text, statement):
        """Doplní výroku podmět, NEBO otevře dialog o identitě (#43).

        Jméno, které se na osobu grafu rozřeší jen ČÁSTEČNĚ („Emil" →
        „Emil Filla"), se nepřipisuje mlčky — zákon dialog > figly: karta
        clarify-identity nabídne existující osobu i založení nové. Přesná
        shoda ani elidovaný podmět (kontext) se nedoptávají.

        Returns:
            IrisResponse | dict | None: Dialogová nabídka, NEBO výrok
            s podmětem, NEBO None (není komu připsat).
        """
        subject, rest, span = self._statement_subject(
            statement["objects"], statement.get("places", ()))
        if subject is None:
            return None
        statement = dict(statement, subject=subject, objects=rest)
        if span is None or subject == span:
            return statement
        card = self.deck.best("statement.subject",
                              {"features": {"inexact_person"}})
        if card is None:
            return statement           # bez karty platí původní chování
        label_new = f"nový {span}"
        prompt = card.dialog.format(candidates=f"{subject}, {label_new}",
                                    term=span)
        self.answerer.context.warm(subject,
                                   card.action.get("warm_candidates", 0.0))
        self.state.pending = PendingIdentity(
            text=text, statement=statement,
            options=[(subject, subject), (label_new, span)], card=card.name)
        response = IrisResponse(
            text=prompt, kind="dialog", assurance=0.5,
            activation_window=activation_window(self.answerer),
            docs_window=docs_window(self.answerer),
            used=self._used([card.name]),
            clarify={"prompt": prompt, "candidates": [subject, label_new]})
        self.state.remember(text, response)
        return response

    def _resume_identity(self, text):
        """Rozpozná volbu identity podmětu z rozpracované nabídky (#43).

        Volba jménem (průnik slov), „ano" = první nabídnutý, slovo z tabulky
        `new_person_words` = založit novou osobu se jménem z výroku. Otazník
        i nerozpoznaný text nabídku ruší (jde o nové téma). Remíza průniku
        („Emil" sedí na obojí) drží existující osobu — první možnost.

        Returns:
            IrisResponse | None: Potvrzení zápisu, nebo None (jiný vstup).
        """
        pending = self.state.pending
        if not isinstance(pending, PendingIdentity):
            return None
        self.state.pending = None          # nabídka je jednorázová
        if "?" in text:
            return None
        words = {deaccent(w.lower()) for w in re.findall(r"[\w.]+", text)}
        if not words:
            return None
        chosen = None
        if words & set(current().get("new_person_words", ())):
            chosen = pending.options[-1]
        elif words <= {"ano", "jo", "yes"}:
            chosen = pending.options[0]
        else:
            overlaps = [(len(words & {deaccent(w.lower())
                                      for w in label.split()}),
                         (label, subj)) for label, subj in pending.options]
            best = max(overlaps, key=lambda o: o[0])
            chosen = best[1] if best[0] > 0 else None
        if chosen is None:
            return None                    # nové téma — nabídka končí
        statement = dict(pending.statement, subject=chosen[1])
        self.answerer.context.warm(chosen[1], _PICK_WARMTH)
        return self._memorize(pending.text, statement)

    def _known_word(self, token):
        """DOSLOVNÉ slovo uzlu grafu? Veto pro detekci slovesa v Mnemos:
        „nádraží" (věc) sloveso není, „prší" (v grafu nefiguruje) ano.
        Záměrně bez kmenových pater — volná shoda by vetovala i slovesa
        („prší"≈„prsa"). Cache se doplňuje při zápisu paměti."""
        if self._words is None:
            # bez výrokových uzlů — obsah řeči jsou celé věty („bydlí"
            # ve výroku nesmí vetovat sloveso); táž sémantika jako
            # answerer._node_word
            self._words = {word for node in self.answerer.graph.nodes.values()
                           if node.type != "výrok"
                           for word in node.id.lower().split()}
        return token.lower() in self._words

    def _nominativize_name(self, token, client):
        """Nominativ jména přes morfo LEMMA (Karlem→Karel) — TÝŽ princip jako
        korpusová `nominativize` (PROPN lemma = nominativ). Bezpečné tam, kde
        holý kmen ne: Pavel→Pavel, Pavla→Pavla (různí lidé mají různé lemma).
        Mění JEN jednoslovný kapitalizovaný tvar s KAPITALIZOVANÝM lemmatem
        (vlastní jméno); e-mail, obecné slovo, víceslovný text i neznámé jméno
        (lemma == tvar, malé lemma) se nechají."""
        if not token or " " in token or not token[:1].isupper():
            return token
        try:
            analysis = client.analyze(token)
        except Exception:  # pylint: disable=broad-exception-caught
            return token
        if not analysis:
            return token
        lemma = (analysis[0].get("lemma") or "").split("_")[0]
        if lemma and lemma[:1].isupper() and lemma != token:
            return lemma
        return token

    def _nominativize_statement(self, text, statement):
        """Nominativizuje jména a MÍSTA výroku před zápisem. MÍSTA přes UDPipe
        (kontext věty jednoznačně určí pád — Brně→Brno; toponyma nemají past
        rodu jako Pavla/Pavel). JMÉNA přes morpho lemma (Karlem→Karel), kde by
        UDPipe naopak různé lidi slil (Pavla→Pavel). Vrací (příp. upravenou)
        kopii statementu."""
        client = getattr(self.answerer, "client", None)
        if client is None:
            return statement
        statement = dict(statement)
        places = set(statement.get("places", ()))
        try:
            parsed = client.parse(text)
        except Exception:  # pylint: disable=broad-exception-caught
            parsed = []
        place_lemma = {}
        for sent in parsed:
            for tok in sent:
                form = tok.get("form") or ""
                if form in places and tok.get("upos") in ("PROPN", "NOUN"):
                    lemma = (tok.get("lemma") or "").split("_")[0]
                    # nominativ místa nebývá KRATŠÍ než tvar (Brně→Brno 4=4,
                    # Praze→Praha 5=5); kratší = zmršení (Lhotě→Lhot) → zahoď.
                    # VÝJIMKA (#35): známé toponymní koncovky nominativu
                    # (Petrovice, Vodňany) — kratší lemma je legitimní
                    endings = tuple(current().get(
                        "place_nominative_endings", ()))
                    if (lemma and lemma[:1].isupper() and lemma != form
                            and (len(lemma) >= len(form)
                                 or lemma.endswith(endings))):
                        place_lemma[form] = lemma

        def nom(word):
            if word in place_lemma:
                return place_lemma[word]
            if word in places:
                return word              # místo bez verdiktu — morfo by spletlo
            return self._nominativize_name(word, client)   # jméno přes morpho

        statement["objects"] = [nom(o) for o in statement["objects"]]
        statement["places"] = [place_lemma.get(p, p)
                               for p in statement.get("places", ())]
        if statement.get("subject"):
            statement["subject"] = nom(statement["subject"])
        return statement

    def _memorize(self, text, statement):
        """Uloží konstatování do grafu (Mnemos) a rozsvítí jeho uzly.

        Uložení JE zaostření: nové téma svítí, takže navazující otázka
        („Kdy jsem měl…?") jede po zahřáté aktivaci.
        """
        # NOMINATIVIZACE PŘED zápisem (do grafu I deníku → merge pádů, konzistence
        # po restartu i ve vizualizaci): MÍSTA přes UDPipe (kontext věty: Brně→Brno;
        # u toponym není past Pavla/Pavel), JMÉNA přes morpho lemma (bezpečné:
        # Pavla≠Pavel, kdežto UDPipe by je slil).
        statement = self._nominativize_statement(text, statement)
        detail = remember(self.answerer.graph, statement,
                          current()["user_entity"])
        if self.memory_path:
            persist(statement, self.memory_path)   # paměť přežije restart
        places = statement.get("places", ())
        if len(places) >= 2 and hasattr(self.answerer, "_gazetteer"):
            # PŘESAH DO TOPOSU: vnořená místa („na Barrandově v Praze")
            # učí kontejnment za pochodu — zápis subsystému
            from jellyai.iris.subsystems.topos import (area_keys,
                                                       learn_containment)
            for inner, outer in zip(places, places[1:]):
                learn_containment(self.answerer._gazetteer,
                                  getattr(self.answerer, "_gazetteer_path",
                                          None), inner, outer)
            self.answerer._area_keys = area_keys(self.answerer._gazetteer)
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
                  "patterns": patterns},
            memorized=statement)   # vizualizace ho přidá do grafu i s atributy
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
