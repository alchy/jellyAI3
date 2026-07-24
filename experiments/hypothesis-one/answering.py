#!/usr/bin/env python3
"""⑤ QUERYPARSER — matcher živé otázky na odpověď (parent-model × window-VZOR × aktivace).

Runtime tah: otázka → predikát + role díry + lemmata; slova rozsvítí soubory (② brána) a
NAHRAJÍ (mount) jen jejich fakty/šablony (#60). Kandidáti vznikají DVĚMA cestami:
  • predikát+role (`FactStore`) — PRIMÁRNÍ, parent-model: fakt a otázka se potkají predikátem
    + rolí díry nezávisle na slovosledu (window-VZOR to nesouměří — viz fact_store.py);
  • window-VZOR (`TemplateStore`) — jemný tie-break pro strukturně souběžné případy.
AKTIVACE (spreading) mezi nimi vybere osvětlený; glow-orders-ties (`_rank`): base rozhoduje,
glow řadí jen remízu. Odpověď zpětně přihřeje (řetěz).

Deterministické parsování otázky (UDPipe 2) = jediná živá anotace; rozhodování symbolické.
"""
import os
import json
import math

from grammar_vzor import GrammarVzor
from role_catalog import RoleCatalog
from dataloader import Dataloader
from template_store import TemplateStore
from fact_store import FactStore
from topos import Topos
from activation_field import ActivationField
from annotate_corpus import AnnotateCorpus

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class Answering:
    """Matcher — spojí fakty (`FactStore`), šablony (`TemplateStore`) a aktivaci (`ActivationField`).

    `answer` je celý runtime tah: nahraj dle otázky, sesbírej kandidáty (predikát+role ∪
    window-VZOR), zvaž aktivací (glow-orders-ties), odpověz.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Vytvoří primitiva, zavaděč, fakt-store, šablony, aktivační pole a parser otázek."""
        self.g = GrammarVzor(config_path)
        self.rc = RoleCatalog(self.g, config_path)
        self.dl = Dataloader(config_path)
        self.facts = FactStore(config_path)
        self.store = TemplateStore(config_path)
        self.topos = Topos(config_path)
        self.field = ActivationField(config_path)
        self.parser = AnnotateCorpus(config_path)
        cfg = json.load(open(config_path, encoding="utf-8"))
        # parent-model predikát+role cesta: MĚŘENO regresní bez koreference → default OFF (viz config)
        self.facts_enabled = cfg.get("fact_store", {}).get("enabled", False)
        # Přesná, ale kapacitně omezená routingová cesta nad metadata fact shardů.
        # Při chybějící shodě zůstává beze změny původní tf-idf loader.
        self.entity_retrieval = cfg.get("entity_retrieval", {})
        # KVANTIFIKÁTORY („množství povídek") nejsou nikdy faktická odpověď — data v configu,
        # filtrují se z kandidátů jako echo (viz _candidates). Jazyk = JSON, ne kód.
        self.nonanswer = set(cfg.get("answering", {}).get("nonanswer_lemmas", []))
        # ALIASY (vztah mezi tokeny = táž identita): {doc: {alias: kanon}} — sloučí rozpadlé
        # jméno v assurance (Barbora Panklová ≡ Božena Němcová → jeden vítěz místo remízy).
        self.aliases = self.dl._load_aliases()
        # POLÁRNÍ (ano/ne) otázky: tázací lemmata (i „kolik" bez PronType=Int); jejich
        # NEPŘÍTOMNOST + modalita ? = polární tvar (viz _polar).
        self.interrog = set(cfg.get("answering", {}).get("interrog_lemmas", []))
        # NAUČENÁ ASSURANCE BRÁNA (gate.json = malé pevné váhy): p(odpověď správná)<práh →
        # nehádej (viz docs/naucene-smerovani.html). Offline trénink, RUNTIME deterministický.
        acfg = cfg.get("answering", {})
        self.gate = None
        self.gate_threshold = acfg.get("gate_threshold", 0.3)
        gpath = os.path.join(HERE, "gate.json")
        if acfg.get("gate_enabled", False) and os.path.exists(gpath):
            self.gate = json.load(open(gpath, encoding="utf-8"))

    def _gate_p(self, feat):
        """Pravděpodobnost SPRÁVNOSTI odpovědi dle naučené brány (čistá python inference —
        sigmoid((rys−mu)/sd · w + b)). Bez brány vrací 1.0 (nebrání). Runtime deterministický."""
        g = self.gate
        if not g or len(feat) != g["dim"]:
            return 1.0
        z = g["b"] + sum(((feat[i] - g["mu"][i]) / g["sd"][i]) * g["w"][i] for i in range(g["dim"]))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def _parse(self, question):
        res = self.parser._udpipe2(question)
        if not res:
            return []
        sents, _ = AnnotateCorpus.parse_conllu(res)
        return sents[0] if sents else []

    def _question(self, toks):
        """Z otázky → (query-VZOR stínu, lemmata obsahu, role díry).

        `hole_role` = role tázacího slova přes TÝŽ mapovač jako fakty (`RoleCatalog.role_of`)
        — díra „Kdo" = who, „Co napsal" = whom_what, … (DRY). Bez tázacího slova → (None, …, None).
        """
        piv = next((i for i, t in enumerate(toks)
                    if "Int" in (t.get("feats") or {}).get("PronType", "")), None)
        q_vzor = (self.g.frame_sig(toks, piv, self.g.sentence_modality(toks))
                  if piv is not None else None)
        lemmas = [self.g.canon_lemma(t) for t in toks
                  if t["upos"] not in ("PUNCT",)
                  and "Int" not in (t.get("feats") or {}).get("PronType", "")]
        hole_role = None
        if piv is not None:
            byid = {i + 1: t for i, t in enumerate(toks)}
            hole_role = self.rc.role_of(toks[piv], toks, byid)
        return q_vzor, lemmas, hole_role

    def _answer_role(self, toks, hole_role):
        """Kopulová identita „Kdo/Co JE X?" → hole je KOMPLEMENT (state), ne who/what.

        „Kdo je pes?" se NEPTÁ na who-podmět (pes je známý/horký), ale na to CO pes JE.
        Známý subjekt (pes) vybere fakt, VZOR (role `state`) zajistí, že vezmeme komplement —
        ne nejteplejší who-podmět z jakékoli věty „X je Y" (bůh z bible). Vstup: tokeny, role
        díry. Výstup: efektivní role odpovědi.
        """
        if hole_role in ("who", "what", "what_subject", None):
            cop = self.g.LANG["copula_lemma"]
            has_cop = any(t["deprel"] == "cop" or (t["upos"] == "AUX" and t["lemma"] == cop)
                          for t in toks)
            # „Kdo je X" — tázací slovo je jmenný přísudek, role_of vrací None; přesto kopula
            has_int = any("Int" in (t.get("feats") or {}).get("PronType", "") for t in toks)
            if has_cop and has_int:
                return "state"
        return hole_role

    def _predicate(self, toks):
        """Predikát otázky = lemma hlavního slovesa (root, jinak první VERB), jinak spona.

        JOIN klíč pro parent-model fakt-match: „Kde se NARODIL X" ↔ fakt „NARODIL se v Y".
        Vstup: tokeny otázky. Výstup: lemma predikátu nebo None.
        """
        verbs = [t for t in toks if t["upos"] == "VERB"]
        root = next((t for t in verbs if t["deprel"] == "root"), None)
        if root or verbs:
            return self.g.canon_lemma(root or verbs[0])
        if any(t["upos"] == "AUX" or t["deprel"] == "cop" for t in toks):
            return self.g.LANG.get("copula_lemma", "být")
        return None

    def _relation_query(self, toks):
        """„Kdo byl bratr X?" — kopulový root je VZTAHOVÉ slovo (bratr/otec/manžel…) s genitivní
        entitou → ptáme se na OSOBU vztahu (fakt-hrana z extract_relations). Vrací vztah-lemma
        (predikát pro fakt-match) nebo None. Vztahová slova = jazyková data."""
        reln = set(self.g.LANG.get("relation_nouns", []))
        if not reln or not any(t.get("deprel") == "cop" for t in toks):
            return None
        root_id = next((i + 1 for i, t in enumerate(toks) if t.get("head") == 0), None)
        root = toks[root_id - 1] if root_id else None
        if root is None or root["upos"] != "NOUN":
            return None
        rel = self.g.canon_lemma(root).lower()
        if rel not in reln:
            return None
        has_gen = any(t.get("head") == root_id and (t.get("feats") or {}).get("Case") == "Gen"
                      for t in toks)
        return rel if has_gen else None

    def _candidates(self, q_vzor, predicate, hole_role, known):
        """Sesbírá kandidáty DVĚMA cestami do jednotného tvaru {answer, role, doc, mq, kind}.

        `pred` = parent-model (predikát+role) — PŘESNÝ: přijme jen fakt, který sám nese
        ZNÁMOU entitu otázky (jinak by predikát „napsal"+who zaplavil VŠEMI autory a glow by
        mispicknul — měřeno). Bez resolvované koreference (pro-drop podmět) se tato vazba
        v jedné větě zřídka sejde → cesta je zatím tichá; ožije s vyplněním děr (#díry-first).
        `vzor` = window-VZOR (strukturně souběžné případy). Výstup: list kandidátů (dict).
        """
        cands = []
        if self.facts_enabled:
            for lem, f in self.facts.match(predicate, hole_role):   # parent-model (přesný)
                # známá entita musí být v JINÉ roli než odpověď — ne co-filler slepence
                # (who=[Švejk,Olbracht] není vazba; who=[Karel]+where=? je); viz koref do faktů
                link = {l.lower() for r, lems in f.roles.items() if r != hole_role for l in lems}
                if not (link & known):
                    continue
                cands.append({"answer": lem, "role": hole_role, "doc": f.doc,
                              "fact_ref": [f.doc, f.sent], "mq": 1.0, "kind": "pred"})
        if q_vzor:
            for tpl, mq in self.store.match_scored(q_vzor):         # window-VZOR
                cands.append({"answer": tpl.answer, "role": tpl.role,
                              "doc": tpl.fact_ref[0] if tpl.fact_ref else None,
                              "fact_ref": list(tpl.fact_ref) if tpl.fact_ref else None,
                              "mq": mq, "kind": "vzor"})
        # KVANTIFIKÁTORY nejsou faktická odpověď („napsal množství" → ne „množství")
        return [c for c in cands if (c["answer"] or "").lower() not in self.nonanswer]

    def _known_overlap(self, fact_ref, role, known):
        """Kolik ZNÁMÝCH entit otázky nese fakt kandidáta v JINÉ roli než odpovědní (vazba).

        Entitní vazba parent-modelu: „Kde se narodil Čapek" → fakt narodit s Čapkem v roli who
        (≠ where odpovědi). Co-filler odpovědní role NEHLÁSÍ (slepenec who=[Švejk,Olbracht]).
        Sdíleno oběma cestami (fakt přes `FactStore.by_ref`). Vstup: fact_ref, role odpovědi, známé.
        """
        f = self.facts.by_ref.get(tuple(fact_ref)) if fact_ref else None
        if not f:
            return 0
        link = {l.lower() for r, lems in f.roles.items() if r != role for l in lems}
        return len(link & known)

    def _canon(self, lemma, doc):
        """Kanonická entita: alias (jiné jméno téže osoby v tomtéž dokumentu) → hlavní jméno.
        Doc-scoped (Barbora→Němcová jen ve wiki_božena_němcová) → merge nezasáhne cizí entity."""
        lem = (lemma or "").lower()
        return self.aliases.get(doc, {}).get(lem, lem)

    def _rank(self, cands, known_lemmas, hole_role):
        """Vybere nejlepší kandidát (glow-orders-ties, jako parent).

        Base (STRUKTURA) rozhoduje první, aktivace (glow) řadí jen remízu — leaked světlo
        tak NEMŮŽE přebít strukturní shodu. Base tuple sestupně:
          echo_ok    — odpověď NENÍ slovo z otázky (self-answer „Kdo je pes?"→pes je nesmysl);
          role_fit   — role kandidáta sedí na roli díry (Hašek/who > román/what_subject);
          known_ov   — fakt nese ZNÁMOU entitu otázky (Švejk → Hašek, ne libovolný autor);
          type_fit   — TYP entity sedí na díru (Topos: where → MÍSTO; Svatoňovice > rodina);
          kind_pred  — parent-model (predikát+role) bije window-VZOR (souměřitelnější spoj);
          mq         — exaktní shoda VZORu bije fuzzy;
          glow       — teplo aktivace (poslední — jen remíza).
        Vstup: kandidáti (dict), lemmata otázky, role díry. Výstup: nejlepší kandidát (dict).
        """
        known = {l.lower() for l in known_lemmas}

        def key(c):
            ans = (c["answer"] or "").lower()
            echo_ok = 0 if ans in known else 1
            role_fit = 1 if hole_role and c["role"] == hole_role else 0
            known_ov = self._known_overlap(c.get("fact_ref"), c["role"], known)
            def_fit = 1 if c["role"] == "state" and (c.get("fact_ref") or [0, 1])[1] == 0 else 0
            type_fit = 1 if hole_role == "where" and self.topos.is_place(c["answer"]) else 0
            kind_pred = 1 if c["kind"] == "pred" else 0
            return (echo_ok, role_fit, known_ov, def_fit, type_fit, kind_pred, c["mq"],
                    self.field.weight_answer(c["answer"], c["doc"]))

        return max(cands, key=key)

    def _assurance(self, cands, known_lemmas, hole_role):
        """Jasno vs nejasno (jako parent `assurance`) — dialogový stavový automat.

        Rozhodne, jestli je vítěz strukturně jasný, nebo je víc rovnocenných kandidátů
        (rivalů) → doptat se. Vrací (mode, winner, offer):
          answer  — silná strukturní evidence a unikátní vítěz → odpověz;
          clarify — slabší evidence / víc rovnocenných → „mám data o X, Y — upřesni";
          unsure  — jen echo, žádná struktura → upřímný terminál (nehádej).
        `offer` = top distinktní kandidáti (pro klarifikaci). Vstup: kandidáti, známé, role díry.
        """
        if not cands:
            return "unsure", None, []
        known = {l.lower() for l in known_lemmas}

        def sig(c):
            ans = (c["answer"] or "").lower()
            echo = 0 if ans in known else 1
            rf = 1 if hole_role and c["role"] == hole_role else 0
            ko = 1 if self._known_overlap(c.get("fact_ref"), c["role"], known) else 0
            df = 1 if c["role"] == "state" and (c.get("fact_ref") or [0, 1])[1] == 0 else 0
            tf = 1 if hole_role == "where" and self.topos.is_place(c["answer"]) else 0
            kp = 1 if c["kind"] == "pred" else 0
            glow = self.field.weight_answer(c["answer"], c["doc"])
            return (echo, rf, ko, df, tf, kp), glow, rf + ko + df + tf + kp    # base, glow, síla

        best = {}
        for c in cands:
            base, glow, stg = sig(c)
            a = (c["answer"] or "").lower()
            if a not in best or (base, glow) > best[a][0]:
                best[a] = ((base, glow), c, stg)
        # SLOUČENÍ ALIASŮ (vztah mezi tokeny = táž identita): variantní jména téže osoby
        # → kanonické, glow se SEČTE. Rozpadlá remíza („Barbora" 0.16 vs „Němcová" 0.28)
        # se stane jasným vítězem (0.44). Blast-radius = jen kandidáti s alias variantami.
        if self.aliases:
            merged = {}
            for lem, ((base, glow), c, stg) in best.items():
                canon = self._canon(lem, c.get("doc"))
                if canon not in merged:
                    merged[canon] = [base, glow, c, stg]
                else:
                    m = merged[canon]
                    m[1] += glow                                 # SEČTI světlo přes varianty
                    if base > m[0] or (base == m[0] and lem == canon):
                        m[0], m[2], m[3] = base, c, stg          # reprezentant = kanon/nejsilnější
            best = {k: ((v[0], v[1]), v[2], v[3]) for k, v in merged.items()}
        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
        (win_base, win_glow), win_c, win_stg = ranked[0]
        offer = [c for _bg, c, _s in ranked[:4]]
        # ODSTUP vítěze od druhého: base (struktura) NEBO glow (světlo zaostřilo). Jako parent
        # assurance = quality/(1+rivals) + glow: vítěz s dominantním světlem je jistý i při remíze base.
        if len(ranked) > 1:
            (run_base, run_glow), _rc, _rs = ranked[1]
            base_sep = win_base > run_base
            glow_sep = win_glow > 2.0 * run_glow and win_glow > 0
        else:
            base_sep = glow_sep = True
        if win_stg >= 1 and (base_sep or glow_sep):
            return "answer", win_c, offer          # jasno → odpověz
        if win_stg >= 1 or win_glow > 0:
            return "clarify", win_c, offer         # evidence, ale rovnocenní → doptej se
        return "unsure", win_c, offer              # jen echo → nehádej

    def _hot_entities(self, top=6, floor=0.4, idf_min=1.5):
        """Nejteplejší DISTINKTIVNÍ slova pole = TÉMA z minulých tahů (kontext dialogu).

        Navazující otázka („Kdy se narodil?") nemá explicitní entitu — vezme ji z toho, co
        ještě SVÍTÍ z minula (Karel Čapek). Jen distinktivní (idf ≥ 1.5 — funkční slova mají
        idf ≤ 1.0), ne funkční slova. Vstup: prahy. Výstup: množina horkých entit-lemmat.
        """
        out = set()
        for w, h in sorted(self.field.words.items(), key=lambda x: -x[1]):
            if h < floor:
                break
            if self.field.idf.get(w, 0.0) >= idf_min:
                out.add(w)
                if len(out) >= top:
                    break
        return out

    def _copula_states(self, subj):
        """Kopulové STAVY podmětu (co daný subjekt JE) — z copula faktů, kde subjekt stojí
        v JINÉ roli než state. Pro polární „Je/Byl X Y?" (Y ∈ states → ano, jinak ne)."""
        cop = self.g.LANG["copula_lemma"]
        subj = set(subj)
        states = []
        for f in self.facts.by_predicate.get(cop, []):
            link = {l.lower() for r, lems in f.roles.items() if r != "state" for l in lems}
            if link & subj:
                states += [l.lower() for l in f.roles.get("state", [])]
        return states

    def _fact_cooccur(self, subj, objs):
        """Sejdou se PODMĚT ∈ subj a OBJEKT ∈ objs v JEDNOM faktu (napříč predikáty)?

        Predikát se ZÁMĚRNĚ neváže — obchází synonymii (získat/obdržet cenu, mít/mateřství).
        Spolu-výskyt v jednom faktu = silná evidence spoje → polární „ano". Vrací fakt / None.
        """
        subj, objs = set(subj), set(objs)
        for f in self.facts.by_ref.values():
            vals = [{l.lower() for l in lems} for lems in f.roles.values()]
            if any(v & subj for v in vals) and any(v & objs for v in vals):
                return f
        return None

    def _polar_result(self, ans, fact_ref=None):
        """Odpovědní dict polární smyčky (kontrakt answer())."""
        return {"answer": ans, "mode": "answer", "offer": [], "fact_ref": fact_ref,
                "hole_role": "polar", "candidates": 1, "via": "polar"}

    def _polar(self, toks):
        """POLÁRNÍ (ano/ne) — uzavřená smyčka VZORŮ: tvar dotazu (přísudek+podmět+objekt,
        modalita ?, BEZ tázacího slova) → EXISTENCE fakt-tvaru (predikát+role) → ano/ne.

        Kopulová „Je/Byl X Y?": Y ∈ stavech X → ano; X stavy má, Y mezi nimi ne → ne; jinak
        nevíme (None → nehádá). Predikátová „Napsal/Měl X Z?": fakt predikát(X,Z) existuje →
        ano; nenašli → None (nehádá, ne „ne" z absence). Vstup: tokeny. Výstup: dict / None.
        """
        if not self.facts_enabled or self.g.sentence_modality(toks) != "?":
            return None
        if any("Int" in (t.get("feats") or {}).get("PronType", "") for t in toks):
            return None                                      # tázací zájmeno → wh-otázka
        if any((t["lemma"] or "").lower() in self.interrog for t in toks):
            return None                                      # kolik/jaký/… → wh-otázka
        subj = [self.g.canon_lemma(t).lower() for t in toks
                if t.get("deprel") in ("nsubj", "nsubj:pass")]
        if not subj:
            return None
        hot = [d for d, _s in self.dl.select_files(subj)]    # doména podmětu
        if not hot:
            return None
        self.facts.mount(hot)
        root = next((t for t in toks if t.get("deprel") == "root"), None)
        has_cop = any(t.get("deprel") == "cop" for t in toks)
        if has_cop and root is not None and root["upos"] in ("NOUN", "ADJ", "PROPN"):
            obj = self.g.canon_lemma(root).lower()           # komplement „Je X <obj>?"
            states = self._copula_states(subj)
            if not states:
                return None                                  # o subjektu nic → nehádej
            hit = any(obj == s or obj in s or s in obj for s in states)
            return self._polar_result("ano" if hit else "ne")
        objs = [self.g.canon_lemma(t).lower() for t in toks   # predikátová polární
                if t.get("deprel") in ("obj", "obl", "obl:arg", "xcomp", "iobj")]
        if objs:
            f = self._fact_cooccur(subj, objs)               # podmět+objekt v jednom faktu → ano
            if f is not None:
                return self._polar_result("ano", [f.doc, f.sent])
        return None                                          # nenašli → nehádej (ne „ne" z absence)

    def _gate_feats(self, best, cands, known_lemmas, hole_role):
        """Rysy VÍTĚZE pro NAUČENOU ASSURANCE bránu — má rozlišit glow-správně (Švejk→Hašek)
        od confident-wrong (bratr→Eliáš). Base signály + glow + odstup od rivala + DOMÉNA
        (je vítězův soubor horký, nebo cizí?) + počet kandidátů. Vstup: vítěz, kandidáti,
        lemmata otázky, role díry. Výstup: rys-vektor (list).
        """
        known = {l.lower() for l in known_lemmas}
        ans = (best["answer"] or "").lower()
        echo = 0 if ans in known else 1
        rf = 1 if hole_role and best["role"] == hole_role else 0
        ko = 1 if self._known_overlap(best.get("fact_ref"), best["role"], known) else 0
        df = 1 if best["role"] == "state" and (best.get("fact_ref") or [0, 1])[1] == 0 else 0
        tf = 1 if hole_role == "where" and self.topos.is_place(best["answer"]) else 0
        kp = 1 if best["kind"] == "pred" else 0
        gb = self.field.weight_answer(best["answer"], best["doc"])
        rivals = [self.field.weight_answer(c["answer"], c["doc"]) for c in cands
                  if (c["answer"] or "").lower() != ans]
        margin = gb / ((max(rivals) if rivals else 0.0) + 1e-6)      # dominance nad 2. odpovědí
        n_ans = len({(c["answer"] or "").lower() for c in cands})
        fmax = max(self.field.files.values(), default=1.0) or 1.0
        doc_act = self.field.files.get(best["doc"], 0.0) / fmax      # je vítězův soubor horký?
        # RICHER: je odpověď JMÉNO (PROPN-like)? její idf (distinktivita)? je vítězův soubor
        # DOMOVSKÝ soubor entity otázky (Hašek z vlastního článku vs Eliáš z cizí bible)?
        is_name = int(bool(best["answer"]) and best["answer"][:1].isupper())
        ans_idf = self.field.idf.get(ans, 0.0)
        subs = self.dl._load_subjects()
        ents = [l for l in known if self.field.idf.get(l, 0.0) >= 1.5]
        home = None
        if ents:
            ent = max(ents, key=lambda l: self.field.idf.get(l, 0.0))
            home = max(subs, key=lambda d: subs.get(d, {}).get(ent, 0), default=None)
            if home is not None and subs.get(home, {}).get(ent, 0) == 0:
                home = None
        home_match = int(home is not None and best["doc"] == home)
        return [echo, rf, ko, df, tf, kp, round(gb, 4), round(margin, 4), n_ans,
                round(doc_act, 4), is_name, round(ans_idf, 3), home_match]

    def answer(self, question, carry_context=False, hole_override=None, return_features=False):
        """Živá otázka → odpověď / klarifikace / upřímný terminál (dialogový stavový automat).

        `carry_context=True` = interaktivní session: světlo i mount PŘETRVAJÍ mezi tahy (jen
        `decay` je utlumí), navazující otázka bez entity si vezme TÉMA z horkých entit minula
        („Kdy se narodil?" po „Kdo je Karel Čapek?" → Karel je horký → narodit-fakt Karla).
        Default False = samostatný tah (volající si pole/mount čistí sám, např. etalon).
        `hole_override` = vnucená role díry (měření routing-stropu / naučený router místo
        ručního _answer_role) — obejde odvození role z tvaru.

        Výstup: dict {answer, mode, offer, fact_ref, hole_role, candidates, via} nebo None.
        """
        toks = self._parse(question)
        q_vzor, lemmas, hole_role = self._question(toks)
        hole_role = self._answer_role(toks, hole_role)          # kopula → komplement (state)
        if hole_override is not None:                           # router / oracle vnutí roli
            hole_role = hole_override
        predicate = self._predicate(toks)
        rel = self._relation_query(toks)                        # „Kdo byl bratr X?" → vztah-fakt
        if rel is not None:
            predicate, hole_role = rel, "who"
        if q_vzor is None and hole_role is None:            # bez tázacího slova → možná POLÁRNÍ
            pol = self._polar(toks)
            if pol is not None:
                return pol
        if not (q_vzor or (predicate and hole_role)):
            return None
        known = {l.lower() for l in lemmas}
        follow_up = False
        if carry_context:
            # NAVAZUJÍCÍ otázka BEZ vlastní entity si vezme TÉMA z minula, otázka s VLASTNÍ
            # entitou téma přepíná. Predikát z testu VYNECHÁN — „narodit/zemřít" mají vysoké
            # idf, ale jsou to slovesa, ne téma; rozhoduje distinktivní NEslovesné slovo.
            pred_l = (predicate or "").lower()
            follow_up = not any(self.field.idf.get(l.lower(), 0.0) >= 1.5
                                for l in lemmas if l.lower() != pred_l)
            if follow_up:
                self.field.decay()                               # minulé téma jen POHASNE (drží kontext)
                known |= self._hot_entities()                    # + horké entity z minula (téma)
            else:
                # NOVÉ téma (otázka nese vlastní entitu): zbytkový žár minulého protagonisty
                # zhasne, jinak by v glow-remíze („napsal"+who nese ko=0 u všech autorů) přebil
                # čerstvou entitu — „Kdo napsal Švejka?" po Čapkovi nesmí dát Čapka místo Haška.
                # A ZAHOĎ i mount minulého tématu — nakupené dokumenty by přinášely rivaly a
                # kazily assurance (answer→clarify), i když identitu vítěze nemění.
                self.field.words.clear()
                self.field.files.clear()
                self.store.mounted.clear()
                if self.facts_enabled:
                    self.facts.mounted.clear()
        routed = []
        if self.facts_enabled and self.entity_retrieval.get("enabled", False):
            routed = self.facts.route_docs(
                predicate, hole_role, known,
                max_docs=self.entity_retrieval.get("max_docs", 4),
                max_fact_refs=self.entity_retrieval.get("max_fact_refs", 24),
            )
        # Přímý faktový důkaz má přednost před lexikálním mountem. Bez něj je
        # chování identické s původním selective loaderem.
        hot = routed or [d for d, _s in self.dl.select_files(lemmas)]
        if not hot and not (carry_context and self.store.mounted):
            return None
        self.store.mount(hot)                                    # NAHRAJ šablony horkých (#60)
        if self.facts_enabled:
            self.facts.mount(hot)                                # NAHRAJ fakty horkých (#60)
        self.field.adj.clear()                                   # graf hran = VŠECHNY mountnuté
        self.field.build_graph(self.dl.mount(list(self.store.mounted)), self.g)
        self.field.feed(lemmas, self.dl)                         # rozsvícení
        self.field.spread()                                      # teplo po hranách
        if follow_up:
            known |= self._hot_entities()                        # po spreadu přibydou horké
        cands = self._candidates(q_vzor, predicate, hole_role, known)   # predikát+role ∪ window-VZOR
        if not cands:
            return None
        mode, best, offer = self._assurance(cands, lemmas, hole_role)   # jasno vs nejasno
        if best is None:
            return None
        gate_p = None
        if mode == "answer" and self.gate is not None:                  # NAUČENÁ brána: nehádej,
            gate_p = self._gate_p(self._gate_feats(best, cands, lemmas, hole_role))
            if gate_p < self.gate_threshold:                            # nejspíš špatně → unsure
                mode = "unsure"
        if mode == "answer":
            self.field.reinforce(best["answer"], best["doc"])    # zpětný tok jen u jasné odpovědi
            if carry_context:                                    # udrž téma i entity otázky horké
                for e in known:
                    self.field.words[e] = self.field.words.get(e, 0.0) + 1.0
        result = {"answer": best["answer"], "mode": mode,
                  "offer": [c["answer"] for c in offer], "fact_ref": best.get("fact_ref"),
                  "hole_role": hole_role, "candidates": len(cands), "via": best["kind"]}
        if return_features:                                      # telemetrie pro naučenou bránu
            result["features"] = self._gate_feats(best, cands, lemmas, hole_role)
        return result


def _demo():
    """Odpoví na pár otázek nad JIŽ POSTAVENÝMI story (nedestruktivní — bez reset()).

    Předpoklad: `build_facts_all.py` + `build_templates_all.py` proběhly. Čistý per-tah tah.
    """
    a = Answering()
    if not a.facts.document_ids() and not a.store.document_ids():
        print("Nejdřív postav story:  python3 build_facts_all.py && python3 build_templates_all.py")
        return
    for q in ["Kdo napsal R.U.R.?", "Co napsal Karel Čapek?",
              "Kde se narodil Karel Čapek?", "Kdo je pes domácí?"]:
        r = a.answer(q)
        a.store.mounted.clear()
        a.facts.mounted.clear()
        a.field.words.clear()
        a.field.files.clear()
        a.field.adj.clear()
        print(f"  Q: {q}\n     → {r['answer'] if r else '—'}   "
              f"(via {r['via'] if r else '—'}, fakt {r['fact_ref'] if r else '—'}, "
              f"kandidátů {r['candidates'] if r else 0})")


if __name__ == "__main__":
    _demo()
