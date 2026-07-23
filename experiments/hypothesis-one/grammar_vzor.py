#!/usr/bin/env python3
"""③ GRAMMAR — VZOR primitiva (SLOT / frame_sig / kanonizace / role_key).

Jazykové jádro systému: z tokenu (WORD_W_ATTR) a jeho okolí staví SLOT a VZOR
(frame_sig), kanonizuje lemma a mapuje surový UDPipe `deprel` na NÁŠ klíč rolového
katalogu. Pracuje nad tokeny z anotovaného korpusu (vrstva ① / `dataloader.mount`) —
je nezávislé na tom, odkud korpus přišel.

VZOR je přesný na gramatiku (pád/slovní druh/čas), ale abstraktní na lexém; pokrytí
se získává KVANTITOU přesných vzorů, ne rozmazáním jednoho (mluvnický vzor „pán").
Pád v pivotu JE role, proto ve VZORu vždy musí být.

Parametry VZORu (poloměr okna, interpunkce, modalita) i jazyková data jsou v JSON
(`config.json`, `lang/cs.json`) — nic natvrdo v kódu.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")


class GrammarVzor:
    """VZOR primitiva a mapování surové anotace na náš rolový klíč.

    Instance drží parametry VZORu (poloměr okna, ponechaná interpunkce, značky
    modality) a jazyková data (`cs.json`). Metody jsou čisté transformace nad
    tokeny/větami bez vedlejších efektů, takže se dají volat kdekoli v pipeline.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte parametry VZORu z `config.json` a jazyková data z `paths.lang`.

        Nic není natvrdo: jiný jazyk/nasazení = jiný JSON, žádný zásah do kódu.
        """
        cfg = json.load(open(config_path, encoding="utf-8"))
        self.radius = cfg["radius"]
        self.modality_marks = cfg["modality_marks"]
        self.punct_keep = set(cfg["punct_keep"])
        self.LANG = json.load(open(os.path.join(HERE, cfg["paths"]["lang"]), encoding="utf-8"))
        self._vow = self.LANG["vowels"]
        self._uncertain = set(self.LANG.get("uncertainty_markers", []))
        self._hedge_root = set(self.LANG.get("hedge_root_predicates", []))

    def slot(self, tok):
        """SLOT — match-obálka jednoho tokenu: UPOS + nosné rysy (pád, čas).

        Přesný na gramatiku, abstraktní na lexém. Pád v pivotu JE role (Kdo=Nom /
        Komu=Dat / Co=Acc), proto se do slotu vždy bere.

        Vstup: WORD_W_ATTR (dict). Výstup: SLOT (str), např. `"PROPN:Nom"`.
        """
        up = tok["upos"]
        if up == "PUNCT":
            return tok["form"] if tok["form"] in self.punct_keep else "PUNCT"
        feats = tok.get("feats") or {}
        vals = [feats[k] for k in ("Case", "Tense") if k in feats]
        return up + (":" + ":".join(vals) if vals else "")

    def sentence_modality(self, toks):
        """Koncová modalita věty (. ? ! :) z poslední ponechané interpunkce.

        Když věta žádnou nemá, vrací tečku (oznamovací). Vstup: věta (list tokenů).
        """
        for t in reversed(toks):
            if t["upos"] == "PUNCT" and t["form"] in self.modality_marks:
                return t["form"]
        return "."

    def frame_sig(self, toks, i, modality, r=None):
        """VZOR (SLOT_ARRAY) — pole SLOTŮ kolem pivotu: r vlevo/vpravo + pivot + modalita.

        `r` řídí velikost okna (výchozí z configu); okraje věty jsou `^` / `$`.
        Pokrytí = mnoho přesných vzorů, ne rozostření jednoho.

        Vstup: `toks` (věta), `i` (index pivotu), `modality` (str). Výstup: VZOR (str).
        """
        r = self.radius if r is None else r
        left = [self.slot(toks[i - k]) if i - k >= 0 else "^" for k in range(r, 0, -1)]
        right = [self.slot(toks[i + k]) if i + k < len(toks) else "$" for k in range(1, r + 1)]
        return "·".join(left + [self.slot(toks[i])] + right + [modality])

    def _epen_stem(self, stem):
        """Vloží epentetické -e- do koncového shluku souhlásek (Čapk→Čapek).

        Jen mezi písmeny (ne u zkratek). Pomocná funkce kanonizace.
        """
        if len(stem) >= 3 and stem[-1].isalpha() and stem[-2].isalpha() \
                and stem[-1].lower() not in self._vow and stem[-2].lower() not in self._vow:
            return stem[:-1] + "e" + stem[-1]
        return stem

    def canon_lemma(self, tok):
        """Základní tvar; přivlastňovací ADJ (Poss=Yes) → base jméno.

        Karlova→Karel, Čapkův→Čapek — uniformně, bez výjimek („Univerzita Karlova" →
        Karel + univerzita). Ostatní lemma se nechává (fold jmen dělá korpusová
        evidence jinde, ne bezpodmínečně tady).

        Vstup: WORD_W_ATTR (dict). Výstup: lemma (str).
        """
        lemma = tok["lemma"]
        if tok["upos"] == "ADJ" and (tok.get("feats") or {}).get("Poss") == "Yes":
            for suf in self.LANG["possessive_adj_suffixes"]:
                if lemma.endswith(suf) and len(lemma) > len(suf) + 1:
                    return self._epen_stem(lemma[:-len(suf)])
        return lemma

    def prep_of(self, sent, tid):
        """Předložka (case-marker) závislá na tokenu s 1-based id `tid`, jinak `""`.

        Slouží `role_key` — předložka může roli přebít (o kom, s kým).
        """
        for t in sent:
            if t.get("head") == tid and t.get("deprel") == "case":
                return t["lemma"]
        return ""

    def role_key(self, deprel, upos, feats, prep="", nominal_pred=False):
        """Surový UDPipe `deprel` (+ pád/životnost/předložka) → NÁŠ klíč katalogu rolí.

        Tady se surová anotace normalizuje na naši konvenci (who/where/whom_what/…).
        Jediný zdroj mapování; mapa i předložky jsou data v `cs.json`
        (`deprel_to_role`, `role_prepositions`). Vrací `None`, když deprel nemapujeme.

        Příklad: `role_key("obl", "NOUN", {"Case":"Loc"})` → `"where"`.
        """
        if nominal_pred:                          # jmenný přísudek se sponou → stav
            return "state"
        if prep:                                  # předložka může roli přebít
            for key, preps in self.LANG.get("role_prepositions", {}).items():
                if prep in preps:
                    return key
        table = self.LANG["deprel_to_role"].get(deprel)
        if not table:
            return None
        if "anim" in table:                       # subjekt rozlišuje životnost
            anim = feats.get("Animacy") == "Anim" or bool(feats.get("NameType")) or upos == "PROPN"
            return table["anim"] if anim else table["inanim"]
        return table.get(feats.get("Case", ""), table["default"])

    def is_factual(self, tokens):
        """True, když věta NESE fakt — ne hedging/nejistota/nezávaznost.

        Odmítne větu s markerem nejistoty (možná/snad/prý…), s kondicionálem hlavního
        přísudku (hypotéza „mohlo by") nebo otázku. Faktualitní brána syntézy: taková věta
        se lingvisticky zanotuje, ale fakt/query se z ní netvoří.

        Vstup: věta (list tokenů). Výstup: bool.
        Příklad: `is_factual(parse("Tak nějak nevím."))` → False.
        """
        # KONCOVÁ otázka = není fakt. Ale stray „?" UVNITŘ věty (závorková nejistota
        # „Vídeň ?" v definici) NESMÍ zahodit celou oznamovací větu — jinak se ztratí
        # úvodní kopula („byla česká spisovatelka"). Sladěno se sentence_modality.
        if self.sentence_modality(tokens) == "?":
            return False
        for t in tokens:
            f = t.get("feats") or {}
            lem = t["lemma"].lower()
            if lem in self._uncertain:                       # možná/snad/prý/nějak…
                return False
            if t["deprel"] == "root":
                if f.get("Mood") == "Cnd":                   # kondicionál = hypotéza
                    return False
                if lem in self._hedge_root:                  # modální možnost „může to být"
                    return False
            if lem == "vědět" and f.get("Polarity") == "Neg" and f.get("Person") == "1":
                return False                                 # „nevím" = nejistota mluvčího
        return True


if __name__ == "__main__":
    import pickle
    g = GrammarVzor()
    shard = pickle.load(open(os.path.join(HERE, "../../data/corpus/wiki_r.u.r..pkl"), "rb"))
    sent = next(iter(shard.values()))["sentences"][0]
    mod = g.sentence_modality(sent)
    print("věta:", " ".join(t["form"] for t in sent)[:70])
    for i, t in enumerate(sent[:6]):
        print(f"  {t['form']:14} slot={g.slot(t):12} vzor={g.frame_sig(sent, i, mod)}")
