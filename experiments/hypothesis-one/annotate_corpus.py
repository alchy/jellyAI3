#!/usr/bin/env python3
"""① PREPROCESSING — surový text → anotovaný korpus (UDPipe 2).

Vrstva PREPROCESSING má JEDEN vstup (raw text) a JEDEN výstup (anotovaný korpus),
na kterém stojí všechny další vrstvy (② dataloader, ③ grammar, ④ synthesis, …).
Anotaci děláme JEDNOU offline přes UDPipe 2 (LINDAT REST, plná pipeline:
tokenizace + segmentace vět + morfologie + závislostní rozbor) a uložíme; runtime
pak korpus jen čte — žádný neuron v rozhodovací smyčce (teze drží).

Výstup je SHARD PER SOUBOR (koncept #60 fragmentový graf): každý zdrojový soubor
má vlastní pickle `corpus_dir/<doc>.pkl`, aby se do grafu daly nahrávat soubory
DYNAMICKY PODLE MATCHE (frekvence × unikátnost = tf·idf), ne všechny naráz —
jeden monolitický pickle by lazy-mount znemožnil. Jeden shard:
    { (doc_id, index_věty): {"entities": [...], "sentences": [[token, …], …]} }
  token  = {form, lemma, upos, feats(dict), head(int, 0=root), deprel, start, end}
  entity = {text, type, start, end},  type ∈ {"P" osoba, "G" místo}
Index tf·idf per soubor a mount/unmount podle matche staví vrstva ② (dataloader).

ENTITY se NEberou z NameTagu (zdroj slepenců „Ježíš Duchem"), ale ODVOZUJÍ se
z UDPipe 2 rysu `NameType` (Giv/Sur → osoba, Geo → místo). Osobní span drží jen
souvislé tokeny SHODNÉHO PÁDU → pádová neshoda span neslepí (řez zabudován rovnou).

Data se posílají PŘES SOUBOR (`-F data=@file`), NE inline: inline `-F data=…`
ořezává velký/víceřádkový vstup na ~485 znaků (past, kvůli které bible ztrácela
95 % textu). Model se PINUJE (config) a skutečně použitý se zaznamená do meta.
"""
import os
import re
import sys
import json
import glob
import pickle
import tempfile
import subprocess
import time

from logger import logger

HERE = os.path.dirname(os.path.abspath(__file__))
# REUSE fragmentu z parentu: univerzální oprava tokenizace tečkovaných zkratek
# („R.U.R." → 1 PROPN, ne R/./U/./R/.). Parent = studnice nápadů; tohle přesahuje k nám.
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from jellyai.normalize import merge_abbreviations
CONFIG_PATH = os.path.join(HERE, "config.json")

PERSON_NAMETYPE = {"Giv", "Sur"}          # UDPipe 2 NameType → osoba ("P")
GEO_NAMETYPE = {"Geo"}                     # UDPipe 2 NameType → místo ("G")


class AnnotateCorpus:
    """Anotátor korpusu: raw texty z `raw_dir` → jeden `out_corpus` pickle.

    Instance drží konfiguraci vrstvy PREPROCESSING (URL a pinnutý model UDPipe 2,
    cesty, velikost dávky). Čisté transformace (parsování CoNLL-U, odvození entit)
    jsou statické metody bez stavu; síťové a I/O kroky jsou instanční.
    """

    def __init__(self, config_path=CONFIG_PATH):
        """Načte blok `preprocessing` z JSON konfigurace (nic natvrdo v kódu).

        Args:
            config_path (str): cesta ke `config.json` s klíčem "preprocessing".
        """
        self.cfg = json.load(open(config_path, encoding="utf-8"))["preprocessing"]

    # ---- čisté transformace (bez stavu) -------------------------------------

    @staticmethod
    def _token_range(misc):
        """Z MISC sloupce CoNLL-U vytáhne znakové offsety `TokenRange=start:end`.

        Vrací (start, end) jako int, nebo (None, None) když sloupec offset nenese.
        """
        for kv in misc.split("|"):
            if kv.startswith("TokenRange="):
                a, b = kv[len("TokenRange="):].split(":")
                return int(a), int(b)
        return None, None

    @staticmethod
    def parse_conllu(text):
        """CoNLL-U text → (list[list[token]], model_name).

        Věty se dělí prázdným řádkem; víceslovné a prázdné uzly se přeskakují.
        Každý token dostane i znakové offsety z `TokenRange`.

        Args:
            text (str): CoNLL-U výstup UDPipe 2.
        Returns:
            (list[list[dict]], str|None): věty jako seznamy tokenů + jméno modelu.

        Příklad:
            >>> sents, model = AnnotateCorpus.parse_conllu(conllu_str)
            >>> sents[0][0]["lemma"], sents[0][0]["start"]
            ('být', 0)
        """
        sents, cur, model = [], [], None
        for ln in text.splitlines():
            if ln.startswith("# udpipe_model ="):
                model = ln.split("=", 1)[1].strip()
            if not ln.strip():
                if cur:
                    sents.append(cur); cur = []
                continue
            if ln[0] == "#":
                continue
            c = ln.split("\t")
            if len(c) < 10 or "-" in c[0] or "." in c[0]:
                continue
            feats = {} if c[5] == "_" else dict(kv.split("=", 1) for kv in c[5].split("|"))
            start, end = AnnotateCorpus._token_range(c[9])
            cur.append({"form": c[1], "lemma": c[2], "upos": c[3], "feats": feats,
                        "head": int(c[6]), "deprel": c[7], "start": start, "end": end})
        if cur:
            sents.append(cur)
        return sents, model

    @staticmethod
    def entities_from_nametype(tokens):
        """Odvodí entity z rysu `NameType` jedné věty (osoba "P" / místo "G").

        Osobní span drží jen souvislé tokeny SHODNÉHO PÁDU, takže pádová neshoda
        („Ježíš"Nom + „Duchem"Ins) span neslepí — řez proti slepencům je zabudován.

        Args:
            tokens (list[dict]): tokeny jedné věty (z `parse_conllu`).
        Returns:
            list[dict]: entity {text, type, start, end}.

        Příklad:
            >>> AnnotateCorpus.entities_from_nametype(veta_tokeny)
            [{'text': 'Karla Čapka', 'type': 'P', 'start': 112, 'end': 123}]
        """
        def cat(tok):
            nt = tok["feats"].get("NameType")
            if nt in PERSON_NAMETYPE:
                return "P"
            if nt in GEO_NAMETYPE:
                return "G"
            return None

        ents, i, n = [], 0, len(tokens)
        while i < n:
            c = cat(tokens[i])
            if c is None:
                i += 1
                continue
            j, case0 = i, tokens[i]["feats"].get("Case")
            while j + 1 < n and cat(tokens[j + 1]) == c and \
                    (c != "P" or tokens[j + 1]["feats"].get("Case") == case0):
                j += 1
            span = tokens[i:j + 1]
            ents.append({"text": " ".join(t["form"] for t in span), "type": c,
                         "start": span[0]["start"], "end": span[-1]["end"]})
            i = j + 1
        return ents

    @staticmethod
    def _chunks(text, limit):
        """Rozdělí text na dávky ≤ `limit` znaků na hranicích ODSTAVCŮ.

        Dělí na prázdném řádku, aby se věta nikdy nerozťala; odstavec delší než
        limit jde v jedné dávce sám.
        """
        buf, out = "", []
        for p in re.split(r"\n\s*\n", text):
            p = p.strip()
            if not p:
                continue
            if buf and len(buf) + len(p) + 2 > limit:
                out.append(buf); buf = p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            out.append(buf)
        return out

    # ---- síť + I/O (instanční) ----------------------------------------------

    def _udpipe2(self, text, tries=4):
        """Plná pipeline UDPipe 2 na `text` → CoNLL-U (str) nebo None.

        Data posílá PŘES DOČASNÝ SOUBOR (`-F data=@file`), protože inline `-F data=`
        ořezává velký/víceřádkový vstup na ~485 znaků. Retry se zvětšující se pauzou.

        Args:
            text (str): raw text jedné dávky.
            tries (int): počet pokusů při síťovém selhání.
        Returns:
            str|None: CoNLL-U výsledek, nebo None po vyčerpání pokusů.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as f:
            f.write(text)
            tmp = f.name
        try:
            for a in range(tries):
                r = subprocess.run(
                    ["curl", "-s", "-m", "180",
                     "-F", f"data=@{tmp};type=text/plain",
                     "-F", f"model={self.cfg['udpipe2_model']}",
                     "-F", "tokenizer=ranges", "-F", "tagger=", "-F", "parser=",
                     self.cfg["udpipe2_url"]],
                    capture_output=True, text=True)
                try:
                    return json.loads(r.stdout)["result"]
                except Exception:
                    time.sleep(2 + 2 * a)
            return None
        finally:
            os.unlink(tmp)

    def annotate_document(self, doc_id, text):
        """Anotuje jeden dokument → ({(doc_id, index_věty): rec}, model).

        Věty se číslují průběžně napříč dávkami; selhaná dávka se zaloguje jako
        error a přeskočí, aby zbytek doběhl.

        Args:
            doc_id (str): identifikátor dokumentu (jméno souboru bez přípony).
            text (str): surový text dokumentu.
        Returns:
            (dict, str|None): záznamy korpusu pro dokument + skutečně použitý model.

        Příklad:
            >>> recs, model = AnnotateCorpus().annotate_document("wiki_r.u.r.", raw)
            >>> len(recs)
            49
        """
        recs, si, model = {}, 0, None
        for chunk in self._chunks(text, self.cfg["batch_chars"]):
            result = self._udpipe2(chunk)
            if result is None:
                logger("!", f"{doc_id}: dávka selhala (síť), přeskočeno {len(chunk)} znaků")
                continue
            sents, m = self.parse_conllu(result)
            sents = merge_abbreviations(sents)     # oprava tokenizace zkratek (reuse z parentu)
            model = m or model
            for sent in sents:
                recs[(doc_id, si)] = {"entities": self.entities_from_nametype(sent),
                                      "sentences": [sent]}
                si += 1
        return recs, model

    def run(self):
        """Zanotuje každý soubor z `raw_dir` do SAMOSTATNÉHO shardu `<doc>.pkl`.

        Každý zdrojový soubor = jedna mountovatelná fragmentová jednotka (koncept
        #60): soubory se do grafu nahrávají dynamicky podle matche, ne všechny
        naráz. Vedle shardů zapíše `corpus_dir/_meta.json` s pinnutým i skutečně
        použitým modelem (reprodukovatelnost). Vstupní data se nikdy nepřepisují.
        """
        raw_dir = os.path.join(HERE, self.cfg["raw_dir"])
        corpus_dir = os.path.join(HERE, self.cfg["corpus_dir"])
        os.makedirs(corpus_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
        logger("i", f"anotace: dokumentů {len(files)}, model(pin) {self.cfg['udpipe2_model']}")

        model_used, per_doc, total, skipped = None, {}, 0, 0
        for path in files:
            doc_id = os.path.splitext(os.path.basename(path))[0]
            shard = os.path.join(corpus_dir, f"{doc_id}.pkl")
            # PŘÍRŮSTKOVĚ: shard aktuální (novější než raw) → přeskoč (levné add-content)
            if os.path.exists(shard) and os.path.getmtime(shard) >= os.path.getmtime(path):
                per_doc[doc_id] = len(pickle.load(open(shard, "rb")))
                total += per_doc[doc_id]
                skipped += 1
                continue
            text = open(path, encoding="utf-8").read()
            recs, m = self.annotate_document(doc_id, text)
            model_used = m or model_used
            pickle.dump(recs, open(shard, "wb"))
            per_doc[doc_id] = len(recs)
            total += len(recs)
            n_ent = sum(len(r["entities"]) for r in recs.values())
            logger("i", f"  {doc_id}: vět {len(recs)}, entit {n_ent} → {doc_id}.pkl")
        if skipped:
            logger("i", f"  přeskočeno {skipped} aktuálních shardů (přírůstkově)")

        meta = {"model": model_used, "model_pinned": self.cfg["udpipe2_model"],
                "documents": len(files), "sentences": total, "per_doc": per_doc,
                "raw_dir": self.cfg["raw_dir"]}
        json.dump(meta, open(os.path.join(corpus_dir, "_meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        logger("i", f"hotovo → {corpus_dir}/<doc>.pkl (vět {total}, dokumentů {len(files)})")


if __name__ == "__main__":
    AnnotateCorpus().run()
