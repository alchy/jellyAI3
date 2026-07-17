# Jak psát fakta pro jelly — doporučení pro (lokální) generující model

Návod, jak formulovat český text, aby ho extrakční pipeline jellyAI3
(UDPipe + NameTag + faktová extrakce + korpusová hygiena) přečetla co
nejlépe. Text je použitelný přímo jako SYSTEM PROMPT generujícího modelu.
Výstup modelu se ukládá jako `.txt` do `data/raw/` a zpracuje se
`./jelly index && ./jelly annotate && ./jelly graph`.

## Principy (proč to funguje)

Jelly staví **reifikovaný faktový graf**: každá slovesná věta = jeden
fakt (predikát + účastníci v rolích podmět/předmět/čas/místo/počet).
Parser je deterministický, ne jazykový model — nejlépe čte **krátké
oznamovací věty s jasným podmětem**. Hygiena je korpusová: tvary a značky
se ověřují HLASOVÁNÍM přes celý korpus, takže ojedinělý výskyt jména se
nedá obhájit — opakování pomáhá.

## Pravidla (DO)

1. **Jedna věta = jeden fakt.** Podmět → sloveso → předmět/okolnosti.
   > ✅ „Karel Čapek napsal drama R.U.R. v roce 1920."
   > ❌ „R.U.R., které, jak známo, sepsal r. 1920 Čapek, jenž…"

2. **Slovosled podmět první (SVO).** Volný slovosled je hlavní zdroj
   slepenců — obrácená věta „Ježíš Martu miloval" vyrobila falešnou
   osobu „Ježíš Martu". Piš „Ježíš miloval Martu".

3. **Plné jméno, vždy stejné.** První i další zmínky týmž tvarem
   („Karel Čapek", ne střídavě „Čapek"/„K. Čapek"/„spisovatel").
   Pádové varianty (Čapka, Čapkovi) kanonizace sloučí; RŮZNÁ jména
   téže osoby ale graf rozdělí na střepy.

4. **Zámény nahrazuj jménem.** Pro-drop („Narodil se…") se přiřadí
   naposledy zmíněné osobě odstavce — funguje, ale explicitní jméno je
   vždy spolehlivější. Zájmena (on, jeho, ta) minimalizuj.

5. **Každé jméno aspoň jednou v nominativu** (lépe 2–3×). Hlasování
   o pádu, životnosti i nominativizaci id potřebuje ≥2 shodné výskyty —
   jméno zmíněné jen jednou v genitivu zůstane nesouzené.
   > ✅ „Hronov leží ve východních Čechách. Josef Čapek se narodil v Hronově."

6. **Identitu piš sponou, druhové zařazení apozicí.**
   > „Karel Čapek byl český spisovatel." → být(KČ, spisovatel)
   > „drama R.U.R." / „R.U.R. – drama" → druh(R.U.R., drama)
   Každé entitě dej jednu čistou definiční větu „X je/byl Y".

7. **Aliasy tvrzením „řečený/zvaný".** Jediný způsob, jak jelly sloučí
   dvě jména téže entity:
   > „Šimon, zvaný Petr, byl rybář." → jmenovat(Šimon, Petr) → srůst.

8. **Vztahy holým genitivem.**
   > „Josef Čapek byl bratr Karla Čapka." → bratr(Josef, Karel)
   Genitiv bez předložky = vztah; s předložkou („o Karlu Čapkovi") je
   to jen téma, vztah nevznikne — to je záměr.

9. **Datum absolutně, formát „9. ledna 1890".** Vznikne časový uzel
   s pod-fakty rok/měsíc/den (drill „v kterém roce…"). Relativní časy
   („loni", „nedávno") do korpusu nepatří — patří jen do dialogu.

10. **Místo předložkovou vazbou u slovesa:** „narodil se v Hronově",
    „působila v Praze". Počty číslovkou i slovem: „měla čtyři děti".

11. **Řeč uvozovkami za dvojtečkou:** „Ježíš řekl: „Pojďte za mnou.""
    → fakt říci s obsahem výroku (odpovídá na „Co řekl X?").
    Uvozovky odděluj mezerou od slov okolo (přilepené se čistí, ale
    zbytečně).

12. **Odstavec = jedno téma (jedna osoba).** Aktivační koreference
    přiřazuje bezpodmětné věty nejžhavější osobě odstavce — míchání
    osob v odstavci fakta rozhází. Soubor = jedna doména; jméno souboru
    smysluplné s prefixem (`wiki_karel_čapek.txt`, `bible_jan.txt`) —
    rodiny souborů řídí doménové zaostření („v kontextu Bible").

## Čemu se vyhnout (DON'T)

- **Dvě jména vedle sebe bez slovesa** („…řekl Áronovi Mojžíš.") —
  NER je slepí v jednu osobu. Vždy: „Mojžíš řekl Áronovi."
- **Negace, kde stačí pozitivní fakt** — „ne-" tvoří jiný predikát
  a párování zatím není (backlog #24). Místo „Nebyl ženatý" piš
  „Zůstal svobodný".
- **Vsuvky, závorky, výčty bez sloves** — bezslovesná řádka vyrobí jen
  slabou kontextovou asociaci, ne fakt. (Výjimka: pomlčková definice
  „Titul – druh" funguje.)
- **Tabulky, odrážky, markdown** — vstup je prostý text po větách.
- **Neobvyklé tečkované zkratky** — R.U.R. je normalizované, jiné
  mohou rozbít větné dělení. Piš slovy, co jde.
- **Řetězení přívlastků z jmen** („vztahu Dorothey von Biron s hrabětem
  Karlem Janem Clam-Martinicem") — dlouhé jmenné řetězy jsou nejčastější
  zdroj vadných extrakcí. Rozděl do více vět.

## Vzorový odstavec (šablona výstupu)

> Božena Němcová byla česká spisovatelka. Božena Němcová se narodila
> 4. února 1820 ve Vídni. Božena Němcová napsala povídku Babička
> v roce 1855. Babička je próza. Josef Němec byl manžel Boženy Němcové.
> Božena Němcová měla čtyři děti. Božena Němcová zemřela 21. ledna 1862
> v Praze.

Zní to školsky — a přesně tak to má být: každá věta je jeden čistý,
samostatně ověřitelný fakt. Stylistickou eleganci jelly nečte;
kompozici odpovědí řeší jiná vrstva.

## Ověření po nasypání (checklist)

```bash
./jelly index && ./jelly annotate && ./jelly graph
# sleduj výpis buildu: hygiena/kanonizace nesmí hlásit nezvyklé počty
.venv/bin/python benchmark/run_coverage.py     # podíl vět bez faktu
./jelly graph-ask "kontrolní otázka na nová fakta?"
```

Když kontrolní otázka neodpovídá: nejčastější příčina je nejednotné
jméno (pravidlo 3), zájmeno místo jména (4), nebo obrácený slovosled (2).
