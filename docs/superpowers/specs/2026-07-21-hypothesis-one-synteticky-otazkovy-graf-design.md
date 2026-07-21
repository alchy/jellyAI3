# hypothesis-one — syntetický otázkový graf s aktivačním výběrem odpovědi

*Návrhový dokument. Větev `hypothesis-one`. Datum 2026-07-21. Experiment žije
v `experiments/hypothesis-one/run.py`; změny vizualizéru v `/Users/j/Projects/viewBase`.*

## 1. Motivace

Kartový jazykový model enumeruje povrchové varianty otázek a dlouhodobě roste do
neudržitelné komplexity. Východisko obrací směr práce: místo ručního psaní *jak se
lze zeptat* se otázky **odvozují z textu**, protože **otázka je stín tvrzení/faktu**
— vše, na co se lze ptát, je obsaženo v textu, který systém zná.

## 2. Mentální model (celá smyčka)

```
1. TVRZENÍ  → index:  (Occurrence lemma+rysy)  +  (Frame left·upos·right·mod=.)
2. OTÁZKA   → týž index (mod=?):  „Kdo patřil mezi pátečníky?"
3. SYNTETICKÁ vazba (virtuální, vzniká PŘI GENEROVÁNÍ):
      šablona otázky (OBÁLKA = rámy + díra)  →  konkrétní Occurrence odpovědi
4. Jedna šablona matchne NE jednu, ale VÍC faktů/tvrzení, každý s jinou HLADINOU
5. AKTIVACE = light-beam: vezme KONKRÉTNÍ slova (ne obálku), rozsvítí správnou
      přes vazby; distribuce ∝ (váha hran) a (síla uzlů). Teplo přichází Z ROZHOVORU.
```

## 3. Index: Occurrence + Frame

Každý token korpusu se rozloží na dvě úrovně (dedup na úrovni rámu):

- **Frame** („triplet", poloměr `r`, konfig): bezobsažný gramatický podpis
  `left · upos · right · modalita`. Sousedé nesou jen `upos` (+ tense u sloves);
  interpunkce je plnohodnotný prvek okna; **modalita** (`.` `?` `!` `:`) je součástí
  dedup klíče → otázkový rám je stín tvrzovacího rámu. `frame_id` = deterministický
  `blake2b` (nikdy vestavěný `hash()` — past #58).
- **Occurrence** (výskyt cíle, bohatě): `form`, `lemma` (základní tvar), `upos`,
  `gender`/`case`/`number`/`tense` (aplikovatelné-jen). Sousedé jen třídou, obsah
  nese cíl.

Konfigurace (`IndexConfig`, uložená v manifestu indexu — `index = f(korpus, config)`):
`radius`, `modality_in_key`, `modality_marks`, `self_feats`, `neighbor_feats`, jazyk.

## 4. Kanonizace (1. pád / základní tvar)

Uzel grafu = **základní tvar slova** (lemma), deduplikovaný přes segment. Řešené
artefakty UDPipe (uniformní pravidla, bez seznamu výjimek):

- **Jednopísmenné iniciály** (T. G. Masaryk) — NOUN/PROPN délky 1 se nestávají uzly.
- **Epentetické -e- příjmení** — `Čapků` → lemma `Čapk` → vlož -e- → `Čapek`
  (je-li výsledek známé četnější PROPN).
- **Přivlastňovací přídavná jména** (`Poss=Yes` ADJ) → base jméno: `Karlova`→`Karel`,
  `Čapkův`→`Čapek`, `Bechtěrevovou`→`Bechtěrev`. Uniformně: „Univerzita Karlova"
  se rozloží na `Karel` + `univerzita` (že je to Karel IV., ne Karel Čapek, tenhle
  slovní graf nerozlišuje — stejně jako `Josef` splývá napříč Biblí i Čapkem).

## 5. Salience: tf·idf (df-based)

Váha slova není holá četnost, ale **jak často je ZDE × jak je vzácné v korpusu**:

```
tf(w)   = četnost základního tvaru v segmentu (článku)
df(w)   = v kolika dokumentech korpusu (paměťového textu) se vyskytuje
idf(w)  = ln((D+1)/(df(w)+1))          [D = počet dokumentů]
tf·idf  = tf · idf
emise e = tf·idf / max(tf·idf)          [normalizace 0..1]
```

Důsledek: `v`/`a`/`být` (ve všech dokumentech, df=D) → idf≈0 → nesvítí; `Čapek`
(zde časté, jinde vzácné) → nejjasnější. Ověřeno: Čapek 6.78 > Karel > Josef;
`v`/`a`/`být` = 0. Z `e` vytéká **size** (`0.6+2·e`, vizuál), **mass** (`e`, fyzika +
aktivace) i **barva/jas** uzlu.

## 6. Graf: uzly + dvě vrstvy hran

- **Uzly** = základní tvary slov (bez interpunkce a iniciál).
- **Vrstva 1 — sousednost** (bigram v pořadí čtení): váha = četnost společného
  výskytu. Nad korpusem se z toho stává automat přechodů mezi slovy.
- **Vrstva 2 — mesh** (dopředné hrany z jmen na následující slova, `λ^d`, `λ=0.55`,
  cutoff `ε`): proximitní kotva k entitě, útlum vzdáleností (spreading activation
  povýšená na strukturu).

## 7. Fyzika (layout, viewBase)

Layout uspořádá graf **týmiž veličinami**, které pak řídí aktivaci:

- **link** — délka ∝ opak váhy (silná hrana krátká → shluky), síla ∝ váha.
- **gravitace** (`forceX/Y/Z`) — báze pro všechny + navýšení ∝ `mass` (četná slova
  do středu).
- **odpuzování** (`forceManyBody`) — báze pro všechny + navýšení ∝ `mass` (velké
  uzly víc vzduchu, líp vidět).

Ladicí konstanty (`core.js`): `GRAVITY_BASE/MASS`, `CHARGE_BASE/MASS`,
`LINK_DIST_MIN/MAX`. **Otevřené: kalibrace.**

## 8. Vizualizace (viewBase)

Renderer = viewBase (`dimensions` 2/3), styl projektu. Dodělané ve viewBase:

- **Per-hranový jas** — hrany nesou vertex-colors (`edge.meta.brightness` / `color`);
  dřív jeden materiál pro všechny. Jas rovnoměrně dle pořadí (ne dle skew hodnot),
  glow střední cyan (ne bílá) kvůli bloomu.
- **Fyzika váhami/hmotami** — `mass` uzlu a `weight` hrany protaženy do `d3-force`.
- **`show_sentence(canvas, g, match)`** — zvýrazní větu (uzly+hrany) v červeném
  odstínu: přesune jas do červeného kanálu (`R'=max(R,B), B'=min(R,B)`), jas nechá;
  zvládne modrou (slova) i teplou (jména) paletu → jednotně červená věta.

Všechny atributy jdou do `meta` uzlu → detail okno je ukáže. Okna: ⚡ info, 🏆 TOP 5.

## 9. Generování otázek

`generate_questions(sent)` — pro každý obsahový slot predikátu vyrobí **holou** otázku
(díra + tázací slovo + modalita `?`), zatím bez rozvití a bez složených otázek.
Příklad („Mezi pátečníky patřili … prezident Masaryk / bratři Čapkové"):

| otázka (holá) | díra | odpověď |
|---|---|---|
| Kdo patřil mezi pátečníky? | podmět/osoba | bratři Čapkové, prezident Masaryk |
| Mezi koho patřili bratři Čapkové? | skupina | pátečníci |
| Patřili bratři Čapkové mezi pátečníky? | zjišťovací | ano |

Gramatické mezery surového generátoru (k dořešení): shoda slovesa (`Kdo patřil`, ne
`patřili`), nominativizace odpovědi (`Čapkové`, ne `Čapků`), složení roztrhaných
jmen (`T. G. Masaryk`).

## 10. Syntetická vazba Q→A

Šablona otázky (obálka = rámy + díra) ukazuje **virtuálně** na konkrétní Occurrence
odpovědi. **Proč MUSÍ být syntetická** (ne jen swap modality): slovosled otázky
(fronta tázacího slova) změní i sousedy, takže rámy se nekryjí —

```
Q  patřit  frame(PRON·VERB·ADP·?)
A  patřit  frame(NOUN·VERB·ADP·.)   ← jiný rám, NELZE odvodit swapem modality
```

Proto vazba **vzniká při generování** (víme mapování Q↔A, protože otázku stavíme
z tvrzení). Jedna šablona matchne **víc faktů**, každý s **hladinou** (base = salience).

## 10b. Šablona odpovědi (odpověď jako fakt)

Odpověď **není konečný fragment**. Zacházíme s její indexací **stejně jako s ostatními
částmi věty** — má tedy **vlastní šablonu**. Přichází ve dvou zrnitostech:

| forma | příklad | co to je |
|---|---|---|
| **fragment** | `Čapek`, `1980`, `odešel` | výplň díry — answer-slot v příslušném pádě/tvaru |
| **věta** | `Autorem byl Čapek`, `Stalo se to v roce 1980` | odpověď utvořená jako **tvrzení** |

**Proč odpověď MŮŽE mít vlastní šablonu** — protože věta „Autorem byl Čapek" **je sama
tvrzení = fakt**. A fakt se u nás indexuje principem Occurrence + Frame (§3). Z toho
plynou tři věci:

1. **Uniformita.** Answer-slot je týž druh objektu jako podmět, predikát či doplněk —
   pozice ve faktu popsaná rámem. Šablona otázky, syntetická vazba (§10) i šablona
   odpovědi jsou **tři Occurrence+Frame struktury téhož typu**; nic zvláštního na konci.
2. **Rekurze / skládatelnost.** Odpověď-věta je fakt → může generovat **vlastní otázky**
   („Autorem byl Čapek" → „Kdo byl autorem?" → `Čapek`). Smyčka fakt→otázka→odpověď se
   **uzavírá sama do sebe**; odpověď je uzel, ze kterého vede další dotazování, ne list.
3. **Dvě zrnitosti z jedné šablony.** Fragment je holá výplň díry (čte se z answer-slotu
   přes rolově-tázací mapování §9). Věta vzniká **vsazením rozřešeného účastníka do
   predikátového rámu** (podmět·predikát·doplněk) — tatáž šablona, jen víc slotů vyplněno.

Právě proto není odpověď „mrtvý" výstup: je to fakt v malém, znovu-indexovatelný a
znovu-dotazovatelný. To dělá ze systému **generativní graf**, ne jednosměrnou lookup tabulku.

## 10c. Katalog rolí = schéma odpovědi (větné členy)

Odpověď není `{who, where}` ad hoc — je to fakt **rozložený do univerzálního katalogu
větných členů**, a ten katalog **je schéma odpovědi**. Klíče jsou jazykově NEZÁVISLÉ
(`who`, `where`, `action`, `state`, `to_whom`, `with_whom_what`…), popisy i mapování
předložek/pádů jsou **data** v `lang/cs.json` (zákon 3). Katalog pokrývá podměty,
přísudky (slovesný, jmenný se sponou, příčestí činné/trpné), příslovečná určení
(místo, čas, způsob, míra, příčina, účel, podmínka, přípustka), předmětové pády
(2./3./4./6./7.), přívlastek a doplněk.

**Jeden slovník klíčů, žádné dvojí pojmenování.** UD deprel z UDPipe (`nsubj`, `obj`,
`obl`+pád, …) je jen **vstupní štítek**; přes `deprel_to_role` (+ pád/životnost/předložka)
se překládá na **jediný** kanonický klíč katalogu. Registr (`gen_registry`), tázací slova
(`role_ask`) i rozklad odpovědi (`roles.py`) sdílejí týž mapovač `run.role_key` — stejný
význam má vždy stejný klíč. (`nsubj`+Anim→`who`, `nsubj`+Inan→`what_subject`, `obj`+Acc→
`whom_what`, `iobj`→`to_whom`, `obl`+Loc/„v"→`where`, `obl`+Ins/„s"→`with_whom_what`,
kopulní predikativ→`state`.)

**Rozklad je per klauzule.** Každý přísudek zakládá klauzuli; token patří ke svému
nejbližšímu přísudku. Věta „Josef a Maria odešli do Jeruzaléma, kde se jim narodil Ježíš"
→ dvě klauzule:

```
«odejít»            who=[Josef, Maria]  where=Jeruzalém  past_participle=odešli
«narodit» [acl]     who=[Ježíš]         where=Jeruzalém  to_whom=jim  past_participle=narodil
                                        ↳ #59: „kde" zděděno z antecedentu Jeruzalém
```

**Meziklauzulová inference (#59).** Vedlejší (relativní/účelová) klauzule visí na
antecedentu z řídící věty (`narodit` je `acl` s hlavou `Jeruzalém`). Relativní příslovce
(`kde`, `kam`, `kdy`) je **díra**, kterou plníme hodnotou antecedentu — sub-klauzule
zdědí místo. Tak otázka „Kdo se narodil v Jeruzalémě?" trefí druhou klauzuli a vrátí
`{who: Ježíš, where: Jeruzalém}`, ačkoli slovo „Jeruzalém" v ní přímo není.

## 11. Aktivační výběr (light-beam)

Výběr správné odpovědi mezi kandidáty **nedělá obálka** (rám), dělá ho **aktivace
konkrétních uzlů** (Čapek / bratr / Masaryk), a ta **vyplývá z rozhovoru**:

```
1. SEED   — z rozhovoru se rozsvítí konkrétní slova
2. ŠÍŘENÍ — světlo teče po hranách ∝ VÁHA SPOJNICE (vodivost)
3. AKUMULACE — v uzlech se sčítá, zesíleno ∝ SÍLA UZLU (mass = tf·idf, kapacita)
4. VÍTĚZ  — kandidát (napojený syntetickou vazbou na díru) s nejvíc světlem
```

Příklad: rozhovor o Karlu Čapkovi → seed `Čapek` (mass 1.0) → hrana `bratr–Čapek`
(váha 2) vtáhne `bratr` → vyhrává „bratři Čapkové". Rozhovor o prezidentech → vyhrává
„Masaryk". **Táž otázka, jiná odpověď podle kontextu.**

Aktivace není nový mechanismus — je to projektový `ActivationField` / pro-drop
(„nejteplejší rodově shodná osoba těžiště"), aplikovaný jako výběr odpovědi nad
syntetickým otázkovým grafem.

## 12. Sjednocení

Tytéž dvě veličiny slouží dvěma rolím:

| veličina | fyzika (layout) | aktivace (výběr odpovědi) |
|---|---|---|
| **váha hrany** | délka + síla pružiny | vodivost světla |
| **síla uzlu (mass)** | gravitace + odpuzování | zesílení/kapacita světla |

## 13. Otevřené otázky (výzkum)

1. **Správná distribuce aktivace vzhledem k parametrům grafu** — jak přesně světlo
   teče: útlum na skok, převodní funkce vodivosti z váhy hrany, funkce zesílení ze
   síly uzlu, normalizace, počet skoků, práh. Empirická kalibrace (jako fyzikální
   konstanty) — potřebuje měřicí harness.
2. **Krátkodobá paměť udržující kontext diskuse** — struktura pracovní paměti, která
   drží aktuálně žhavá slova rozhovoru; jak se přenáší mezi tahy. Základ =
   `ActivationField` (warm/step, útlum ×0.55/krok).
3. **Přidávání aktivačních slov a jejich expirace** — jak se do aktivace dostávají
   slova (z uživatelského vstupu, z odpovědí), a jak expirují (per-tah útlum? čas?
   práh vyhasnutí?).

## 14. Stav a soubory

- **Experiment:** `experiments/hypothesis-one/run.py` (index, kanonizace, tf·idf,
  graf 2/3D, `generate_questions`, `show_sentence`). Server na `:8080`.
- **viewBase:** per-hranový jas (`frontend/src/render/renderer.js`), fyzika váhami
  (`frontend/src/physics/core.js`, `engine.js`). Build → `python/viewbase/static`.
- **Korpus:** `data/annotations.pkl` (nacachované UDPipe anotace, 13 dokumentů).

## 15. Další kroky

1. **Spustitelné demo light-beamu** — seed kontextu (rozsvítit `Čapek`/`Masaryk`),
   šablona otázky, aktivace přes váhy/síly vybere vítěze; rozsvícená cesta v grafu.
2. Materializace syntetických Q→A vazeb jako hranového typu v grafu.
3. Kalibrační harness pro distribuci aktivace (otevřená otázka 1).
4. Krátkodobá paměť + expirace aktivačních slov (otevřené otázky 2–3).
5. Dořešení gramatiky generátoru (shoda, nominativizace, složení jmen).
