# Testovací seance nad grafem Bible (2026-07-20 večer) — nálezy a návrhy

> Role tester (zadání user): přes JellyAI zjistit, s kým se Ježíš
> potkal, kde, co komu řekl; identifikovat zázraky, místa, aktéry.
> Dialog veden přes `data/web_inbox.txt` (viditelný ve webové
> konzoli), odpovědi čteny z telemetrie. Build `bdc5121` (handshake
> #40 odhalil, že běžela ranní verze — restart). Cíl: nedostatky
> answereru a návrhy úprav (karty × principy) — krmivo pro plán #51
> a další práci.

## Co FUNGOVALO (dnešní features naostro)

- **E3 chytrá clarifikace:** „Kde uzdravil Ježíš?" → „O ději
  „uzdravit“ vím co, kdo, kolik; kde nevím." ✓
- **E4 proaktivní nabídka + drill:** „Mohu doplnit: kdo." ✓
- **Směr řeči (#55):** „Co řekl Ježíš Petrovi?" → skutečná slova
  Petrovi ✓ (přes poziční šablonu, bez karty)
- **Homonymní volba, q-co-vime agregace, pro-drop** — mechanicky ✓.

## Nálezy (od nejzávažnějších)

| # | Nález | Ukázka | Kořen |
|---|---|---|---|
| T1 | **Pojem „setkání" není predikát** — biblický text ho nese slovesy přistoupit/přijít k/uvidět; `potkat` fakty Ježíše vůbec nemají | „S kým se Ježíš potkal?" → nenašel; „Koho potkal Ježíš?" → figl (T3) | chybí predikátové TŘÍDY (kategorie dějů) |
| T2 | **Vidové páry nesjednocené** — `uzdravil` nevidí `uzdravovat` fakty (půlka evidence) | „Koho uzdravil Ježíš?" → jen „nemocný"; uzdravovat(nemocný/nemoc/choroba) neviditelné | chybí vidová kanonizace predikátů |
| T3 | **Kontextové patro vyrábí figly u dějových otázek** — asociace se tváří jako fakt, E4 nabídka ji zesiluje | „Koho potkal Ježíš?" → „Šimon" (pred=kontext); „Kdo přišel k Ježíšovi?" → „Ježíš Nazaretský" + „Mohu doplnit: co." | kontext tier bez označení + nabídka bez guardu |
| T4 | **Pasivum převrací role** + theme plní person díru | „Koho vzkřísil Ježíš?" → „den" (fakt: vzkřísit(subj Ježíš, theme den) z „byl vzkříšen třetího dne") | extrakce pasiva; typový guard díry |
| T5 | **Předložkové tázací vazby bez karet** | „S kým se…", „Kdo přišel K JEŽÍŠOVI?", „Co udělal X V Y?" — vše mimo karty (šablony/figl) | E1 odklad dekoračních prvků — důkaz potřeby |
| T6 | **Formát výčtů: výroky slepené čárkou, agregace telegraf** | „satane, nemohli jedinou hodinu bdít se mnou" (2 promluvy!); „poslat: Ježíš, otázka, ctít: den, strach…" | Echo (#20) chybí; theme/time role prosakují do obsahu |
| T7 | **Místní filtr propouští fakty bez místa** → falešné „v Kafarnaum" | „Co udělal Ježíš v Kafarnaum?" → „stan" (Proměnění, jinde; a slova Petrova) | pravidlo „bez místa projde" (převzato od času) u obsahových děr |
| T8 | **Střepy Ježíše režou recall** — po volbě „Ježíš" zmizí fakty Nazaretského (syn, chlapec u uzdravit) a volba otravuje každé vlákno | celá seance | #8 fáze 2 (potvrzení priority) |
| T9 | **Drill bez vysvětlení** — „Kde?" po řeči → generický terminál | fakt říci loc nemá — E3 vysvětlení se na drill nevztahuje | rozšířit empty-role na drill cestu |
| T10 | **Šum extrakce:** klauzule jako objekt, imperativ jako jméno | obj=„odešel opět na horu zcela sám", subj=„Proste" | hygienové guardy (#2 rodina) |

## Návrhy úprav

**A. Jazyková data (levné, rychlé):**
1. `aspect_pairs` — katalog vidových dvojic (uzdravit↔uzdravovat…);
   matching přes pár mechanismem `_verb_match` (negační pár už
   existuje — týž vzor). → T2
2. `predicate_classes` — třídy dějů jako data: setkání =
   {potkat, setkat, přistoupit, přijít+k-osobě}, zázrak =
   {uzdravit, vzkřísit, nasytit, utišit…}; otázka na třídu expanduje
   na členy (vzor: slovesná třída přesunu u Topos). Dlouhodobě:
   druh-hrany nad DĚJI = datový základ oceánu #41 i pro predikáty. → T1
3. Řazení `role_labels` ve výčtu clarifikace (kdo, co, kde, kdy,
   kolik — pevné pořadí, ne abecedně ze setu). → kaz E3

**B. Answerer principy:**
4. **Empty-topic clarifikace** (sourozenec E3): predikát roli má, ale
   žádný fakt s TÍMTO tématem → místo kontextového figlu „O ději
   „potkat“ s Ježíšem nic nevím; vím o: Jidáš, Šimon…" (kandidáti =
   subjekty faktů predikátu). Kontextové patro nechat jen pro
   kontextové díry, NEBO odpověď označit šablonou („souvisí: …"). → T3
5. Nabídku (E4) nepřipojovat ke kontextovým odpovědím (guard:
   pred=kontext). → T3
6. Typový guard: theme účastník nesmí plnit person/obj díru
   (zobecnění „pozorovatel není odpověď"). → T4
7. Místní filtr: u obsahové díry s místní oblastí preferovat fakty
   S místem; bez nich E3-styl „nevím, co v Kafarnaum". → T7
8. Empty-role vysvětlení i pro drill cestu. → T9

**C. Karty (plán #51 / E1b):**
9. Dekorační předložkové/pádové prvky rodiny q-otaz („s kým",
   „k Ježíšovi", „v Kafarnaum") — odložené svinutí dekoračních karet
   z E1 má teď důkaz potřeby. → T5
10. Formát hodnot podle TYPU: výroky v uvozovkách po jednom,
    agregace větami, theme/time mimo obsah (zárodek Echo #20). → T6

**D. Extrakce/hygiena (samostatná dávka):**
11. Pasivum (být + příčestí trpné → obrátit role). → T4
12. Guard: klauzule jako objekt, imperativ jako subjekt. → T10

## Doporučené pořadí

A1+A3+B5+B6 (drobné, hned) → B4 (empty-topic — princip, měřený
etalonem) → A2 (třídy dějů — koncept s userem) → C9+C10 (s plánem
#51) → B7+B8 → D (extrakční dávka s přestavbou grafu).
