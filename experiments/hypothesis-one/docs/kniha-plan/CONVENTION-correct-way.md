# Konvence „Correct way" — blok cílového stavu u každé nedokonalosti

Zadání uživatele: *„pro každou nedokonalost dej blok — jak by to mělo fungovat správně 'Correct way:'.
Tato correct way popisuje cílený výsledek, tedy ideální stav, ale nepopisuje, jak se k němu dostat.
To se týká všech kapitol, včetně funkcionalit jako logování."*

## Co to je

U KAŽDÉ nedokonalosti v knize přidej hned za ni blok „Correct way" = **cílový/ideální stav**:
jak by systém fungoval, kdyby ta nedokonalost neexistovala. Popisuje **CO** (cílený výsledek),
NIKDY **JAK** (žádné kroky, žádný mechanismus, žádná implementace).

## HTML tvar (přesně)

```html
<aside class="correct-way">
  <span class="lbl">Correct way — cílový stav</span>
  <p>[Ideální výsledek: co by uživatel/vývojář viděl, kdyby to fungovalo správně. Formulováno jako
  hotový stav, ne jako plán. Bez sloves „přidat/dopsat/implementovat/zavolat" — místo toho „systém
  má/drží/vrací/ohlásí". 2–5 vět.]</p>
</aside>
```

CSS třída `correct-way` už existuje v `assets/style.css` (zelený „north-star" tón, odlišný od
`rationale`/`open-question`). Vzorový hotový blok: `dil-7-program.html`, sekce „Logování a diagnostika".

## Kam blok patří (co je „nedokonalost")

Přidej Correct way ZA každý:
- `<div class="open-question">` a `<aside class="open-question">` (otevřené body, nedodělky),
- sekci/blok **„Co nefunguje a proč"**, který popisuje **skutečný nedostatek systému** (např.
  „Co napsal Karel Čapek → známka", „authorship-what 0/3", tichý pád koreference),
- každý **Návrh** (`mark-navrh`) a známé omezení (domény, mountu, extrakce).

## Kdy NEpřidávat (poctivost před úplností)

Některé bloky „Co nefunguje a proč" NEpopisují nedokonalost — vysvětlují **správné rozhodnutí by
design** nebo **chybu uživatele**. Tam Correct way NEpiš (současné chování UŽ JE ta správná cesta);
místo toho ponech blok, jak je. Příklady, kde se Correct way NEpřidává:
- „naivní nápad by udělal X, proto to děláme jinak" — kde to „jinak" je záměr, ne dluh
  (glow-orders-ties, decay, alias merge, predikát+role místo window-VZOR);
- „systémový Python vizualizaci nespustí" (chyba uživatele), „viewBase spadne ≠ jádro spadne" (záměr).

Když si nejsi jist, zda jde o nedokonalost, nebo o záměr: rozhoduje, **jestli existuje lepší cílový
stav, kterého systém dnes nedosahuje**. Ano → Correct way. Ne (dnešek už je ideál) → přeskoč.

## Styl obsahu

- Cílový stav, ne plán. ŠPATNĚ: „Přidej gazetteer seed a containment." SPRÁVNĚ: „Systém pozná
  i menší obce a rozumí vztahu Praha ⊂ Čechy, takže odpoví i na 'Kde v Čechách…'."
- Konkrétní k té nedokonalosti (ne obecné fráze). Naváž na skutečná čísla/příklady z okolí.
- Krátce (2–5 vět). Když už okolní blok obsahuje „jak" (doporučení/Návrh kroky), Correct way stojí
  NAD ním jako vize; můžeš přidat větu „(cesta k tomu je níže/výše)".
- Nic nevymýšlej o chování systému — cílový stav ano (je to vize), ale nepopisuj neexistující API.
