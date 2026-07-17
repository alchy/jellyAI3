# Šablonový parser otázek → pseudo-QL (bez UDPipe)

> Stav: **ZÁKLAD hotový, VYPNUTÝ** (`GraphAnswerer.use_templates=False`).
> `jellyai/answerer/query.py` + `tests/test_query.py` (6 testů). Etalon 24/24
> drží (UDPipe primární). Flip na primární AŽ po paritě.

**Mechanismus (cíl uživatele):** otázka → identifikace vzoru → transformace do
pseudo query language → dotaz do grafu. Korpus jen staví graf; dotaz jede
deterministickými šablonami. Slovník dotazu = **sám graf** (predikáty faktů) +
jazykové tabulky (tázací slova, spona, vztahová jména, synonyma). Odolné vůči
diakritice (fold) i mis-taggingu UDPipe.

**Hotovo:** tázací slovo → díra (role/typ); slovesný tvar → predikát grafu
**prefixem** (napsal≡napsat); entita = běh obsahových tokenů; vztahové jméno →
predikát; „který" → rekurzivní SubQuery; guard proti falešné shodě jméno↔sloveso
(velké písmeno). Změřeno: šablony PRIMÁRNÍ = **17/24** jádra.

## Gapy do parity (24/24), pak flip na primární

1. **Rozdělení víceslovných běhů**: „Napsal Karel Čapek **Válku s mloky**" slepí
   jméno+titul do jednoho běhu. Potřeba: hranice mezi rozřešitelnými entitami
   (dva sousední uzly grafu = dvě entity).
2. **Typový filtr**: „**Jakou hru** napsal Karel Čapek?" — „hru" má být typový
   filtr díry (join být(?,hra)), ne druhý subj. Role u attr-díry špatná.
3. **Ano/ne otázky**: „Napsal Karel Čapek X?" — počáteční sloveso (velké
   písmeno) + bez tázacího slova = zjišťovací; teď se nerozpozná.
4. **Symetrie vztahů**: relační díra hole=subj koliduje s uloženou rolí faktu
   (bratr(subj=Karel, obj=Josef)); ověřit, že vrací protistranu.
5. **Falešné prefixové shody sloves** obecně (krátká jména vs slovesa) — přísnější
   práh nebo kontrola „token je spíš uzel grafu než sloveso".
6. **date_part / drill** („v kterém roce"), reverse „Co se stalo <datum>".

**Guardrail:** flip `use_templates=True` až když šablony PRIMÁRNÍ dají 24/24
(měřeno skriptem, který přepne `answerer.use_templates`). Do té doby UDPipe
primární, šablony jako budoucí náhrada.
