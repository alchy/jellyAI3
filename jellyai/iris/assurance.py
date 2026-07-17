"""QueryAssurance — číselná jistota ZAOSTŘENÍ subjektu dotazu (0–1).

Odpovídá na otázku „jak moc jsem si jistý, že jsem dotaz zaostřil na správný
uzel grafu?". Automat Iris podle ní přepíná: nad prahem odpověz, pod prahem
veď dialog (viz princip „dialog > figly" — místo hádání se ptát).

Skóre skládá tři síly:

1. **Kvalita jmenné evidence** (`quality`, 0–1) — průměrná váha nejlepšího
   patra rozlišení na term dotazu: přesná shoda (1.0) > povrchová (0.8) >
   kmenová (0.6) > bezdiakritická (0.4) > volná (0.25). Počítá ji resolver
   (`_resolve_topic`) při rozlišení.
2. **Rovnocenní soupeři** (`rivals`) — kandidáti s TOUTÉŽ jmennou evidencí
   z jiného kmenového clusteru („Kdo je Čapek?" → Karel i Josef). Volba mezi
   n rovnocennými je hádání s jistotou ~1/n, proto se kvalita dělí.
3. **Aktivace vítěze** (`activation`) — jas uzlu v konverzačním těžišti.
   Když uživatel v minulém tahu zaostřil (vybral kandidáta), jeho uzel svítí
   a jistota se vrací nad práh — dialogová smyčka tak přirozeně konverguje.
"""

# váhy pater rozlišení (viz GraphAnswerer._resolve_topic — tatáž patra)
TIER_WEIGHTS = {"exact": 1.0, "ins": 0.8, "stem": 0.6, "da": 0.4, "loose": 0.25}

_ACTIVATION_GAIN = 0.3   # kolik jistoty přidá plně svítící vítěz


def assurance(quality, rivals, activation=0.0):
    """Spočítá jistotu zaostření (0–1).

    Args:
        quality (float): Kvalita jmenné evidence vítěze (0–1; viz modul).
        rivals (int): Počet rovnocenných soupeřů z jiných clusterů.
        activation (float): Jas vítěze v konverzačním poli (≥0; nad 1 se
            ořezává — víc než „plně svítí" neexistuje).

    Returns:
        float: Jistota zaostření v intervalu ⟨0, 1⟩.
    """
    score = quality / (1.0 + max(0, rivals))
    score += _ACTIVATION_GAIN * min(max(activation, 0.0), 1.0)
    return max(0.0, min(1.0, score))
