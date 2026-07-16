"""Kanonizace entit clusteringem variant — robustní náhrada za lemma-normalizaci.

⚠️ PARKOVÁNO: de-riskováno a otestováno (viz test_canon.py), ale ZATÍM NEZAPOJENO
do `build_graph`. Prosté zapojení dělá „whack-a-mole" — přejmenování uzlů rozhýbe
pro-drop koreferenci a fakty se stěhují (etalon zůstal 9/11, jen s jinou rozbitou
dvojicí). Robustní nasazení chce **coreference-aware entity resolution** (sjednotit
i fragmenty Karel↔Karel Antonín Čapek a query-side), ne jen tuto clustering vrstvu.
Necháno jako otestovaný stavební kámen pro ten principiální krok.


Povrchový tvar tříští jednu entitu na víc uzlů podle pádu („Božena Němcová" /
„Boženy Němcové" / „Boženu Němcovou"). Morfologie (UDPipe lemma i MorphoDiTa) je
na česká vlastní jména nespolehlivá. Proto **neodvozujeme nominativ** — místo toho
**shlukujeme varianty** podle kmene a za kanonické id bereme **nejčastější tvar**
clusteru (nominativ bývá nejfrekventovanější). Jeden špatný tvar tak identitu
nerozbije. Klíč = n-tice kmenů slov: pádové koncovky mění konec slova, kmen drží.

Souvisí s [[jellyai3-fact-graph]] (řeší tříštění entit — agentem označený #1 dopad).
"""

from collections import Counter

_MIN_STEM = 3      # kmen nezkracuj pod 3 znaky
# pádové koncovky českých jmen (nejdelší první) — ženská -ová, prostá -a/-y/-u…
_SUFFIXES = ("ovými", "ových", "ovém", "ovou", "ové", "ová", "ovi", "ovu", "ovy",
             "ými", "ých", "ém", "ům", "ách", "emi", "ami", "ou", "em", "ěm",
             "e", "ě", "y", "u", "a", "o", "i", "í", "é")
_VOWELS = "aeiouyáéíóúýěů"


def _stem(word):
    """Kmen jména: odstraní pádovou koncovku a **epentetické -e-** (Karel/Karla,
    Čapek/Čapka), aby všechny pády daly stejný kmen. Bez morfologie — jen pravidla.

    Božen-a/-y/-u/-ě → „bož(e)n"; Němcov-á/-é/-ou → „němc"; Karel/Karl-a → „karl";
    Čap-ek/-ka → „čapk". Záměrně hrubé (drobné přeříznutí nevadí, když je konzistentní).
    """
    low = word.lower()
    for suffix in _SUFFIXES:
        if low.endswith(suffix) and len(low) - len(suffix) >= _MIN_STEM:
            low = low[:-len(suffix)]
            break
    # epenteze: koncové „e+souhláska" → jen souhláska (karel→karl, čapek→čapk)
    if len(low) >= _MIN_STEM and low[-2] == "e" and low[-1] not in _VOWELS:
        low = low[:-2] + low[-1]
    return low


def cluster_key(name):
    """Kmenový klíč jména — n-tice kmenů jeho slov (case-varianty → stejný klíč)."""
    return tuple(_stem(word) for word in name.split())


def build_entity_canon(surface_freq):
    """Z četností povrchových tvarů sestaví mapu **tvar → kanonické id**.

    Varianty se stejným kmenovým klíčem tvoří cluster; kanonické id = nejčastější
    tvar clusteru (při shodě četnosti lexikograficky nejmenší, kvůli determinismu).

    Args:
        surface_freq (dict[str, int]): Povrchový tvar entity → jeho četnost v korpusu.

    Returns:
        dict[str, str]: Tvar → kanonické id (nejčastější varianta clusteru).
    """
    clusters = {}
    for surface, freq in surface_freq.items():
        clusters.setdefault(cluster_key(surface), Counter())[surface] += freq
    canon = {}
    for members in clusters.values():
        best = max(members, key=lambda s, m=members: (m[s], _neg_lex(s)))
        for surface in members:
            canon[surface] = best
    return canon


def _neg_lex(text):
    """Pomocná: pro tie-break „nejčastější, pak lexikograficky nejmenší"."""
    return tuple(-ord(ch) for ch in text)
