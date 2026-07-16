"""Kanonizace entit clusteringem variant — robustní náhrada za lemma-normalizaci.

ZAPOJENO (2026-07-16): `cluster_key`/`_stem` pohání **post-build resolver**
`graph.resolve_entities` (build-side sloučení pádových uzlů) a kmenový fallback
v `_resolve_topic` (query-side) — týž mechanismus na obou stranách, takže se
nerozejdou. Post-build pass obchází dřívější „whack-a-mole" (přejmenování uzlů
za běhu buildu hýbalo pro-drop koreferencí): koreference doběhne nad povrchovými
tvary a sjednocuje se až hotový graf. `build_entity_canon` zůstává nezapojený
obecný nástroj (frekvenční kanonizace povrchových tvarů).

Povrchový tvar tříští jednu entitu na víc uzlů podle pádu („Božena Němcová" /
„Boženy Němcové" / „Boženu Němcovou"). Morfologie (UDPipe lemma i MorphoDiTa) je
na česká vlastní jména nespolehlivá. Proto **neodvozujeme nominativ** — místo toho
**shlukujeme varianty** podle kmene. Klíč = n-tice kmenů slov: pádové koncovky
mění konec slova, kmen drží. Pravidla kmenování jsou **data jazyka**
(`jellyai/lang/<jazyk>.json`) — jazyk je zásuvný modul, ne kód.

Souvisí s [[jellyai3-fact-graph]] (řeší tříštění entit — agentem označený #1 dopad).
"""

from collections import Counter

from jellyai.lang import load_rules

_RULES = load_rules("cs")      # výchozí jazyk; přepíná `set_language` (config)


def set_language(language):
    """Přepne jazyk kmenování — kód jazyka („cs") nebo cesta k JSON s pravidly.

    Jazyk je zásuvný datový modul (`jellyai/lang/`): core zůstává agnostický,
    pravidla se mění bez zásahu do kódu.

    Args:
        language (str): Kód jazyka nebo cesta k `.json` souboru.
    """
    global _RULES   # pylint: disable=global-statement
    _RULES = load_rules(language)


def _stem(word):
    """Kmen jména: odstraní pádovou koncovku a **epentetickou samohlásku**
    (Karel/Karla, Čapek/Čapka), aby všechny pády daly stejný kmen. Bez
    morfologie — jen pravidla, a ta jsou **data jazyka** (`jellyai/lang/`).

    Česky: Božen-a/-y/-u/-ě → „bož(e)n"; Němcov-á/-é/-ou → „němc"; Karel/Karl-a
    → „karl"; Čap-ek/-ka → „čapk". Záměrně hrubé (drobné přeříznutí nevadí,
    když je konzistentní).
    """
    rules = _RULES
    low = word.lower()
    for suffix in rules["suffixes"]:
        if low.endswith(suffix) and len(low) - len(suffix) >= rules["min_stem"]:
            low = low[:-len(suffix)]
            break
    # epenteze: koncová „samohláska+souhláska" → jen souhláska (karel→karl)
    vowel = rules["epenthesis_vowel"]
    if vowel and len(low) >= max(rules["min_stem"], 2) \
            and low[-2] == vowel and low[-1] not in rules["vowels"]:
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
