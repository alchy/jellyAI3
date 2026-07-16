"""Šablony odpovědí podle typu otázky.

Odpovědní entita se z textu vytáhne často v „šikmém" pádě („napsala Babičk**u**",
„narodil se v Praz**e**"). Šablona říká, do jakého pádu ji převést, aby odpověď
zněla čistě. Pro V1 volíme bezpečnou variantu: **1. pád (nominativ)** — kanonický
tvar („Babička", „Praha", „Božena Němcová"). Data (Kdy) a čísla (Kolik) se nesklo­
ňují. Frame je zatím jen `{answer}`; místo pro plné věty se tu dá snadno rozšířit.
"""

_NOMINATIVE = "1"  # pozice 5 (pád) v PDT tagu = 1. pád

# typ otázky → {frame, case}. `case` = cílový pád pro skloňování (None = nesklo­ňovat).
TEMPLATES = {
    "Kdo": {"frame": "{answer}", "case": _NOMINATIVE},
    "Co": {"frame": "{answer}", "case": _NOMINATIVE},
    "Kde": {"frame": "{answer}", "case": _NOMINATIVE},
    "Kdy": {"frame": "{answer}", "case": None},
    "Kolik": {"frame": "{answer}", "case": None},
    "Jaký": {"frame": "{answer}", "case": None},        # přísudek/adjektivum — ponech tvar (shodne se sám)
    "Který": {"frame": "{answer}", "case": _NOMINATIVE},
    "Čí": {"frame": "{answer}", "case": _NOMINATIVE},   # přivlastnění → osoba v 1. pádě
}


def target_case(qtype):
    """Vrátí cílový pád pro daný typ otázky (nebo None = neskloňovat).

    Args:
        qtype (str): Typ otázky (Kdo/Co/Kde/Kdy/Kolik).

    Returns:
        str | None: Číslice pádu (PDT), nebo None.
    """
    return TEMPLATES.get(qtype, {}).get("case")


def fill(qtype, inflected_answer):
    """Vloží (už sklonovanou) odpověď do šablony pro daný typ.

    Args:
        qtype (str): Typ otázky.
        inflected_answer (str): Odpověď už převedená do cílového pádu.

    Returns:
        str: Hotová odpověď.
    """
    frame = TEMPLATES.get(qtype, {}).get("frame", "{answer}")
    return frame.format(answer=inflected_answer)
