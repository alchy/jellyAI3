import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")

# Malá sada českých stopslov (funkční slova, tázací zájmena, spojky, předložky).
CZECH_STOPWORDS = {
    "a", "aby", "aj", "ale", "ani", "ano", "az", "až", "bez", "bude", "budou",
    "by", "byl", "byla", "byli", "bylo", "být", "co", "což", "či", "do", "i",
    "je", "jeho", "jej", "její", "jejich", "jen", "ještě", "jak", "jako", "jsem",
    "jsi", "jsme", "jsou", "jste", "k", "kam", "kde", "kdo", "když", "ke", "která",
    "které", "který", "kteří", "ku", "ma", "má", "mají", "mé", "mezi", "mi", "mít",
    "mne", "mně", "můj", "my", "na", "nad", "nade", "nam", "nám", "napiš", "naš",
    "náš", "ne", "nebo", "něco", "něj", "není", "než", "ní", "nic", "o", "od",
    "on", "ona", "oni", "ono", "pak", "po", "pod", "podle", "pokud", "pouze",
    "proč", "pro", "před", "přes", "při", "s", "se", "si", "ta", "tak", "také",
    "tam", "te", "tě", "ten", "tedy", "to", "tom", "tomu", "tuto", "ty", "u", "už",
    "v", "vám", "vas", "vás", "ve", "více", "však", "všech", "vy", "z", "za",
    "ze", "že",
}


def tokenize(text, stopwords=None):
    """Rozdělí text na slovní tokeny (lowercase, diakritika zachována)."""
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks


def split_sentences(text):
    """Rozdělí text na věty podle koncové interpunkce."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]
