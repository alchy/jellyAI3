import os
import re

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF.*", re.IGNORECASE | re.DOTALL)


def clean_text(raw):
    """Vyčistí syrový text; zachová diakritiku, interpunkci i velikost písmen."""
    text = raw
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_processed(raw_dir, processed_dir):
    """Vyčistí všechny .txt z raw_dir do processed_dir. Vrátí zapsané cesty."""
    os.makedirs(processed_dir, exist_ok=True)
    written = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(raw_dir, name), encoding="utf-8") as f:
            cleaned = clean_text(f.read())
        dest = os.path.join(processed_dir, name)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(cleaned)
        written.append(dest)
    return written
