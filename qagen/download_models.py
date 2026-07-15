"""Stažení českých modelů pro MorphoDiTa a NameTag z repozitáře LINDAT.

Modely jsou zabalené v ZIPech; stáhneme je a vytáhneme z nich potřebný soubor
(*.tagger pro MorphoDiTu, *.ner pro NameTag). Licence modelů je CC BY-NC-SA
(nekomerční) — pro osobní/výukový projekt v pořádku. Stahování je best-effort:
když zdroj selže, jen to ohlásí a pokračuje dál.
"""

import fnmatch
import os
import shutil
import tempfile
import urllib.request
import zipfile

# ÚFAL nainstalovaná verze je NameTag 1 + MorphoDiTa 1 → odpovídající modely:
#   MorphoDiTa: MorfFlex CZ 2.0 + PDT-C 1.0 (LINDAT handle 11234/1-4794)
#   NameTag: CNEC (LINDAT handle 11858/00-097C-0000-0023-7D42-8)
_LINDAT = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle"
MODELS = [
    {
        "url": f"{_LINDAT}/11234/1-4794/czech-morfflex2.0-pdtc1.0-220710.zip"
               "?sequence=1&isAllowed=y",
        "member": "*.tagger",
        "dest": "data/models/czech-morfflex.tagger",
    },
    {
        "url": f"{_LINDAT}/11858/00-097C-0000-0023-7D42-8/czech-cnec-140304.zip"
               "?sequence=1&isAllowed=y",
        "member": "*.ner",
        "dest": "data/models/czech-cnec.ner",
    },
]


def _find_member(zf, pattern):
    """Najde v ZIPu největší soubor, jehož jméno odpovídá vzoru.

    Když balíček obsahuje víc kandidátů (např. plný a „pos-only" tagger), vezme
    ten největší — to bývá plný model, který chceme.

    Args:
        zf (zipfile.ZipFile): Otevřený ZIP.
        pattern (str): Glob na název souboru (např. „*.tagger").

    Returns:
        str: Název položky v ZIPu.

    Raises:
        KeyError: Když žádná položka vzoru neodpovídá.
    """
    matches = [i for i in zf.infolist()
               if fnmatch.fnmatch(os.path.basename(i.filename), pattern)]
    if not matches:
        raise KeyError(f"V ZIPu není soubor odpovídající {pattern}")
    return max(matches, key=lambda i: i.file_size).filename


def download_models(config=None, models=MODELS):
    """Stáhne modely a rozbalí je do data/models.

    Args:
        config: Nepovinné (rezervováno pro budoucí cesty z konfigurace).
        models (list): Položky {url, member, dest}.

    Returns:
        list[tuple[str, bool]]: (cílová cesta, úspěch) pro každý model.
    """
    results = []
    for model in models:
        dest = model["dest"]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = tmp.name
            print(f"Stahuji {model['url']} …")
            urllib.request.urlretrieve(model["url"], tmp_path)
            with zipfile.ZipFile(tmp_path) as zf:
                member = _find_member(zf, model["member"])
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            print(f"Rozbaleno: {dest}")
            results.append((dest, True))
        except Exception as exc:  # noqa: BLE001 - chceme pokračovat dalšími
            print(f"Přeskočeno {model['url']}: {exc}")
            results.append((dest, False))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return results
