"""Stažení českých modelů pro MorphoDiTa a NameTag z repozitáře LINDAT.

LINDAT běží na DSpace 7, kde se k souboru nedostaneš přímou URL — musí se dohledat
přes REST API z trvalého „handle". Tady tedy z handle vyřešíme download link,
stáhneme ZIP a vytáhneme z něj potřebný model (*.tagger pro MorphoDiTu, *.ner pro
NameTag). Handle je stabilní, kdežto vnitřní UUID bitstreamu se může měnit — proto
řešíme dynamicky. Modely jsou CC BY-NC-SA (nekomerční) — pro osobní/výukový projekt
v pořádku. Stahování je best-effort: když zdroj selže, jen to ohlásí a jede dál.
"""

import fnmatch
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

_API = "https://lindat.mff.cuni.cz/repository/server/api"

# V2a používá jen NameTag (MorphoDiTa se s ním v jednom procesu pere — viz
# UfalTagger). Model CNEC pro NameTag 1: handle 11858/00-097C-0000-0023-7D42-8.
MODELS = [
    {"handle": "11858/00-097C-0000-0023-7D42-8", "member": "*.ner",
     "dest": "data/models/czech-cnec.ner"},
]


def _get_json(url):
    """Stáhne a rozparsuje JSON z DSpace REST API.

    Args:
        url (str): Endpoint API.

    Returns:
        dict: Rozparsovaná odpověď.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _resolve_zip_url(handle):
    """Z handle vyřeší přímou download URL ZIPu (bitstream v bundlu ORIGINAL).

    Args:
        handle (str): Trvalý handle položky (např. „11234/1-4794").

    Returns:
        str: URL na obsah ZIP bitstreamu.

    Raises:
        KeyError: Když položka nemá žádný .zip bitstream.
    """
    item = _get_json(f"{_API}/pid/find?id=hdl:{handle}")
    bundles = _get_json(f"{_API}/core/items/{item['uuid']}/bundles")
    for bundle in bundles["_embedded"]["bundles"]:
        if bundle["name"] != "ORIGINAL":
            continue
        bitstreams = _get_json(bundle["_links"]["bitstreams"]["href"])
        for stream in bitstreams["_embedded"]["bitstreams"]:
            if stream["name"].lower().endswith(".zip"):
                return stream["_links"]["content"]["href"]
    raise KeyError(f"Položka {handle} nemá žádný .zip bitstream")


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
    """Stáhne modely (přes handle→API) a rozbalí je do data/models.

    Args:
        config: Nepovinné (rezervováno pro budoucí cesty z konfigurace).
        models (list): Položky {handle, member, dest}.

    Returns:
        list[tuple[str, bool]]: (cílová cesta, úspěch) pro každý model.
    """
    results = []
    for model in models:
        dest = model["dest"]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp_path = None
        try:
            url = _resolve_zip_url(model["handle"])
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp_path = tmp.name
            print(f"Stahuji {model['handle']} …")
            urllib.request.urlretrieve(url, tmp_path)
            with zipfile.ZipFile(tmp_path) as zf:
                member = _find_member(zf, model["member"])
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            print(f"Rozbaleno: {dest}")
            results.append((dest, True))
        except Exception as exc:  # noqa: BLE001 - chceme pokračovat dalšími
            print(f"Přeskočeno {model['handle']}: {exc}")
            results.append((dest, False))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    return results
