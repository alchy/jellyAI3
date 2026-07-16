"""Stažení českých ÚFAL modelů z repozitáře LINDAT (DSpace 7).

Potřebujeme tři modely: NameTag (entity), MorphoDiTa (morfologie + skloňování) a
UDPipe (syntaktický rozbor). LINDAT běží na DSpace 7, kde se k souboru dostaneš
přes REST API z trvalého „handle". Některé položky mají model zabalený v ZIPu
(vytáhneme z něj `.tagger`/`.ner`), jiné nabízejí model jako samostatný bitstream
(UDPipe — stáhneme přímo). Modely jsou CC BY-NC-SA (nekomerční, pro osobní OK).
Stahování je best-effort: co selže, ohlásí a jede dál.
"""

import fnmatch
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

_API = "https://lindat.mff.cuni.cz/repository/server/api"

MODELS = [
    # NameTag CNEC (entity) — v ZIPu
    {"handle": "11858/00-097C-0000-0023-7D42-8", "member": "*.ner",
     "dest": "data/models/czech-cnec.ner"},
    # MorphoDiTa MorfFlex/PDT-C (morfologie + generování tvarů) — v ZIPu
    {"handle": "11234/1-4794", "member": "*.tagger",
     "dest": "data/models/czech-morfflex.tagger"},
    # UDPipe český PDT (syntaktický rozbor) — přímý bitstream
    {"handle": "11234/1-3131", "bitstream": "czech-pdt-ud-2.5-191206.udpipe",
     "dest": "data/models/udpipe-czech.model"},
]


def _get_json(url):
    """Stáhne a rozparsuje JSON z DSpace REST API."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _bitstreams(handle):
    """Vrátí soubory (bitstreamy) položky z bundlu ORIGINAL.

    Args:
        handle (str): Trvalý handle položky.

    Returns:
        list[dict]: [{name, url}] pro každý bitstream.
    """
    item = _get_json(f"{_API}/pid/find?id=hdl:{handle}")
    bundles = _get_json(f"{_API}/core/items/{item['uuid']}/bundles")
    out = []
    for bundle in bundles["_embedded"]["bundles"]:
        if bundle["name"] != "ORIGINAL":
            continue
        listing = _get_json(bundle["_links"]["bitstreams"]["href"] + "?size=500")
        for stream in listing["_embedded"]["bitstreams"]:
            out.append({"name": stream["name"],
                        "url": stream["_links"]["content"]["href"]})
    return out


def _find_member(zf, pattern):
    """Najde v ZIPu největší soubor odpovídající vzoru (plný model)."""
    matches = [i for i in zf.infolist()
               if fnmatch.fnmatch(os.path.basename(i.filename), pattern)]
    if not matches:
        raise KeyError(f"V ZIPu není soubor odpovídající {pattern}")
    return max(matches, key=lambda i: i.file_size).filename


def _download_zip_member(url, member_pattern, dest):
    """Stáhne ZIP a vytáhne z něj soubor odpovídající vzoru do dest."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        urllib.request.urlretrieve(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as zf:
            member = _find_member(zf, member_pattern)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def download_models(config=None, models=MODELS):
    """Stáhne (a případně rozbalí) modely do data/models.

    Podporuje dva typy položek: `bitstream` (stáhne konkrétní soubor přímo) a
    `member` (stáhne ZIP a vytáhne z něj soubor podle vzoru).

    Args:
        config: Nepovinné (rezervováno pro budoucí cesty z konfigurace).
        models (list): Položky {handle, dest, (bitstream|member)}.

    Returns:
        list[tuple[str, bool]]: (cílová cesta, úspěch) pro každý model.
    """
    results = []
    for model in models:
        dest = model["dest"]
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            streams = _bitstreams(model["handle"])
            if "bitstream" in model:
                url = next(s["url"] for s in streams if s["name"] == model["bitstream"])
                print(f"Stahuji {model['bitstream']} …")
                urllib.request.urlretrieve(url, dest)
            else:
                url = next(s["url"] for s in streams
                           if s["name"].lower().endswith(".zip"))
                print(f"Stahuji {model['handle']} (ZIP) …")
                _download_zip_member(url, model["member"], dest)
            print(f"Hotovo: {dest}")
            results.append((dest, True))
        except Exception as exc:  # noqa: BLE001 - pokračuj dalšími modely
            print(f"Přeskočeno {model['handle']}: {exc}")
            results.append((dest, False))
    return results
