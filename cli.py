"""Příkazová řádka — jedno rozhraní k celé knihovně.

Sešívá bloky do čtyř příkazů, kterými se dá projít celý životní cyklus:
`prepare-data` (obstarat texty), `build-index` (sanity check indexu),
`ask` (zeptat se) a `explain` (nechat si vysvětlit blok). Cíl je, aby se dal
projekt vyzkoušet bez psaní jediného řádku Pythonu — jen z terminálu.
"""

import argparse
import os
import shutil

from config import Config, DataConfig
from dataprep.download import download_books
from dataprep.clean import build_processed
from jellyai.pipeline import QAPipeline
from jellyai.explain import explain_block, list_blocks

# Původní R.U.R. je už v repu (dědictví po LSTM verzi). Použijeme ho jako výchozí
# korpus, aby QA fungovalo hned po `prepare-data` i bez připojení k internetu.
_SEED_RUR = "training_text/karel_capek_rur.txt"


def cmd_prepare_data(config):
    """Obstará a připraví korpus: stáhne knihy, naseeduje R.U.R., vyčistí.

    Nejdřív zkusí stáhnout nakonfigurované knihy, pak zkopíruje původní R.U.R.
    do raw adresáře (pojistka, ať je vždy z čeho odpovídat) a nakonec všechny
    syrové texty vyčistí do processed adresáře.

    Args:
        config (Config): Konfigurace s cestami a seznamem knih.

    Returns:
        list[str]: Cesty k vyčištěným textům připraveným k indexaci.
    """
    os.makedirs(config.data.raw_dir, exist_ok=True)
    download_books(config)
    if os.path.exists(_SEED_RUR):
        shutil.copyfile(_SEED_RUR, os.path.join(config.data.raw_dir, "rur.txt"))
    written = build_processed(config.data.raw_dir, config.data.processed_dir)
    print(f"Připraveno {len(written)} textů v {config.data.processed_dir}")
    return written


def cmd_build_index(config):
    """Postaví index a vypíše, kolik pasáží vzniklo (rychlá kontrola dat).

    Args:
        config (Config): Konfigurace (čte se processed_dir a nastavení bloků).

    Returns:
        int: Počet zaindexovaných pasáží.
    """
    pipe = QAPipeline.from_corpus(config.data.processed_dir, config)
    n = len(pipe.retriever.passages)
    print(f"Index postaven: {n} pasáží z {config.data.processed_dir}")
    return n


def cmd_ask(config, question):
    """Odpoví na jeden dotaz a naformátuje odpověď i se zdrojem.

    Args:
        config (Config): Konfigurace (čte se processed_dir a nastavení bloků).
        question (str): Dotaz uživatele v češtině.

    Returns:
        str: Odpověď a pod ní řádek se zdrojem (nebo pomlčka, když zdroj není).
    """
    pipe = QAPipeline.from_corpus(config.data.processed_dir, config)
    ans = pipe.ask(question)
    src = ", ".join(ans.sources) if ans.sources else "—"
    return f"{ans.text}\n(zdroj: {src})"


def cmd_explain(name):
    """Vrátí popis jednoho bloku pro výpis na terminál.

    Args:
        name (str): Název bloku (viz `list_blocks`).

    Returns:
        str: Popis bloku.
    """
    return explain_block(name)


def _build_parser():
    """Sestaví argparse parser se čtyřmi příkazy.

    Sdílený přepínač `--processed-dir` je připojen ke každému podpříkazu přes
    společný „rodičovský" parser, takže funguje i napsaný ZA názvem příkazu
    (třeba `ask "..." --processed-dir X`). Bez tohoto triku by argparse takový
    přepínač za podpříkazem odmítal.

    Returns:
        argparse.ArgumentParser: Připravený parser.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--processed-dir", default=None,
                        help="adresář s vyčištěnými texty (výchozí data/processed)")

    parser = argparse.ArgumentParser(prog="cli", description="Český QA nad texty (V1)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data", parents=[common], help="stáhne + vyčistí texty")
    sub.add_parser("build-index", parents=[common], help="postaví index a vypíše statistiku")
    p_ask = sub.add_parser("ask", parents=[common], help="odpoví na dotaz")
    p_ask.add_argument("question", help="otázka v češtině")
    p_explain = sub.add_parser("explain", parents=[common], help="popíše blok")
    p_explain.add_argument("block", nargs="?",
                           help=f"jeden z: {', '.join(list_blocks())}")
    return parser


def main(argv=None):
    """Vstupní bod CLI — rozparsuje argumenty a spustí zvolený příkaz.

    Args:
        argv (list[str] | None): Argumenty (bez názvu programu). None = vezmou
            se ze `sys.argv`. Předávání seznamu usnadňuje testování.

    Returns:
        None: Výstup jde na standardní výstup (print).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = Config()
    if args.processed_dir:
        config = Config(data=DataConfig(processed_dir=args.processed_dir))

    if args.command == "prepare-data":
        cmd_prepare_data(config)
    elif args.command == "build-index":
        cmd_build_index(config)
    elif args.command == "ask":
        print(cmd_ask(config, args.question))
    elif args.command == "explain":
        if not args.block:
            print("Dostupné bloky: " + ", ".join(list_blocks()))
        else:
            print(cmd_explain(args.block))


if __name__ == "__main__":
    main()
