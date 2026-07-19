"""Příkazová řádka — jedno rozhraní k celé knihovně.

Sešívá bloky do příkazů, kterými se dá projít celý životní cyklus:
`prepare-data` (bootstrap dat), `reindex` (přegenerovat index po přidání textu),
`build-index` (jen postavit index z hotových dat), `ask` (jednorázový dotaz),
`repl` (interaktivní prompt) a `explain` (nechat si vysvětlit blok). Cíl je, aby
se dal projekt ovládat bez psaní jediného řádku Pythonu — a přes wrapper `./jelly`
i bez ruční aktivace venv.
"""

# Příkazy lazy-importují těžké bloky (rychlý start, volitelné závislosti jako
# viewBase); to je záměr, ne chyba.
# pylint: disable=import-outside-toplevel
import argparse
import os
import shutil

from config import Config, DataConfig
from dataprep.download import download_books
from dataprep.clean import build_processed
from jellyai.pipeline import QAPipeline
from jellyai.explain import explain_block, list_blocks
from qagen.build import build_dataset
from qagen.download_models import download_models

# Původní R.U.R. je už v repu (dědictví po LSTM verzi). Použijeme ho jako výchozí
# korpus, aby QA fungovalo hned po `prepare-data` i bez připojení k internetu.
_SEED_RUR = "training_text/karel_capek_rur.txt"


def _drain_lines(path):
    """Přečte a VYPRÁZDNÍ souborovou schránku (DEV most webu).

    Řádky připsané do `data/web_inbox.txt` zpracuje web, jako by je uživatel
    napsal do dialogového okna — testovací dotazy (skript, echo) jsou tak
    vidět v GUI včetně aktivace a růstu grafu. Chybějící soubor = žádné.

    Returns:
        list[str]: Neprázdné řádky v pořadí zápisu.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r+", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
        fh.seek(0)
        fh.truncate()
    return lines


def _build_and_save_index(config):
    """Postaví index z vyčištěných textů a uloží ho na disk.

    Sdílený krok „vygeneruj vektory": z processed adresáře se postaví retriever
    a uloží do `index_path`, aby dotazování později naskočilo okamžitě.

    Args:
        config (Config): Konfigurace (čte processed_dir, index_path, nastavení bloků).

    Returns:
        int: Počet zaindexovaných pasáží.
    """
    pipe = QAPipeline.from_corpus(config.data.processed_dir, config)
    pipe.retriever.save(config.data.index_path)
    n = len(pipe.retriever)
    print(f"Index postaven a uložen: {n} jednotek → {config.data.index_path}")
    return n


def _load_pipeline(config):
    """Načte pipeline — přednostně z uloženého indexu, jinak ji postaví.

    Když existuje uložený index, načte se bleskově z disku; když ne (třeba se
    zapomnělo na `reindex`), pipeline se poctivě postaví z processed textů, aby
    dotaz nespadl.

    Args:
        config (Config): Konfigurace s cestami a nastavením bloků.

    Returns:
        QAPipeline: Pipeline připravená odpovídat.
    """
    if os.path.exists(config.data.index_path):
        return QAPipeline.from_index(config.data.index_path, config)
    return QAPipeline.from_corpus(config.data.processed_dir, config)


def cmd_prepare_data(config):
    """Bootstrap dat: stáhne knihy, naseeduje R.U.R., vyčistí a zaindexuje.

    Kompletní „nastartuj projekt" příkaz. Nejdřív zkusí stáhnout nakonfigurované
    knihy, přidá původní R.U.R. (pojistka, ať je vždy z čeho odpovídat), a pak
    vyčistí a postaví index.

    Args:
        config (Config): Konfigurace s cestami a seznamem knih.

    Returns:
        int: Počet zaindexovaných pasáží.
    """
    os.makedirs(config.data.raw_dir, exist_ok=True)
    download_books(config)
    if os.path.exists(_SEED_RUR):
        shutil.copyfile(_SEED_RUR, os.path.join(config.data.raw_dir, "rur.txt"))
    return cmd_reindex(config)


def cmd_reindex(config):
    """Přegeneruje index po ruční změně textů v data/raw.

    Přesně ten krok, který uživatel spustí po vhození nového `.txt` do
    `data/raw/`: vyčistí syrové texty do processed a postaví/uloží nad nimi index.
    Nestahuje ani neseeduje — pracuje čistě s tím, co je v raw adresáři.

    Args:
        config (Config): Konfigurace s cestami a nastavením bloků.

    Returns:
        int: Počet zaindexovaných pasáží.
    """
    written = build_processed(config.data.raw_dir, config.data.processed_dir)
    print(f"Vyčištěno {len(written)} textů z {config.data.raw_dir}")
    return _build_and_save_index(config)


def cmd_build_index(config):
    """Postaví a uloží index z už vyčištěných textů (bez čištění).

    Args:
        config (Config): Konfigurace (čte processed_dir a nastavení bloků).

    Returns:
        int: Počet zaindexovaných pasáží.
    """
    return _build_and_save_index(config)


def cmd_ask(config, question):
    """Odpoví na jeden dotaz a naformátuje odpověď i se zdrojem.

    Args:
        config (Config): Konfigurace s cestami a nastavením bloků.
        question (str): Dotaz uživatele v češtině.

    Returns:
        str: Odpověď a pod ní řádek se zdrojem (nebo pomlčka, když zdroj není).
    """
    pipe = _load_pipeline(config)
    ans = pipe.ask(question)
    src = ", ".join(ans.sources) if ans.sources else "—"
    return f"{ans.text}\n(zdroj: {src})"


def cmd_repl(config):
    """Spustí interaktivní prompt na dotazování.

    Index se načte jednou na začátku a pak se ve smyčce ptá pořád dokola — takže
    v rámci jednoho sezení se nic nepřestavuje a odpovědi chodí okamžitě. Ukončí
    se prázdným řádkem, slovem „konec"/„exit"/„quit" nebo Ctrl-D.

    Args:
        config (Config): Konfigurace s cestami a nastavením bloků.

    Returns:
        None: Interakce probíhá na standardním vstupu/výstupu.
    """
    pipe = _load_pipeline(config)
    print("Ptej se česky. Prázdný řádek nebo 'konec' ukončí.\n")
    while True:
        try:
            question = input("❓ ").strip()
        except EOFError:  # Ctrl-D
            break
        if not question or question.lower() in {"konec", "exit", "quit"}:
            break
        ans = pipe.ask(question)
        src = ", ".join(ans.sources) if ans.sources else "—"
        print(f"💬 {ans.text}")
        print(f"   (zdroj: {src})\n")
    print("Měj se! 👋")


def cmd_explain(name):
    """Vrátí popis jednoho bloku pro výpis na terminál.

    Args:
        name (str): Název bloku (viz `list_blocks`).

    Returns:
        str: Popis bloku.
    """
    return explain_block(name)


def cmd_qa_models(config):
    """Stáhne ÚFAL modely (MorphoDiTa + NameTag) pro generování QA dat.

    Args:
        config (Config): Konfigurace (rezervováno pro cesty modelů).

    Returns:
        list[tuple[str, bool]]: Výsledek stažení každého modelu.
    """
    return download_models(config)


def cmd_wiki(config):
    """Stáhne kurátorované české wiki články do raw adresáře.

    Args:
        config (Config): Konfigurace (wiki_titles, raw_dir).

    Returns:
        list[str]: Cesty ke staženým souborům.
    """
    from dataprep.wiki import fetch_articles
    return fetch_articles(list(config.data.wiki_titles), config.data.raw_dir)


def cmd_gen_qa(config, tagger=None):
    """Vygeneruje syntetický QA dataset z korpusu.

    Args:
        config (Config): Konfigurace (processed_dir, chunker, qagen).
        tagger (Tagger | None): Analyzátor; když None, vytvoří se UfalTagger
            z modelů v konfiguraci. Injektování usnadňuje testy (FakeTagger).

    Returns:
        int: Počet vygenerovaných párů.
    """
    if tagger is None:
        from qagen.tagger import UfalTagger
        tagger = UfalTagger(config.qagen.nametag_model)
    pairs = build_dataset(config, tagger)
    print(f"Vygenerováno {len(pairs)} QA párů → {config.qagen.qa_path}")
    return len(pairs)


def cmd_annotate(config, client=None):
    """Offline anotace vět dokumentů (entity + syntaktický rozbor) k indexu (V3/B1).

    Načte dokumenty z processed adresáře, přes ÚFAL služby je po větách obohatí o
    entity a role a uloží do sidecaru (klíč = doc_id + index věty). Dotazování
    v režimu "template" pak jen čte a skládá anotaci pasáže z rozsahu jejích vět —
    funguje pro chunkerová i ostřicí okna větného retrieveru.

    Args:
        config (Config): Konfigurace (processed_dir, services).
        client: ÚFAL klient; None = vytvoří UfalClient (injektování usnadňuje testy).

    Returns:
        int: Počet anotovaných vět.
    """
    from jellyai.tasks import annotate_corpus
    count = annotate_corpus(config, client)
    print(f"Anotováno {count} vět → {config.services.annotations_path}")
    return count


def cmd_graph(config, view=False):
    """Postaví reifikovaný faktový graf z větných anotací a uloží ho.

    Načte anotace (`services.annotations_path`), poskládá vážený faktový graf
    (fakty = uzly, role-hrany, váha = opakování) a uloží do `graph.graph_path`.
    S `view=True` navíc exportuje do viewBase.

    Args:
        config (Config): Konfigurace (annotations_path, graph_path).
        view (bool): Zda po postavení exportovat do viewBase.

    Returns:
        int: Počet entitních uzlů grafu.
    """
    from jellyai.tasks import build_fact_graph
    graph = build_fact_graph(config)
    print(f"Faktový graf: {len(graph.nodes)} uzlů, {len(graph.facts)} faktů "
          f"→ {config.graph.graph_path}")
    if view:
        from jellyai.graph.viewbase_export import to_networkx
        try:
            from viewbase import Canvas, serve
            serve(Canvas.from_networkx(to_networkx(graph)))   # serve je modulová funkce
        except ImportError:
            print("viewBase/networkx není k dispozici — přeskočeno.")
    return len(graph.nodes)


def cmd_web(config, view=None, client=None):
    """Spustí webovou vizualizaci: graf ve viewBase + dialog s Iris přes REST.

    Prohlížeč mluví s automatem Iris VÝHRADNĚ přes REST službu (jeden vstupní
    bod): odpověď, jistota, použité karty, aktivační okno uzlů i aktivních
    dokumentů jdou z API odpovědi. Tři okna: dialog (konzole), ⚡ aktivační
    okno uzlů (seřazený jas, bez dialogu) a 📄 aktivní dokumenty.

    Args:
        config (Config): Konfigurace (graf, služby).
        view: Injektovaný GraphView (None = ViewBaseView).
        client: Injektovaný Iris klient (None = IrisClient — připojí se na
            běžící službu, nebo ji nastartuje).

    Returns:
        None: Interakce běží v prohlížeči.
    """
    from jellyai.tasks import load_fact_graph
    from jellyai.viz.pulse import TracePulse
    from jellyai.answerer.template import _to_nominative
    from jellyai.iris.client import IrisClient
    from jellyai.ufal_client import UfalClient

    if client is None:
        client = IrisClient(config.services, config.graph.graph_path)
    morpho = UfalClient(config.services)   # jen popisky uzlů (nominativizace)

    _label_cache = {}

    def base_form_label(node):
        """Titulek uzlu v ZÁKLADNÍM tvaru: skloněné povrchy (geo/dílo/instituce)
        nominativizuje morfologií; osoby jsou kanonické, pojmy lemmata."""
        if node.type not in ("geo", "dílo", "institution"):
            return node.id
        if node.id not in _label_cache:
            _label_cache[node.id] = _to_nominative(node.id, morpho) or node.id
        return _label_cache[node.id]

    from jellyai.iris.subsystems.mnemos import forget_entity, remember, replay
    from jellyai.lang import current
    user_entity = current()["user_entity"]
    graph = None                     # webová kopie grafu (sdílená load i během)
    if view is None:
        # lazy import: viewBase je volitelný, jádro ho nepotřebuje
        from jellyai.viz.viewbase_view import ViewBaseView
        graph = load_fact_graph(config)
        # PAMĚŤ DO VIZUALIZACE: přehraj deník Mnemos do grafu PŘED stavbou
        # plátna (jako to dělá automat na straně Iris). Vizualizace pak ukáže
        # i fakty uživatele, místa (Topos) a časové kotvy (Chronos) — a protože
        # je přidává from_graph, nesou i ATRIBUTY (node_detail_rows/fact_detail_rows),
        # ne jen holé uzly. (Web a Iris tak mají shodný výchozí graf.)
        replay(graph, config.graph.memory_path, user_entity)
        view = ViewBaseView("jellyAI3").from_graph(graph, labeler=base_form_label)

    pulse = TracePulse()             # aktivace („context window") → jas + provoz tras

    def on_query(question):
        # temperature > 0 → fuzzy: k odpovědi i další kontext s menší vahou
        out = client.query(question, temperature=0.35)
        # ŽIVÝ RŮST GRAFU: co Iris v tomto tahu uložil (Mnemos), přehraj do
        # webové kopie grafu a přidej NOVÉ fakty na plátno TÝMŽ wrapperem jako
        # při loadu (feed_fact → uzly i s atributy). DRY: load i interakce.
        mem = out.get("memorized")
        if mem and graph is not None and hasattr(view, "feed_fact"):
            before = set(graph.facts)
            remember(graph, mem, user_entity)
            for key, fact in graph.facts.items():
                if key not in before:
                    view.feed_fact(graph, fact, labeler=base_form_label)
        nodes = [tuple(n) for n in out.get("activation", {}).get("nodes", [])]
        scores = {node: float(jas) for node, jas in nodes}
        trace = out.get("trace")
        path = [trace["topic"], trace["answer"]] if trace else []
        # vizualizace zrcadlí stav aktivace po tahu: rozsvítí pole, přidá
        # trasu a kamera se vystředí na nejaktivnější uzel
        state = pulse.ignite(scores, path)
        for node_id, bright in state["sizes"].items():
            view.update_node(node_id, size=1.0 + bright,
                             **{"aktivace (attention)": f"{bright:.2f}"})
        for node_id in state["extinguish"]:   # uzly mimo pole → zpět na základ
            view.update_node(node_id, size=1.0,
                             **{"aktivace (attention)": "0"})
        if state["sizes"] and hasattr(view, "focus"):
            view.focus(max(state["sizes"], key=state["sizes"].get))
        if hasattr(view, "write_docs"):       # 📄 aktivní dokumenty
            view.write_docs([tuple(d) for d in
                             out.get("activation", {}).get("docs", [])])
        if hasattr(view, "write_nodes"):      # ⚡ aktivační okno uzlů
            view.write_nodes(nodes)
        used = out.get("used", {})
        reply = f"❓ {question}\n💬 {out['answer']}"
        if used.get("patterns"):
            reply += f"\n   karty: {', '.join(used['patterns'])}"
        if out.get("alternatives"):  # souvislosti (krmivo pro budoucí kompozitor)
            reply += f"\n   souvislosti: {', '.join(out['alternatives'][:4])}"
        view.write(reply)            # odpověď (+ metadata) v prohlížeči
        # ŽIVÉ ZAPOMENUTÍ (AŽ PO aktivaci — jinak by update_node smazané uzly
        # znovu oživil): „zapomeň na Ronika" → odeber entitu i její fakty
        # z plátna přes remove akce, symetricky k feed_fact.
        forgotten = out.get("forgotten")
        if forgotten and graph is not None and hasattr(view, "remove_facts"):
            before_facts = set(graph.facts)
            before_nodes = set(graph.nodes)
            forget_entity(graph, None, forgotten, user_entity)   # web graf, bez deníku
            view.remove_facts([k for k in before_facts if k not in graph.facts])
            for nid in before_nodes - set(graph.nodes):
                view.remove_node(nid)
        # --- debug výpis konverzace (do logu serveru; flush kvůli RT) ---
        top = ", ".join(f"{n}={v:.2f}" for n, v in nodes[:6])
        print(
            f"\n❓ {question}"
            f"\n💬 [{out.get('kind')}, assurance {out.get('assurance', 0.0):.2f}] "
            f"{out['answer']}"
            f"\n   karty: {', '.join(used.get('patterns', ())) or '—'}"
            f"   komponenty: {', '.join(used.get('components', ())) or '—'}"
            f"\n   aktivace: {top or '—'}",
            flush=True)
        return out["answer"]

    def animate():
        # ~7×/s: každá aktivní trasa vyšle paket podle své hustoty (∝ aktivaci)
        for packet_path in pulse.tick()["packets"]:
            view.packet(packet_path)

    view.every(0.15, animate)
    view.open_terminal(on_query)     # okno 1: dialog (vstup i výstup)

    # DEV MOST: řádky ze souborové schránky jdou TOUŽ cestou jako vstup
    # z dialogového okna (on_query) — testování skriptem je vidět v GUI
    inbox_path = os.path.join(
        os.path.dirname(config.graph.memory_path) or "data", "web_inbox.txt")

    def drain_inbox():
        for line in _drain_lines(inbox_path):
            try:
                on_query(line)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"⚠️ inbox dotaz selhal: {exc}", flush=True)

    view.every(1.0, drain_inbox)
    if hasattr(view, "on_event"):
        # most pro CHRONOS: služba Iris pushne dozrálou připomínku REST
        # eventem — řádek v konzoli (terminal_write) a STATICKÉ okno
        # (reminder_window), které visí, dokud ho uživatel nezavře;
        # web je pasivní displej, iniciátorem je tep Chronos
        view.on_event("terminal_write", lambda e: view.write(e.line))
        view.on_event("reminder_window", lambda e: view.open_reminder(e.line))
    if hasattr(view, "open_docs_panel"):
        view.open_docs_panel()       # okno 3: aktivní dokumenty
    if hasattr(view, "open_nodes_panel"):
        view.open_nodes_panel()      # okno 2: aktivační okno uzlů
    view.serve(open_browser=False)   # server (systemd) — žádný prohlížeč lokálně


def _build_parser():
    """Sestaví argparse parser se všemi příkazy.

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
    common.add_argument("--graph", action="store_true",
                        help="odpovídat přes faktový graf (mode=graph)")
    common.add_argument("--template", action="store_true",
                        help="použít pravidlový (template) answerer V3 místo extraktivního")

    parser = argparse.ArgumentParser(prog="cli", description="Český QA nad texty (V1)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data", parents=[common], help="stáhne + seeduje + vyčistí + zaindexuje")
    sub.add_parser("reindex", parents=[common], help="přegeneruje index z data/raw")
    sub.add_parser("build-index", parents=[common], help="postaví index z hotových textů")
    sub.add_parser("repl", parents=[common], help="interaktivní prompt na dotazování")
    sub.add_parser("qa-models", parents=[common], help="stáhne ÚFAL modely (MorphoDiTa+NameTag)")
    sub.add_parser("wiki", parents=[common], help="stáhne české wiki články do data/raw")
    sub.add_parser("gen-qa", parents=[common], help="vygeneruje syntetický QA dataset z korpusu")
    sub.add_parser("annotate", parents=[common], help="offline anotace pasáží (entity+role) pro V3")
    p_graph = sub.add_parser("graph", parents=[common], help="postaví faktový graf z anotací")
    p_graph.add_argument("--view", action="store_true", help="export do viewBase")
    sub.add_parser("web", parents=[common], help="webová vizualizace grafu + prompt (viewBase)")
    p_ask = sub.add_parser("ask", parents=[common], help="odpoví na jeden dotaz")
    p_ask.add_argument("question", help="otázka v češtině")
    p_explain = sub.add_parser("explain", parents=[common], help="popíše blok")
    p_explain.add_argument("block", nargs="?",
                           help=f"jeden z: {', '.join(list_blocks())}")
    return parser


def main(argv=None):  # pylint: disable=too-many-branches
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
        # Vlastní processed adresář = vlastní korpus → i vlastní index vedle něj,
        # ať se omylem nenačte uložený index jiného (výchozího) korpusu.
        config = Config(data=DataConfig(
            processed_dir=args.processed_dir,
            index_path=os.path.join(args.processed_dir, "index.pkl"),
        ))
    if getattr(args, "template", False):
        config.answerer.mode = "template"  # přepnout na pravidlový answerer (V3)
    if getattr(args, "graph", False):
        config.answerer.mode = "graph"     # přepnout na faktový graf

    if args.command == "prepare-data":
        cmd_prepare_data(config)
    elif args.command == "reindex":
        cmd_reindex(config)
    elif args.command == "build-index":
        cmd_build_index(config)
    elif args.command == "repl":
        cmd_repl(config)
    elif args.command == "qa-models":
        cmd_qa_models(config)
    elif args.command == "wiki":
        cmd_wiki(config)
    elif args.command == "gen-qa":
        cmd_gen_qa(config)
    elif args.command == "annotate":
        cmd_annotate(config)
    elif args.command == "graph":
        cmd_graph(config, view=args.view)
    elif args.command == "web":
        cmd_web(config)
    elif args.command == "ask":
        print(cmd_ask(config, args.question))
    elif args.command == "explain":
        if not args.block:
            print("Dostupné bloky: " + ", ".join(list_blocks()))
        else:
            print(cmd_explain(args.block))


if __name__ == "__main__":
    main()
