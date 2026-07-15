from config import Config, DataConfig
import cli


def _prepared(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "rur.txt").write_text(
        "Roboty vyráběla firma Rossumovy univerzální roboty. "
        "Starý Rossum je vynalezl.",
        encoding="utf-8",
    )
    return Config(data=DataConfig(processed_dir=str(proc)))


def test_cmd_ask_formats_answer(tmp_path):
    cfg = _prepared(tmp_path)
    out = cli.cmd_ask(cfg, "kdo vyráběl roboty")
    assert "roboty" in out.lower()
    assert "zdroj" in out.lower()


def test_cmd_explain_known():
    out = cli.cmd_explain("retriever")
    assert "BM25" in out or "TF-IDF" in out


def test_main_ask_smoke(tmp_path, capsys):
    cfg = _prepared(tmp_path)
    cli.main(["ask", "kdo vyráběl roboty", "--processed-dir", cfg.data.processed_dir])
    captured = capsys.readouterr()
    assert "roboty" in captured.out.lower()


def test_reindex_builds_saves_and_answers(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kniha.txt").write_text(
        "Roboty vyráběla firma Rossumovy univerzální roboty. "
        "Starý Rossum je vynalezl.",
        encoding="utf-8",
    )
    cfg = Config(data=DataConfig(
        raw_dir=str(raw),
        processed_dir=str(tmp_path / "processed"),
        index_path=str(tmp_path / "index.pkl"),
    ))
    n = cli.cmd_reindex(cfg)
    assert n > 0
    assert (tmp_path / "processed" / "kniha.txt").exists()
    assert (tmp_path / "index.pkl").exists()
    # Následný dotaz teď načte uložený index a odpoví.
    out = cli.cmd_ask(cfg, "kdo vyráběl roboty")
    assert "roboty" in out.lower()
