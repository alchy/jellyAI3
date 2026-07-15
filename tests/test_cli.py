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
