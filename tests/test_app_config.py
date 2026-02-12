from pathlib import Path
import os

from app_config import get_unified_vector_db_path, load_project_env


def test_get_unified_vector_db_path():
    parent = r"C:\repo\study-sphere"
    path = get_unified_vector_db_path(parent)
    parts = Path(path).parts
    assert parts[-2:] == ("vector_db", "class_12_unified_vector_db")


def test_load_project_env_prefers_src_env(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (tmp_path / ".env").write_text("DEVICE=cuda\n", encoding="utf-8")
    (src_dir / ".env").write_text("DEVICE=cpu\n", encoding="utf-8")

    monkeypatch.delenv("DEVICE", raising=False)
    parent = load_project_env(str(src_dir))

    assert Path(parent) == tmp_path
    # src/.env should override parent .env
    assert os.getenv("DEVICE") == "cpu"
