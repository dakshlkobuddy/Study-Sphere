from pathlib import Path

import chatbot_utility


def test_get_chapter_list_sorts_numeric_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(chatbot_utility, "parent_dir", str(tmp_path))
    physics_dir = tmp_path / "data" / "class_12" / "physics"
    physics_dir.mkdir(parents=True)

    (physics_dir / "10. Chapter Ten.pdf").write_text("x", encoding="utf-8")
    (physics_dir / "2. Chapter Two.pdf").write_text("x", encoding="utf-8")
    (physics_dir / "1. Chapter One.pdf").write_text("x", encoding="utf-8")
    (physics_dir / "readme.txt").write_text("ignore", encoding="utf-8")

    chapters = chatbot_utility.get_chapter_list("Physics")
    assert chapters == ["1. Chapter One", "2. Chapter Two", "10. Chapter Ten"]


def test_get_chapter_list_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(chatbot_utility, "parent_dir", str(tmp_path))
    chapters = chatbot_utility.get_chapter_list("Physics")
    assert chapters == []

