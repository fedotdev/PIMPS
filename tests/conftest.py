"""Общие фикстуры для тестов PIMPS."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def write_yaml(tmp_path: Path):
    """Фикстура-фабрика: записывает YAML-строку во временный файл и возвращает Path."""

    def _write(content: str, filename: str = "station.yaml") -> Path:
        p = tmp_path / filename
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _write
