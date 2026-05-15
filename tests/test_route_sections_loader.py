"""Тесты загрузки физических секций маршрута из станционного YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from src.traction.loader import load_route_sections


def _load_station_yaml() -> dict[str, Any]:
    """Читает YAML станции Миитовская как словарь."""
    path = Path(__file__).resolve().parents[1] / "stations" / "miitovskaya_station.yaml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    return cast(dict[str, Any], data)


def test_load_route_sections_uses_station_line_profile() -> None:
    """RouteSection получает уклон, радиус и порядок линий из station_yaml['lines']."""
    station_yaml = _load_station_yaml()
    line_ids = [
        "l_ep_A_sigN",
        "l_sigN_sw1",
        "l_sw1_sigN1",
        "l_sigN1_buf1P_w",
        "l_track_1P",
        "l_track_1P_sw10",
        "l_sw10_sigCH1",
        "l_sigCH1_sw2",
        "l_sw2_sigCH",
        "l_sigCH_ep_B",
    ]

    sections = load_route_sections(station_yaml, line_ids)
    sections_by_id = {section.section_id: section for section in sections}

    assert sections[0].grade == 3.0
    assert sections_by_id["l_track_1P"].grade == 0.5
    assert sections_by_id["l_track_1P"].radius == 0.0
    assert sections_by_id["l_sw1_sigN1"].radius == 0.0

    branch_sections = load_route_sections(station_yaml, ["l_sigN_sw7"])
    assert branch_sections[0].radius == 300.0
