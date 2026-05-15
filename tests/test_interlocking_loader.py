"""Тесты для src/interlocking/loader.py — загрузка YAML-конфигурации станции."""
from __future__ import annotations

import math

import pytest

from src.interlocking.loader import StationConfigError, load_station
from src.models import RouteType, SwitchPosition


# ---------------------------------------------------------------------------
# Минимальный валидный YAML (1 стрелка + 1 маршрут)
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
station_id: ST1
name: Тестовая
switches:
  - switch_id: SW1
    normal_position: plus
    transfer_time_s: 4.0
routes:
  - route_id: R1
    name: Маршрут 1
    route_type: arrival
    sections:
      - SEC_A
    switches:
      SW1: plus
    v_limit: 40.0
"""


class TestLoadStationSuccess:
    """Успешная загрузка простейших конфигураций."""

    def test_minimal_station(self, write_yaml):
        """1 стрелка + 1 маршрут — минимальный валидный конфиг."""
        path = write_yaml(MINIMAL_YAML)
        config = load_station(path)

        assert config.station_id == "ST1"
        assert config.name == "Тестовая"

        assert "SW1" in config.switches
        sw = config.switches["SW1"]
        assert sw.normal_position is SwitchPosition.PLUS
        assert sw.transfer_time_s == 4.0

        assert "R1" in config.routes
        route = config.routes["R1"]
        assert route.route_type is RouteType.ARRIVAL
        assert route.sections == ["SEC_A"]
        assert route.switches == {"SW1": SwitchPosition.PLUS}
        assert route.v_limit == 40.0

        assert config.extra_conflicts == []

    def test_two_routes_no_extra_conflicts(self, write_yaml):
        """Два непересекающихся маршрута, extra_conflicts отсутствует."""
        yaml_text = """\
        station_id: ST2
        name: Двойная
        switches:
          - switch_id: SW1
            normal_position: plus
          - switch_id: SW2
            normal_position: minus
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches:
              SW1: plus
          - route_id: R2
            route_type: departure
            sections: [SEC_B]
            switches:
              SW2: minus
        """
        config = load_station(write_yaml(yaml_text))

        assert len(config.switches) == 2
        assert len(config.routes) == 2
        assert config.extra_conflicts == []

    def test_extra_conflicts_parsed(self, write_yaml):
        """extra_conflicts корректно разбираются."""
        yaml_text = """\
        station_id: ST3
        name: С конфликтами
        switches:
          - switch_id: SW1
            normal_position: plus
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches: {SW1: plus}
          - route_id: R2
            route_type: departure
            sections: [SEC_B]
            switches: {SW1: minus}
        extra_conflicts:
          - [R1, R2]
        """
        config = load_station(write_yaml(yaml_text))

        assert len(config.extra_conflicts) == 1
        assert config.extra_conflicts[0] == ("R1", "R2")

    def test_joints_intervals_parsed(self, write_yaml):
        """Нормативные интервалы ПТР на стыках корректно разбираются."""
        yaml_text = """\
        station_id: ST_J
        name: Со стыками
        joints:
          - id: ep_A
            kind: entry_point
            interval_base_min: 8.0
            interval_vc_min: 4.5
        switches: []
        routes: []
        """
        config = load_station(write_yaml(yaml_text))

        joint = config.joints["ep_A"]
        assert joint.kind == "entry_point"
        assert joint.interval_base_min == 8.0
        assert joint.interval_vc_min == 4.5

    def test_default_values(self, write_yaml):
        """Значения по умолчанию: transfer_time_s=4.0, v_limit=60.0."""
        yaml_text = """\
        station_id: ST4
        name: Дефолты
        switches:
          - switch_id: SW1
            normal_position: plus
        routes:
          - route_id: R1
            route_type: passthrough
            sections: [SEC_A]
            switches: {}
        """
        config = load_station(write_yaml(yaml_text))

        assert config.switches["SW1"].transfer_time_s == 4.0
        assert config.routes["R1"].v_limit == 60.0

    def test_accepts_string_path(self, write_yaml):
        """load_station принимает str, а не только Path."""
        path = write_yaml(MINIMAL_YAML)
        config = load_station(str(path))
        assert config.station_id == "ST1"


# ---------------------------------------------------------------------------
# Ошибки — стрелки
# ---------------------------------------------------------------------------

class TestSwitchErrors:
    """Ошибки валидации блока switches."""

    def test_duplicate_switch_id(self, write_yaml):
        """Дублирующийся switch_id → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - switch_id: SW1
            normal_position: plus
          - switch_id: SW1
            normal_position: minus
        routes: []
        """
        with pytest.raises(StationConfigError, match="Дублирующийся switch_id 'SW1'"):
            load_station(write_yaml(yaml_text))

    def test_missing_switch_id(self, write_yaml):
        """Стрелка без switch_id → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - normal_position: plus
        routes: []
        """
        with pytest.raises(StationConfigError, match="отсутствует поле 'switch_id'"):
            load_station(write_yaml(yaml_text))

    def test_invalid_normal_position(self, write_yaml):
        """Неизвестная позиция стрелки → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - switch_id: SW1
            normal_position: левая
        routes: []
        """
        with pytest.raises(StationConfigError, match="неизвестная позиция"):
            load_station(write_yaml(yaml_text))

    def test_negative_transfer_time(self, write_yaml):
        """Отрицательное transfer_time_s → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - switch_id: SW1
            normal_position: plus
            transfer_time_s: -1.0
        routes: []
        """
        with pytest.raises(StationConfigError, match="transfer_time_s"):
            load_station(write_yaml(yaml_text))

    def test_zero_transfer_time(self, write_yaml):
        """transfer_time_s = 0 → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - switch_id: SW1
            normal_position: plus
            transfer_time_s: 0
        routes: []
        """
        with pytest.raises(StationConfigError, match="transfer_time_s"):
            load_station(write_yaml(yaml_text))


# ---------------------------------------------------------------------------
# Ошибки — маршруты
# ---------------------------------------------------------------------------

class TestRouteErrors:
    """Ошибки валидации блока routes."""

    def test_duplicate_route_id(self, write_yaml):
        """Дублирующийся route_id → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - switch_id: SW1
            normal_position: plus
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches: {SW1: plus}
          - route_id: R1
            route_type: departure
            sections: [SEC_B]
            switches: {SW1: minus}
        """
        with pytest.raises(StationConfigError, match="Дублирующийся route_id 'R1'"):
            load_station(write_yaml(yaml_text))

    def test_duplicate_sections_in_route(self, write_yaml):
        """Дублирующиеся секции внутри одного маршрута → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A, SEC_B, SEC_A]
            switches: {}
        """
        with pytest.raises(StationConfigError, match="дублирующиеся секции"):
            load_station(write_yaml(yaml_text))

    def test_empty_sections(self, write_yaml):
        """Пустой список секций → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: arrival
            sections: []
            switches: {}
        """
        with pytest.raises(StationConfigError, match="не может быть пустым"):
            load_station(write_yaml(yaml_text))

    def test_unknown_switch_in_route(self, write_yaml):
        """Маршрут ссылается на несуществующую стрелку → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches:
          - switch_id: SW1
            normal_position: plus
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches:
              SW_UNKNOWN: plus
        """
        with pytest.raises(
            StationConfigError, match="ссылка на неизвестную стрелку 'SW_UNKNOWN'"
        ):
            load_station(write_yaml(yaml_text))

    def test_invalid_route_type(self, write_yaml):
        """Неизвестный route_type → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: teleportation
            sections: [SEC_A]
            switches: {}
        """
        with pytest.raises(StationConfigError, match="неизвестный тип"):
            load_station(write_yaml(yaml_text))

    @pytest.mark.parametrize(
        "bad_vlimit",
        [0, -10, "inf", "-.inf", ".nan"],
        ids=["zero", "negative", "inf", "neg_inf", "nan"],
    )
    def test_invalid_v_limit(self, write_yaml, bad_vlimit):
        """Невалидные значения v_limit → StationConfigError."""
        yaml_text = f"""\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches: {{}}
            v_limit: {bad_vlimit}
        """
        with pytest.raises(StationConfigError, match="v_limit"):
            load_station(write_yaml(yaml_text))


# ---------------------------------------------------------------------------
# Ошибки — extra_conflicts
# ---------------------------------------------------------------------------

class TestExtraConflictErrors:
    """Ошибки валидации блока extra_conflicts."""

    def test_self_conflict(self, write_yaml):
        """Маршрут конфликтует сам с собой → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches: {}
        extra_conflicts:
          - [R1, R1]
        """
        with pytest.raises(StationConfigError, match="не может конфликтовать сам с собой"):
            load_station(write_yaml(yaml_text))

    def test_unknown_route_in_extra_conflicts(self, write_yaml):
        """Ссылка на несуществующий маршрут в extra_conflicts → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches: {}
        extra_conflicts:
          - [R1, R_GHOST]
        """
        with pytest.raises(
            StationConfigError, match="ссылка на неизвестный маршрут 'R_GHOST'"
        ):
            load_station(write_yaml(yaml_text))

    def test_duplicate_extra_conflict_pair(self, write_yaml):
        """Дублирующаяся пара (A,B) и (B,A) → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        name: Тест
        switches: []
        routes:
          - route_id: R1
            route_type: arrival
            sections: [SEC_A]
            switches: {}
          - route_id: R2
            route_type: departure
            sections: [SEC_B]
            switches: {}
        extra_conflicts:
          - [R1, R2]
          - [R2, R1]
        """
        with pytest.raises(StationConfigError, match="дублирующийся конфликт"):
            load_station(write_yaml(yaml_text))


# ---------------------------------------------------------------------------
# Ошибки — файл и YAML
# ---------------------------------------------------------------------------

class TestFileErrors:
    """Ошибки чтения файла и парсинга YAML."""

    def test_file_not_found(self, tmp_path):
        """Несуществующий файл → StationConfigError."""
        with pytest.raises(StationConfigError, match="Файл не найден"):
            load_station(tmp_path / "ghost.yaml")

    def test_invalid_yaml(self, write_yaml):
        """Синтаксически невалидный YAML → StationConfigError."""
        path = write_yaml("key: [\nunmatched", filename="bad.yaml")
        with pytest.raises(StationConfigError, match="Ошибка парсинга YAML"):
            load_station(path)

    def test_root_not_dict(self, write_yaml):
        """Корень YAML — список, а не словарь → StationConfigError."""
        path = write_yaml("- item1\n- item2\n", filename="list.yaml")
        with pytest.raises(StationConfigError, match="Корень YAML должен быть словарём"):
            load_station(path)

    def test_missing_station_id(self, write_yaml):
        """Отсутствует station_id → StationConfigError."""
        yaml_text = """\
        name: Тест
        switches: []
        routes: []
        """
        with pytest.raises(StationConfigError, match="station_id"):
            load_station(write_yaml(yaml_text))

    def test_missing_name(self, write_yaml):
        """Отсутствует name → StationConfigError."""
        yaml_text = """\
        station_id: ST1
        switches: []
        routes: []
        """
        with pytest.raises(StationConfigError, match="name"):
            load_station(write_yaml(yaml_text))
