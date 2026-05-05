from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import yaml

from src.models import (
    RouteConfig,
    RouteType,
    StationConfig,
    SwitchConfig,
    SwitchPosition,
)

__all__ = ["load_station", "ConfigError", "StationConfigError"]

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Базовый класс ошибок конфигурации."""


class StationConfigError(ConfigError):
    """Ошибка при разборе YAML-конфига станции."""


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def load_station(path: Path | str) -> StationConfig:
    """Загружает конфигурацию станции из YAML-файла.

    Args:
        path: путь к YAML-файлу конфигурации станции.

    Returns:
        Заполненный объект StationConfig.

    Raises:
        StationConfigError: если файл не найден, содержит невалидный YAML
            или нарушает логику конфигурации.
    """
    path = Path(path)
    raw = _read_yaml(path)

    station_id, name = _parse_station_meta(raw, path)

    switches = _parse_switches(raw, path)
    routes = _parse_routes(raw, switches, path)
    extra_conflicts = _parse_extra_conflicts(raw, set(routes.keys()), path)

    logger.debug(
        "Станция '%s': %d стрелок, %d маршрутов, %d доп. конфликтов",
        station_id,
        len(switches),
        len(routes),
        len(extra_conflicts),
    )

    logger.info(
        "Конфигурация станции '%s' (%s) загружена из %s",
        name,
        station_id,
        path,
    )

    return StationConfig(
        station_id=station_id,
        name=name,
        switches=switches,
        routes=routes,
        extra_conflicts=extra_conflicts,
    )


# ---------------------------------------------------------------------------
# Приватные парсеры
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """Читает YAML-файл и возвращает его корень как словарь.

    Raises:
        StationConfigError: файл не найден, невалидный YAML или корень не dict.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError:
        raise StationConfigError(f"Файл не найден: {path}")
    except yaml.YAMLError as exc:
        raise StationConfigError(
            f"Ошибка парсинга YAML в файле {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise StationConfigError(
            f"Корень YAML должен быть словарём, получено "
            f"{type(data).__name__}: {path}"
        )
    return data


def _parse_switches(
    raw: dict[str, Any],
    path: Path,
) -> dict[str, SwitchConfig]:
    """Разбирает секцию 'switches' YAML-конфига.

    Raises:
        StationConfigError: дублирующийся switch_id, неизвестная позиция,
            некорректное время перевода.
    """
    raw_switches = raw.get("switches") or []
    switches: dict[str, SwitchConfig] = {}

    for idx, item in enumerate(raw_switches):
        switch_id = item.get("switch_id") or item.get("id")
        if not switch_id:
            raise StationConfigError(
                f"Стрелка #{idx}: отсутствует поле 'switch_id'/'id' ({path})"
            )

        if switch_id in switches:
            raise StationConfigError(
                f"Дублирующийся switch_id '{switch_id}' в файле {path}"
            )

        raw_pos = item.get("normal_position") or item.get("normal") or "plus"
        try:
            normal_position = SwitchPosition(raw_pos)
        except ValueError:
            valid = [p.value for p in SwitchPosition]
            raise StationConfigError(
                f"Стрелка '{switch_id}': неизвестная позиция '{raw_pos}'. "
                f"Допустимые значения: {valid} ({path})"
            )

        transfer_time_s: float = item.get("transfer_time_s", item.get("switch_time_s", 4.0))
        if not isinstance(transfer_time_s, (int, float)):
            raise StationConfigError(
                f"Стрелка '{switch_id}': transfer_time_s должно быть числом ({path})"
            )
        transfer_time_s = float(transfer_time_s)
        if not (math.isfinite(transfer_time_s) and transfer_time_s > 0):
            raise StationConfigError(
                f"Стрелка '{switch_id}': transfer_time_s должно быть "
                f"конечным положительным числом, получено {transfer_time_s} ({path})"
            )

        switches[switch_id] = SwitchConfig(
            switch_id=switch_id,
            normal_position=normal_position,
            transfer_time_s=transfer_time_s,
        )

    logger.debug("Разобрано %d стрелок", len(switches))
    return switches


def _parse_routes(
    raw: dict[str, Any],
    known_switches: dict[str, SwitchConfig],
    path: Path,
) -> dict[str, RouteConfig]:
    """Разбирает секцию 'routes' YAML-конфига.

    Raises:
        StationConfigError: дублирующийся route_id, неизвестный тип маршрута,
            пустой список секций, дубли секций, ссылка на неизвестную стрелку.
    """
    raw_routes = raw.get("routes") or []
    routes: dict[str, RouteConfig] = {}

    for idx, item in enumerate(raw_routes):
        route_id = item.get("route_id") or item.get("id")
        if not route_id:
            raise StationConfigError(
                f"Маршрут #{idx}: отсутствует поле 'route_id'/'id' ({path})"
            )

        if route_id in routes:
            raise StationConfigError(
                f"Дублирующийся route_id '{route_id}' в файле {path}"
            )

        raw_type = item.get("route_type") or _normalize_route_type(item.get("kind"))
        try:
            route_type = RouteType(raw_type)
        except ValueError:
            valid = [t.value for t in RouteType]
            raise StationConfigError(
                f"Маршрут '{route_id}': неизвестный тип '{raw_type}'. "
                f"Допустимые значения: {valid} ({path})"
            )

        sections: list[str] = item.get("sections") or []
        if not sections:
            raise StationConfigError(
                f"Маршрут '{route_id}': список секций не может быть пустым ({path})"
            )

        dup_sections = [s for s in sections if sections.count(s) > 1]
        if dup_sections:
            raise StationConfigError(
                f"Маршрут '{route_id}': дублирующиеся секции: "
                f"{sorted(set(dup_sections))} ({path})"
            )

        raw_vlimit = item.get("v_limit", 60.0)
        v_limit = float(raw_vlimit)
        if not (math.isfinite(v_limit) and v_limit > 0):
            raise StationConfigError(
                f"Маршрут '{route_id}': v_limit должно быть конечным "
                f"положительным числом, получено {v_limit} ({path})"
            )

        route_switches = _parse_route_switches(
            item.get("switches") or {},
            route_id,
            known_switches,
            path,
        )

        name: str = item.get("name", route_id)

        routes[route_id] = RouteConfig(
            route_id=route_id,
            name=name,
            route_type=route_type,
            sections=sections,
            switches=route_switches,
            v_limit=v_limit,
        )

    logger.debug("Разобрано %d маршрутов", len(routes))
    return routes


def _parse_route_switches(
    raw_switches: dict[str, Any] | list[dict[str, Any]],
    route_id: str,
    known_switches: dict[str, SwitchConfig],
    path: Path,
) -> dict[str, SwitchPosition]:
    """Разбирает блок стрелок внутри маршрута с перекрёстной валидацией.

    Raises:
        StationConfigError: ссылка на неизвестную стрелку или неизвестная позиция.
    """
    result: dict[str, SwitchPosition] = {}
    normalized: dict[str, Any] = {}
    if isinstance(raw_switches, dict):
        normalized = raw_switches
    elif isinstance(raw_switches, list):
        for idx, item in enumerate(raw_switches):
            if not isinstance(item, dict):
                raise StationConfigError(
                    f"Маршрут '{route_id}': switches[{idx}] должен быть словарём ({path})"
                )
            sw_id = item.get("id")
            raw_pos = item.get("position")
            if not sw_id:
                raise StationConfigError(
                    f"Маршрут '{route_id}': switches[{idx}] без поля 'id' ({path})"
                )
            if not raw_pos:
                raise StationConfigError(
                    f"Маршрут '{route_id}': switches[{idx}] без поля 'position' ({path})"
                )
            normalized[sw_id] = raw_pos
    else:
        raise StationConfigError(
            f"Маршрут '{route_id}': секция 'switches' должна быть dict или list ({path})"
        )

    for sw_id, raw_pos in normalized.items():
        if sw_id not in known_switches:
            raise StationConfigError(
                f"Маршрут '{route_id}': ссылка на неизвестную стрелку "
                f"'{sw_id}' ({path})"
            )
        try:
            result[sw_id] = SwitchPosition(raw_pos)
        except ValueError:
            valid = [p.value for p in SwitchPosition]
            raise StationConfigError(
                f"Маршрут '{route_id}', стрелка '{sw_id}': "
                f"неизвестная позиция '{raw_pos}'. "
                f"Допустимые значения: {valid} ({path})"
            )

    return result


def _parse_station_meta(raw: dict[str, Any], path: Path) -> tuple[str, str]:
    """Извлекает station_id и name из старой или новой схемы YAML."""
    station_id = raw.get("station_id")
    name = raw.get("name")
    if station_id and name:
        return str(station_id), str(name)

    station = raw.get("station")
    if isinstance(station, dict):
        station_id = station.get("id")
        name = station.get("name")
        if station_id and name:
            return str(station_id), str(name)

    raise StationConfigError(
        f"Отсутствуют обязательные поля станции ('station_id'/'name' "
        f"или 'station.id'/'station.name') в файле {path}"
    )


def _normalize_route_type(raw_type: Any) -> Any:
    """Нормализует альтернативные обозначения route_type в канонические значения."""
    if not isinstance(raw_type, str):
        return raw_type
    mapping = {
        "reception": "arrival",
    }
    return mapping.get(raw_type, raw_type)


def _parse_extra_conflicts(
    raw: dict[str, Any],
    known_routes: set[str],
    path: Path,
) -> list[tuple[str, str]]:
    """Разбирает секцию 'extra_conflicts'.

    Каждый элемент — пара [route_a, route_b]. Запрещены:
    - self-conflict (route_a == route_b)
    - дублирующиеся пары (A,B) и (B,A)
    - ссылки на несуществующие маршруты

    Raises:
        StationConfigError: при любом нарушении выше.
    """
    raw_conflicts = raw.get("extra_conflicts")
    if not raw_conflicts:
        return []

    conflicts: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()

    for idx, item in enumerate(raw_conflicts):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise StationConfigError(
                f"extra_conflicts[{idx}]: ожидается пара [route_a, route_b], "
                f"получено {item!r} ({path})"
            )

        route_a, route_b = str(item[0]), str(item[1])

        if route_a == route_b:
            raise StationConfigError(
                f"extra_conflicts[{idx}]: маршрут '{route_a}' не может "
                f"конфликтовать сам с собой ({path})"
            )

        for r in (route_a, route_b):
            if r not in known_routes:
                raise StationConfigError(
                    f"extra_conflicts[{idx}]: ссылка на неизвестный маршрут "
                    f"'{r}' ({path})"
                )

        key = frozenset({route_a, route_b})
        if key in seen:
            raise StationConfigError(
                f"extra_conflicts[{idx}]: дублирующийся конфликт "
                f"({route_a!r}, {route_b!r}) — пара уже объявлена ({path})"
            )
        seen.add(key)

        conflicts.append((route_a, route_b))

    logger.debug("Разобрано %d дополнительных конфликтов", len(conflicts))
    return conflicts
