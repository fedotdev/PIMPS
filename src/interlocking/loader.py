# src/interlocking/loader.py
"""
Загрузка конфигурации станции из YAML.
Единственная точка входа для блока ЭЦ: читает файл,
собирает StationConfig, не выполняет проверку враждебности маршрутов.
"""
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

    try:
        station_id = str(raw["station_id"])
        name       = str(raw["name"])
        switches   = _parse_switches(raw)
        routes     = _parse_routes(raw, switches)
        extra      = _parse_extra_conflicts(raw, routes)

        station = StationConfig(
            station_id      = station_id,
            name            = name,
            routes          = routes,
            switches        = switches,
            extra_conflicts = extra,
        )
    except (StationConfigError, ConfigError):
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StationConfigError(
            f"Отсутствует обязательное поле 'name' в файле {path}"
        )

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
        "Станция загружена: id=%s, маршрутов=%d, стрелок=%d, "
        "доп. конфликтов=%d",
        station.station_id,
        len(station.routes),
        len(station.switches),
        len(station.extra_conflicts),
    )
    return station


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
        switch_id = item.get("switch_id")
        if not switch_id:
            raise StationConfigError(
                f"Стрелка #{idx}: отсутствует поле 'switch_id' ({path})"
            )

        if switch_id in switches:
            raise StationConfigError(
                f"Дублирующийся switch_id '{switch_id}' в файле {path}"
            )

        raw_pos = item.get("normal_position")
        try:
            normal_position = SwitchPosition(raw_pos)
        except ValueError:
            valid = [p.value for p in SwitchPosition]
            raise StationConfigError(
                f"Стрелка '{switch_id}': неизвестная позиция '{raw_pos}'. "
                f"Допустимые значения: {valid} ({path})"
            )

        transfer_time: float = float(item.get("transfer_time_s", 4.0))
        if transfer_time <= 0:
            raise StationConfigError(
                f"switches[{i}] (id={switch_id!r}): "
                f"transfer_time_s должен быть > 0, получено {transfer_time}."
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
        route_id = item.get("route_id")
        if not route_id:
            raise StationConfigError(
                f"Маршрут #{idx}: отсутствует поле 'route_id' ({path})"
            )

        if route_id in routes:
            raise StationConfigError(
                f"Дублирующийся route_id '{route_id}' в файле {path}"
            )

        try:
            name = str(item["name"])
        except KeyError as exc:
            raise StationConfigError(
                f"routes[{i}] (id={route_id!r}): отсутствует поле 'name'."
            ) from exc

        try:
            route_type = RouteType(item["route_type"])
        except (KeyError, ValueError) as exc:
            valid = [rt.value for rt in RouteType]
            raise StationConfigError(
                f"Маршрут '{route_id}': список секций не может быть пустым ({path})"
            )

        dup_sections = [s for s in sections if sections.count(s) > 1]
        if dup_sections:
            raise StationConfigError(
                f"Маршрут '{route_id}': дублирующиеся секции: "
                f"{sorted(set(dup_sections))} ({path})"
            )
        sections: list[str] = [str(s) for s in sections_raw]

        v_limit: float = float(item.get("v_limit", 60.0))
        if v_limit <= 0:
            raise StationConfigError(
                f"routes[{i}] (id={route_id!r}): "
                f"v_limit должен быть > 0, получено {v_limit}."
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

    return routes


def _check_unique_sections(sections: list[str], route_id: str) -> None:
    """Проверяет отсутствие дублирующихся секций в маршруте."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for sec in sections:
        if sec in seen:
            duplicates.append(sec)
        seen.add(sec)
    if duplicates:
        raise StationConfigError(
            f"routes[id={route_id!r}]: дублирующиеся секции: {duplicates}."
        )


def _parse_route_switches(
    raw_switches: dict[str, Any],
    route_id: str,
    known_switches: dict[str, SwitchConfig],
    path: Path,
) -> dict[str, SwitchPosition]:
    """Разбирает блок стрелок внутри маршрута с перекрёстной валидацией.

    Raises:
        StationConfigError: ссылка на неизвестную стрелку или неизвестная позиция.
    """
    result: dict[str, SwitchPosition] = {}

    for sw_id, raw_pos in raw_switches.items():
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


def _parse_extra_conflicts(
    raw: dict[str, Any],
    known_routes: set[str],
    path: Path,
) -> list[tuple[str, str]]:
    """
    Разбирает необязательный список явно заданных дополнительных конфликтов.
    Каждый элемент — пара [route_a, route_b].
    Проверяет, что оба маршрута объявлены в known_routes.

    Если ключ 'extra_conflicts' отсутствует — возвращает пустой список.
    """
    raw_conflicts = raw.get("extra_conflicts")
    if not raw_conflicts:
        return []
    if not isinstance(items, list):
        raise StationConfigError(
            "Поле 'extra_conflicts' должно быть списком пар маршрутов."
        )

    result: list[tuple[str, str]] = []
    for i, pair in enumerate(items):
        if not isinstance(pair, list) or len(pair) != 2:
            raise StationConfigError(
                f"extra_conflicts[{idx}]: маршрут '{route_a}' не может "
                f"конфликтовать сам с собой ({path})"
            )

        for rid in (route_a, route_b):
            if rid not in known_routes:
                raise StationConfigError(
                    f"extra_conflicts[{idx}]: ссылка на неизвестный маршрут "
                    f"'{r}' ({path})"
                )

        result.append((route_a, route_b))

    return result


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict[str, Any]:
    """Читает YAML-файл, возвращает словарь. Бросает StationConfigError при ошибке."""
    if not path.exists():
        raise StationConfigError(f"Файл не найден: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise StationConfigError(
            f"Ошибка парсинга YAML '{path}': {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise StationConfigError(
            f"Корень YAML должен быть словарём: {path}"
        )
    return data
