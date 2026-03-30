# src/interlocking/loader.py
"""
Загрузка конфигурации станции из YAML.
Единственная точка входа для блока ЭЦ: читает файл,
собирает StationConfig, не выполняет проверку враждебности маршрутов.
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Иерархия исключений
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Базовая ошибка конфигурации PIMPS."""


class StationConfigError(ConfigError):
    """Неправильный YAML или недопустимые значения в конфигурации станции."""


# ---------------------------------------------------------------------------
# Публичная функция
# ---------------------------------------------------------------------------

def load_station(path: Path | str) -> StationConfig:
    """
    Читает YAML-файл станции и возвращает StationConfig.

    Ожидаемая структура файла — stations/<id>.yaml.
    Порядок разбора: стрелки → маршруты (с перекрёстной проверкой) →
    дополнительные конфликты.
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
            f"Ошибка в файле '{path}': {exc}"
        ) from exc

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
# Внутренние парсеры
# ---------------------------------------------------------------------------

def _parse_switches(raw: dict[str, Any]) -> dict[str, SwitchConfig]:
    """
    Разбирает список стрелочных переводов.

    Ожидаемая структура элемента:
        - switch_id: "1"
          normal_position: plus
          transfer_time_s: 4.0   # необязательно, по умолчанию 4.0
    """
    items: list[Any] = raw.get("switches", [])
    if not isinstance(items, list):
        raise StationConfigError("Поле 'switches' должно быть списком.")

    switches: dict[str, SwitchConfig] = {}
    for i, item in enumerate(items):
        try:
            switch_id = str(item["switch_id"])
        except (KeyError, TypeError) as exc:
            raise StationConfigError(
                f"switches[{i}]: отсутствует или неверен 'switch_id': {exc}"
            ) from exc

        try:
            position = SwitchPosition(item["normal_position"])
        except (KeyError, ValueError) as exc:
            valid = [p.value for p in SwitchPosition]
            raise StationConfigError(
                f"switches[{i}] (id={switch_id!r}): "
                f"недопустимое значение normal_position={item.get('normal_position')!r}. "
                f"Допустимые: {valid}."
            ) from exc

        transfer_time: float = float(item.get("transfer_time_s", 4.0))
        if transfer_time <= 0:
            raise StationConfigError(
                f"switches[{i}] (id={switch_id!r}): "
                f"transfer_time_s должен быть > 0, получено {transfer_time}."
            )

        if switch_id in switches:
            raise StationConfigError(
                f"Дублирующийся switch_id={switch_id!r} в блоке 'switches'."
            )

        switches[switch_id] = SwitchConfig(
            switch_id       = switch_id,
            normal_position = position,
            transfer_time_s = transfer_time,
        )

    return switches


def _parse_routes(
    raw: dict[str, Any],
    known_switches: dict[str, SwitchConfig],
) -> dict[str, RouteConfig]:
    """
    Разбирает список маршрутов; проверяет, что все упомянутые стрелки
    объявлены в блоке switches.

    Ожидаемая структура элемента:
        - route_id: "N1"
          name: "Приём на 1 путь"
          route_type: arrival
          sections: [sec_A, sec_B, sec_C]
          v_limit: 60.0             # необязательно, по умолчанию 60.0
          switches:
            "1": plus
            "3": minus
    """
    items: list[Any] = raw.get("routes", [])
    if not isinstance(items, list):
        raise StationConfigError("Поле 'routes' должно быть списком.")

    routes: dict[str, RouteConfig] = {}
    for i, item in enumerate(items):
        try:
            route_id = str(item["route_id"])
        except (KeyError, TypeError) as exc:
            raise StationConfigError(
                f"routes[{i}]: отсутствует или неверен 'route_id': {exc}"
            ) from exc

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
                f"routes[{i}] (id={route_id!r}): "
                f"недопустимое значение route_type={item.get('route_type')!r}. "
                f"Допустимые: {valid}."
            ) from exc

        sections_raw = item.get("sections", [])
        if not isinstance(sections_raw, list) or len(sections_raw) == 0:
            raise StationConfigError(
                f"routes[{i}] (id={route_id!r}): "
                f"'sections' должен быть непустым списком."
            )
        sections: list[str] = [str(s) for s in sections_raw]

        v_limit: float = float(item.get("v_limit", 60.0))
        if v_limit <= 0:
            raise StationConfigError(
                f"routes[{i}] (id={route_id!r}): "
                f"v_limit должен быть > 0, получено {v_limit}."
            )

        route_switches = _parse_route_switches(
            item.get("switches", {}),
            route_id=route_id,
            known_switches=known_switches,
        )

        if route_id in routes:
            raise StationConfigError(
                f"Дублирующийся route_id={route_id!r} в блоке 'routes'."
            )

        routes[route_id] = RouteConfig(
            route_id   = route_id,
            name       = name,
            route_type = route_type,
            sections   = sections,
            switches   = route_switches,
            v_limit    = v_limit,
        )

    return routes


def _parse_route_switches(
    raw_switches: Any,
    route_id: str,
    known_switches: dict[str, SwitchConfig],
) -> dict[str, SwitchPosition]:
    """
    Разбирает блок switches внутри маршрута.
    Проверяет, что каждый switch_id объявлен в known_switches.

    raw_switches — словарь вида {"1": "plus", "3": "minus"}.
    """
    if not isinstance(raw_switches, dict):
        raise StationConfigError(
            f"routes[id={route_id!r}]: поле 'switches' должно быть словарём."
        )

    result: dict[str, SwitchPosition] = {}
    for sw_id, pos_raw in raw_switches.items():
        sw_id = str(sw_id)

        if sw_id not in known_switches:
            raise StationConfigError(
                f"routes[id={route_id!r}]: стрелка switch_id={sw_id!r} "
                f"не объявлена в блоке 'switches' станции."
            )

        try:
            position = SwitchPosition(pos_raw)
        except ValueError as exc:
            valid = [p.value for p in SwitchPosition]
            raise StationConfigError(
                f"routes[id={route_id!r}], switch_id={sw_id!r}: "
                f"недопустимая позиция {pos_raw!r}. Допустимые: {valid}."
            ) from exc

        result[sw_id] = position

    return result


def _parse_extra_conflicts(
    raw: dict[str, Any],
    known_routes: dict[str, RouteConfig],
) -> list[tuple[str, str]]:
    """
    Разбирает необязательный список явно заданных дополнительных конфликтов.
    Каждый элемент — пара [route_a, route_b].
    Проверяет, что оба маршрута объявлены в known_routes.

    Если ключ 'extra_conflicts' отсутствует — возвращает пустой список.
    """
    items = raw.get("extra_conflicts", [])
    if items is None:
        return []
    if not isinstance(items, list):
        raise StationConfigError(
            "Поле 'extra_conflicts' должно быть списком пар маршрутов."
        )

    result: list[tuple[str, str]] = []
    for i, pair in enumerate(items):
        if not isinstance(pair, list) or len(pair) != 2:
            raise StationConfigError(
                f"extra_conflicts[{i}]: ожидается список из двух route_id, "
                f"получено: {pair!r}."
            )
        route_a, route_b = str(pair[0]), str(pair[1])

        for rid in (route_a, route_b):
            if rid not in known_routes:
                raise StationConfigError(
                    f"extra_conflicts[{i}]: route_id={rid!r} "
                    f"не объявлен в блоке 'routes' станции."
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
