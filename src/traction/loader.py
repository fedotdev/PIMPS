# src/traction/loader.py
"""
Загрузка конфигурации локомотива и состава из YAML.
Единственная точка входа для тягового блока: читает файл,
собирает LocomotiveConfig и TrainConfig, не делает расчётов.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models import CurrentType, LocomotiveConfig, TrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Иерархия исключений
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Базовая ошибка конфигурации PIMPS."""


class LocomotiveConfigError(ConfigError):
    """Неправильный YAML или недопустимые значения в конфигурации локомотива."""


class TrainConfigError(ConfigError):
    """Неправильный YAML или недопустимые значения в конфигурации состава."""


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------

def load_locomotive(path: Path | str) -> LocomotiveConfig:
    """
    Читает YAML-файл локомотива и возвращает LocomotiveConfig.

    Ожидаемая структура файла — config/2ES5K.yaml.
    __post_init__ в LocomotiveConfig проверяет voltage и длины таблиц.
    """
    raw = _read_yaml(path, exc_cls=LocomotiveConfigError)
    try:
        v_table, fk_table = _parse_traction_curve(raw)
        wox_table          = _parse_resistance_curve(raw, v_table)
        bt_table           = _parse_brake_curve(raw, v_table)

        loco = LocomotiveConfig(
            loco_id       = str(raw["loco_id"]),
            name          = str(raw["name"]),
            current_type  = CurrentType(raw["current_type"]),
            voltage       = int(raw["voltage"]),
            mass_t        = float(raw["mass_t"]),
            loco_length_m = float(raw["loco_length_m"]),
            num_axes      = int(raw["num_axes"]),
            v_table       = v_table,
            fk_table      = fk_table,
            wox_table     = wox_table,
            bt_table      = bt_table,
            v_max         = float(raw["v_max"]),
        )
    except (LocomotiveConfigError, ConfigError):
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LocomotiveConfigError(
            f"Ошибка в файле '{path}': {exc}"
        ) from exc

    logger.info(
        "Локомотив загружен: id=%s, масса=%.1f т, v_max=%.1f км/ч, "
        "тяговых точек=%d",
        loco.loco_id, loco.mass_t, loco.v_max, len(loco.v_table),
    )

    if loco.v_max > float(loco.v_table[-1]):
        logger.warning(
            "Локомотив '%s': v_max=%.1f км/ч превышает правую границу "
            "тяговой характеристики v_table[-1]=%.1f км/ч. "
            "При v > %.1f км/ч Fk будет равна последнему табличному значению.",
            loco.loco_id,
            loco.v_max,
            float(loco.v_table[-1]),
            float(loco.v_table[-1]),
        )

    return loco


def load_train(
    loco: LocomotiveConfig,
    raw_or_path: dict[str, Any] | Path | str,
    consist_id: str | None = None,
) -> TrainConfig:
    """
    Собирает TrainConfig из локомотива и параметров состава.

    raw_or_path может быть:
    - словарём (уже считанным YAML-блоком 'consist'),
    - путём к отдельному YAML-файлу состава.
    """
    if isinstance(raw_or_path, dict):
        raw = raw_or_path
    else:
        raw = _read_yaml(raw_or_path, exc_cls=TrainConfigError)

    try:
        bt_wagons_table = _parse_wagon_brake_curve(raw, loco.v_table)
        train = TrainConfig(
            consist_id     = consist_id or str(raw["consist_id"]),
            loco           = loco,
            num_wagons     = int(raw["num_wagons"]),
            wagon_mass_t   = float(raw["wagon_mass_t"]),
            wagon_length_m = float(raw["wagon_length_m"]),
            q0             = float(raw["q0"]),
            wagon_type     = int(raw["wagon_type"]),
            bt_wagons_table = bt_wagons_table,
        )
    except (TrainConfigError, ConfigError):
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainConfigError(
            f"Ошибка в конфигурации состава: {exc}"
        ) from exc

    logger.info(
        "Состав загружен: id=%s, локомотив=%s, вагонов=%d, "
        "масса_поезда=%.1f т, длина=%.1f м",
        train.consist_id, loco.loco_id,
        train.num_wagons, train.train_mass_t, train.train_length_m,
    )
    return train


# ---------------------------------------------------------------------------
# Вспомогательные функции — парсинг кривых
# ---------------------------------------------------------------------------

def _parse_traction_curve(
    raw: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит v_table и fk_table как поточечный минимум тяговой и сцепной кривых.

    По ПТР РЖД 2016 расчётная сила тяги Fk(v) = min(Fk_тяга(v), Fk_сцепление(v)).
    Поле adhesion_curve обязательно: по ПТР §1.1.3 расчетная тяга берется как
    Fk(v) = min(Fk_traction(v), Fk_adhesion(v)).
    v_table ограничивается пересечением диапазонов обеих кривых во избежание
    экстраполяции за границы исходных данных.
    """
    if "traction_curve" not in raw:
        raise LocomotiveConfigError("Поле 'traction_curve' отсутствует в YAML.")
    if "adhesion_curve" not in raw:
        raise LocomotiveConfigError("Поле 'adhesion_curve' отсутствует в YAML.")

    v_tr, fk_tr = _curve_to_arrays(raw["traction_curve"], "traction_curve", "v", "Fk")
    v_ad, fk_ad = _curve_to_arrays(raw["adhesion_curve"], "adhesion_curve", "v", "Fk")

    if fk_ad[0] > fk_tr[0]:
        raise LocomotiveConfigError(
            "adhesion_curve[0].Fk должен быть <= traction_curve[0].Fk "
            f"для sanity-check сцепления при v={v_ad[0]:.1f} км/ч "
            f"({fk_ad[0]:.3f} > {fk_tr[0]:.3f})."
        )

    # Предупреждаем, если диапазоны кривых не совпадают — np.interp молча экстраполирует
    if v_tr[0] != v_ad[0] or v_tr[-1] != v_ad[-1]:
        logger.warning(
            "Диапазоны traction_curve [%.1f, %.1f] и adhesion_curve [%.1f, %.1f] "
            "не совпадают. v_table ограничен пересечением [%.1f, %.1f].",
            v_tr[0], v_tr[-1], v_ad[0], v_ad[-1],
            max(v_tr[0], v_ad[0]), min(v_tr[-1], v_ad[-1]),
        )

    # Объединяем скоростные точки, затем обрезаем до пересечения диапазонов
    v_union  = np.union1d(v_tr, v_ad)
    v_lo     = max(v_tr[0],  v_ad[0])
    v_hi     = min(v_tr[-1], v_ad[-1])
    v_table  = v_union[(v_union >= v_lo) & (v_union <= v_hi)]

    if len(v_table) == 0:
        raise LocomotiveConfigError(
            "Диапазоны traction_curve и adhesion_curve не пересекаются — "
            "невозможно построить результирующую тяговую кривую."
        )

    fk_tr_i  = np.interp(v_table, v_tr, fk_tr)
    fk_ad_i  = np.interp(v_table, v_ad, fk_ad)
    fk_table = np.minimum(fk_tr_i, fk_ad_i)

    return v_table, fk_table


def _parse_resistance_curve(
    raw: dict[str, Any],
    v_table: np.ndarray,
) -> np.ndarray:
    """
    Вычисляет wox_table на точках v_table по формулам ПТР РЖД 2016, §1.2, табл. 2,
    формулы 19–20 (для электровозов переменного тока, 2ЭС5К и аналогов):

        wox(v) = a + b·v + c·v²

    Коэффициенты a, b, c берутся из YAML-поля 'wo_coeffs'.
    Для 2ЭС5К: a=6.4, b=0.089, c=0.0022 (ПТР 2016, строка 2ЭС5К).
    Результирующий массив проверяется на отсутствие отрицательных значений:
    отрицательное wox физически невозможно и означает ошибку коэффициентов.
    """
    if "wo_coeffs" not in raw:
        raise LocomotiveConfigError("Поле 'wo_coeffs' отсутствует в YAML.")

    coeffs = raw["wo_coeffs"]
    try:
        a = float(coeffs["a"])
        b = float(coeffs["b"])
        c = float(coeffs["c"])
    except (KeyError, TypeError) as exc:
        raise LocomotiveConfigError(
            f"wo_coeffs должен содержать поля a, b, c: {exc}"
        ) from exc

    wox_table = a + b * v_table + c * v_table ** 2

    if np.any(wox_table < 0):
        logger.warning(
            "wox_table содержит отрицательные значения (min=%.4f) — "
            "проверьте коэффициенты wo_coeffs.",
            float(np.min(wox_table)),
        )

        if float(v_table[0]) > 0.0:
            logger.warning(
                "Тяговая характеристика начинается с v=%.1f км/ч > 0; "
                "сопротивление движения при меньших скоростях равно граничному "
                "значению wox=%.4f Н/кН.",
                float(v_table[0]),
                float(wox_table[0]),
            )

    return wox_table


def _parse_brake_curve(
    raw: dict[str, Any],
    v_table: np.ndarray,
) -> np.ndarray:
    """
    Возвращает bt_table — удельную тормозную силу на точках v_table.

    Если в YAML есть готовая таблица 'bt_curve' — интерполирует на v_table.
    Если есть 'bt_coeffs' с полями K_P, phi_0, phi_1 — вычисляет:
        φ_K(v) = phi_0 / (phi_1 + v)   — расчётный коэффициент трения (ПТР §2.2)
        bt(v)  = K_P · φ_K(v)
    Хотя бы одно из двух должно присутствовать.
    """
    if "bt_curve" in raw:
        v_bt, bt_vals = _curve_to_arrays(raw["bt_curve"], "bt_curve", "v", "bt")
        bt_table = np.interp(v_table, v_bt, bt_vals)

    elif "bt_coeffs" in raw:
        coeffs = raw["bt_coeffs"]
        try:
            k_p   = float(coeffs["K_P"])
            phi_0 = float(coeffs["phi_0"])
            phi_1 = float(coeffs["phi_1"])
        except (KeyError, TypeError) as exc:
            raise LocomotiveConfigError(
                f"bt_coeffs должен содержать K_P, phi_0, phi_1: {exc}"
            ) from exc

        # phi_1 — знаменательный сдвиг: должен быть строго положительным,
        # иначе при v=0 получим деление на ноль или отрицательный знаменатель
        if phi_1 <= 0:
            raise LocomotiveConfigError(
                f"bt_coeffs.phi_1 должен быть > 0, получено {phi_1}."
            )

        # Проверяем, что ни в одной точке v_table знаменатель не обнуляется
        # (теоретически невозможно при v>=0 и phi_1>0, но на случай кривых v<0)
        denominators = phi_1 + v_table
        if np.any(denominators <= 0):
            bad_v = v_table[denominators <= 0]
            raise LocomotiveConfigError(
                f"phi_1 + v <= 0 в точках v={bad_v} — деление на ноль "
                f"в формуле φ_K(v). Проверьте v_table и bt_coeffs.phi_1."
            )

        phi_k    = phi_0 / denominators
        bt_table = k_p * phi_k

    else:
        raise LocomotiveConfigError(
            "YAML должен содержать 'bt_curve' или 'bt_coeffs'."
        )

    if np.any(bt_table < 0):
        logger.warning(
            "bt_table содержит отрицательные значения (min=%.4f) — "
            "проверьте тормозные коэффициенты.",
            float(np.min(bt_table)),
        )

    return bt_table


def _parse_wagon_brake_curve(
    raw: dict[str, Any],
    v_table: np.ndarray,
) -> np.ndarray:
    """
    Возвращает удельную тормозную силу вагонов bT_wagon(v) на точках v_table.

    Поддерживает готовую таблицу 'bt_wagons_curve'/'bt_wagon_curve' или коэффициенты
    'bt_wagons_coeffs'/'bt_wagon_coeffs'/'bt_coeffs'. Для коэффициентов используется
    PTR §2.2: phi(k) = phi0 + phi1 * v / 2.2, bT_wagon = KP * phi(k).
    Если вагонные тормозные данные не заданы, возвращается нулевая таблица.
    """
    curve = raw.get("bt_wagons_curve", raw.get("bt_wagon_curve"))
    if curve is not None:
        try:
            v_bt, bt_vals = _curve_to_arrays(curve, "bt_wagons_curve", "v", "bt")
        except LocomotiveConfigError as exc:
            raise TrainConfigError(str(exc)) from exc
        bt_table = np.interp(v_table, v_bt, bt_vals)
    else:
        coeffs = raw.get(
            "bt_wagons_coeffs",
            raw.get("bt_wagon_coeffs", raw.get("bt_coeffs")),
        )
        if coeffs is None:
            return np.zeros_like(v_table, dtype=float)
        try:
            k_p = float(coeffs["KP"] if "KP" in coeffs else coeffs["K_P"])
            phi_0 = float(coeffs["phi0"] if "phi0" in coeffs else coeffs["phi_0"])
            phi_1 = float(coeffs["phi1"] if "phi1" in coeffs else coeffs["phi_1"])
        except (KeyError, TypeError, AttributeError) as exc:
            raise TrainConfigError(
                "Вагонные bt_coeffs должны содержать KP/K_P, phi0/phi_0, phi1/phi_1."
            ) from exc

        phi_k = phi_0 + phi_1 * v_table / 2.2
        bt_table = k_p * phi_k

    if np.any(bt_table < 0):
        raise TrainConfigError(
            "Вагонная bt_table содержит отрицательные значения "
            f"(min={float(np.min(bt_table)):.4f})."
        )
    return bt_table


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _read_yaml(
    path: Path | str,
    exc_cls: type[ConfigError] = ConfigError,
) -> dict[str, Any]:
    """Читает YAML-файл, возвращает словарь. Бросает exc_cls при ошибке."""
    path = Path(path)
    if not path.exists():
        raise exc_cls(f"Файл не найден: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise exc_cls(f"Ошибка парсинга YAML '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise exc_cls(f"Корень YAML должен быть словарём: {path}")
    return data


def _curve_to_arrays(
    curve: list[dict[str, float]],
    field_name: str,
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Преобразует список dict-ов [{x_key: ..., y_key: ...}]
    в два отсортированных numpy-массива (x, y).

    Параметры x_key и y_key обязательны — вызывающий код всегда
    явно указывает имена столбцов, чтобы избежать молчаливых ошибок.
    """
    if not isinstance(curve, list) or len(curve) == 0:
        raise LocomotiveConfigError(
            f"Поле '{field_name}' должно быть непустым списком точек."
        )
    try:
        pairs = [(float(p[x_key]), float(p[y_key])) for p in curve]
    except (KeyError, TypeError) as exc:
        raise LocomotiveConfigError(
            f"Точки '{field_name}' должны иметь поля '{x_key}' и '{y_key}': {exc}"
        ) from exc

    pairs.sort(key=lambda p: p[0])  # сортировка по оси X (скорость)
    x_arr = np.array([p[0] for p in pairs], dtype=float)
    y_arr = np.array([p[1] for p in pairs], dtype=float)

    if np.any(y_arr < 0):
        raise LocomotiveConfigError(
            f"'{field_name}': поле '{y_key}' содержит отрицательные значения — "
            f"физически недопустимо."
        )

    return x_arr, y_arr
