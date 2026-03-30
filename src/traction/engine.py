# src/traction/engine.py
"""
Публичный API тягового блока PIMPS.

Этот модуль — единственная точка входа, через которую simulation.py
и interlocking/ обращаются к физике движения. Он ничего не знает о SimPy
и не содержит расчётов — только оркестрирует loader.py и dynamics.py.

Публичный API:
    TractionEngine        — фасад над TractionCache для simulation.py
    RouteBuilder          — сборка списка RouteSection из данных станции
    load_route_sections() — удобная обёртка для interlocking/
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import PhysicsResult, RouteSection, TrainConfig
from src.traction.dynamics import DriveMode, TractionCache, solve_route
from src.traction.loader import load_locomotive, load_train

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции — ограничитель скорости
# ---------------------------------------------------------------------------

def _apply_speed_limit(fk_kn: float, v_ms: float, v_limit_ms: float) -> float:
    """
    Ограничитель тяги по скорости секции.

    Если текущая скорость превышает лимит секции, тяга обнуляется (выбег).
    Логика аналогична _fk_kn при v >= v_max, но на уровне секции.

    Вызывается из _ode в dynamics.py при наличии RouteSection.v_limit.
    Здесь — самостоятельная функция для тестирования и будущей интеграции.

    ПТР РЖД 2016 не регламентирует алгоритм, но выбег при превышении — это
    стандартная практика ведения поезда (§1.1 «режим служебного торможения»
    обеспечивается машинистом, здесь используется upрощённая эмуляция).
    """
    if v_limit_ms <= 0.0:
        # v_limit не задан или некорректен — не применяем ограничение
        return fk_kn
    return 0.0 if v_ms > v_limit_ms else fk_kn


# ---------------------------------------------------------------------------
# RouteBuilder — сборка RouteSection из YAML-данных станции
# ---------------------------------------------------------------------------

@dataclass
class RouteBuilder:
    """
    Собирает список RouteSection для передачи в solve_route / TractionEngine.

    Принимает данные в формате, который выдаёт interlocking/loader.py:
    список словарей с полями section_id, s_start, s_end, grade, radius, v_limit.

    Пример использования:
        builder = RouteBuilder()
        builder.add_section_raw(raw_dict)          # из YAML
        sections = builder.build()                 # → list[RouteSection]
    """

    _sections: list[RouteSection] = field(default_factory=list, init=False, repr=False)

    def add_section(self, section: RouteSection) -> RouteBuilder:
        """Добавляет готовый RouteSection. Возвращает self для цепочки вызовов."""
        self._sections.append(section)
        return self

    def add_section_raw(self, raw: dict[str, Any]) -> RouteBuilder:
        """
        Парсит словарь и создаёт RouteSection.

        Обязательные ключи: section_id, s_start, s_end, grade.
        Необязательные: radius (0.0), v_limit (120.0).
        """
        try:
            sec = RouteSection(
                section_id=str(raw["section_id"]),
                s_start=float(raw["s_start"]),
                s_end=float(raw["s_end"]),
                grade=float(raw["grade"]),
                radius=float(raw.get("radius", 0.0)),
                v_limit=float(raw.get("v_limit", 120.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Некорректные данные секции {raw.get('section_id', '?')}: {exc}"
            ) from exc
        self._sections.append(sec)
        return self

    def build(self) -> list[RouteSection]:
        """
        Возвращает список секций, отсортированный по s_start.

        Бросает ValueError если список пуст или есть разрыв между секциями
        (дублирует _validate_sections из dynamics.py, но до вызова solve_ivp —
        чтобы ошибка конфигурации была поймана раньше).
        """
        if not self._sections:
            raise ValueError("RouteBuilder: список секций пуст.")
        sections = sorted(self._sections, key=lambda s: s.s_start)
        for i in range(len(sections) - 1):
            gap = abs(sections[i].s_end - sections[i + 1].s_start)
            if gap > 1e-3:
                raise ValueError(
                    f"Разрыв {gap:.3f} м между секциями "
                    f"'{sections[i].section_id}' и '{sections[i + 1].section_id}'."
                )
        return sections

    def clear(self) -> RouteBuilder:
        """Сбрасывает накопленный список. Позволяет повторно использовать builder."""
        self._sections.clear()
        return self


def load_route_sections(raw_sections: list[dict[str, Any]]) -> list[RouteSection]:
    """
    Удобная функция: собирает list[RouteSection] из списка сырых словарей.

    Предназначена для interlocking/loader.py, который передаёт секции
    маршрута в виде списка dict из station YAML. Избавляет от ручного
    создания RouteBuilder в каждом месте.

    raw_sections: список словарей — см. RouteBuilder.add_section_raw().
    """
    builder = RouteBuilder()
    for raw in raw_sections:
        builder.add_section_raw(raw)
    return builder.build()


# ---------------------------------------------------------------------------
# TractionEngine — фасад для simulation.py
# ---------------------------------------------------------------------------

@dataclass
class TractionEngine:
    """
    Оркестрирует предрасчёт и выдачу физических профилей движения.

    Один экземпляр создаётся в main.py и передаётся во все SimPy-процессы.
    Внутри хранит TractionCache — повторные запросы с теми же параметрами
    возвращают кэшированный результат без перезапуска ODE.

    Пример использования в main.py:
        engine = TractionEngine()
        engine.precompute_route(train, sections, "cs01", "route_A")
        # ... в SimPy-процессе:
        physics = engine.lookup("cs01", "route_A")
        t_travel = physics.t_total_s
    """

    _cache: TractionCache = field(default_factory=TractionCache, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Предрасчёт                                                           #
    # ------------------------------------------------------------------ #

    def precompute_route(
        self,
        train: TrainConfig,
        sections: list[RouteSection],
        consist_id: str,
        route_id: str,
        v0_kmh: float = 0.0,
        mode: DriveMode = "traction",
    ) -> PhysicsResult:
        """
        Считает (или достаёт из кэша) PhysicsResult для пары consist×route.

        Параметры полностью совпадают с TractionCache.get_or_compute.
        Возвращает PhysicsResult для немедленного использования или проверки.
        """
        return self._cache.get_or_compute(
            train=train,
            sections=sections,
            consist_id=consist_id,
            route_id=route_id,
            v0_kmh=v0_kmh,
            mode=mode,
        )

    def precompute_all(
        self,
        trains: dict[str, TrainConfig],
        routes: dict[str, list[RouteSection]],
        mode: DriveMode = "traction",
        v0_kmh: float = 0.0,
    ) -> None:
        """
        Пакетный предрасчёт для всех пар (consist_id, route_id).

        Вызывается из main.py один раз до env.run(), чтобы SimPy-процессы
        работали только с O(1) lookup, а не запускали ODE в реальном времени.

        trains:  {consist_id: TrainConfig}
        routes:  {route_id:   list[RouteSection]}
        """
        total = len(trains) * len(routes)
        logger.info(
            "TractionEngine.precompute_all: %d пар consist×route (mode=%s)",
            total, mode,
        )
        for consist_id, train in trains.items():
            for route_id, sections in routes.items():
                self.precompute_route(
                    train=train,
                    sections=sections,
                    consist_id=consist_id,
                    route_id=route_id,
                    v0_kmh=v0_kmh,
                    mode=mode,
                )
        logger.info("TractionEngine.precompute_all: все %d профилей готовы.", total)

    # ------------------------------------------------------------------ #
    # Получение результатов                                                #
    # ------------------------------------------------------------------ #

    def lookup(self, consist_id: str, route_id: str, mode: DriveMode = "traction", v0_kmh: float = 0.0) -> PhysicsResult:
        """
        Возвращает кэшированный PhysicsResult для (consist_id, route_id).

        Бросает KeyError, если precompute_route не был вызван заранее.
        Такое поведение предпочтительнее молчаливого пересчёта — ошибка
        конфигурации должна быть заметной на этапе инициализации.
        """
        key = f"{consist_id}:{route_id}:{mode}:{v0_kmh}"
        # Обращаемся к внутреннему _store напрямую, чтобы не запускать пересчёт
        result = self._cache._store.get(key)  # noqa: SLF001
        if result is None:
            raise KeyError(
                f"PhysicsResult для '{consist_id}:{route_id}' не найден в кэше. "
                f"Вызовите precompute_route() или precompute_all() перед симуляцией."
            )
        return result

    def invalidate(self, consist_id: str, route_id: str) -> None:
        """Удаляет все кэшированные результаты для данной пары consist×route."""
        self._cache.invalidate(consist_id, route_id)

    def clear(self) -> None:
        """Очищает весь кэш. Использовать при смене конфигурации между прогонами."""
        self._cache.clear()

    def __len__(self) -> int:
        """Количество закэшированных профилей."""
        return len(self._cache)

    # ------------------------------------------------------------------ #
    # Фабричный метод — загрузка из YAML                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(
        cls,
        loco_path: Path | str,
        consist_raw: dict[str, Any],
        consist_id: str | None = None,
    ) -> tuple["TractionEngine", TrainConfig]:
        """
        Создаёт TractionEngine и TrainConfig из YAML-файла локомотива
        и словаря параметров состава.

        Предназначен для быстрого старта в скриптах и тестах:
            engine, train = TractionEngine.from_yaml(
                "config/2ES5K.yaml",
                {"consist_id": "cs01", "num_wagons": 50, ...},
            )

        Возвращает (engine, train) — engine пустой (кэш не заполнен),
        train готов для передачи в precompute_route.
        """
        loco  = load_locomotive(loco_path)
        train = load_train(loco, consist_raw, consist_id=consist_id)
        return cls(), train
