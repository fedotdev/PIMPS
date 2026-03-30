from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from src.models import (
    EngineState,
    RouteConfig,
    RouteRequest,
    RouteStatus,
    StationConfig,
    SwitchConfig,
    SwitchPosition,
    SwitchState,
)

__all__ = [
    "InterlockingEngine",
    "RouteStatus",
    "SwitchState",
    "EngineState",
    "RouteRequest",
    "EngineError",
    "RouteConflictError",
    "RouteNotFoundError",
    "SwitchOccupiedError",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------

class EngineError(RuntimeError):
    """Базовый класс ошибок движка ЭЦ."""

class RouteConflictError(EngineError):
    """Запрошенный маршрут конфликтует с уже открытым."""

class RouteNotFoundError(EngineError):
    """Маршрут с указанным ID не найден в конфигурации."""

class SwitchOccupiedError(EngineError):
    """Стрелка занята (входит в активный маршрут) и не может быть переведена."""

# ---------------------------------------------------------------------------
# Основной класс движка
# ---------------------------------------------------------------------------



class InterlockingEngine:
    """Движок электрической централизации (ЭЦ) для одной станции.

    Принимает :class:`~src.models.StationConfig`, загруженный через
    :func:`~src.interlocking.loader.load_station`, и управляет состоянием
    стрелок и маршрутов в соответствии с логикой взаимозависимостей.

    Не является потокобезопасным — внешняя синхронизация на вызывающей стороне.

    Example::

        config = load_station("stations/miitovskaya.yaml")
        engine = InterlockingEngine(config)
        engine.request_route(RouteRequest(route_id="M1"))
        state = engine.get_state()
    """

    def __init__(self, config: StationConfig) -> None:
        self._config = config

        # Все стрелки в нормальном положении при инициализации
        self._switch_states: dict[str, SwitchState] = {
            sw_id: SwitchState.NORMAL
            for sw_id in config.switches
        }

        # Все маршруты закрыты при инициализации
        self._route_statuses: dict[str, RouteStatus] = {
            route_id: RouteStatus.CLOSED
            for route_id in config.routes
        }

        # Матрица конфликтов: {route_id → {конфликтующие route_id}}
        self._conflict_matrix: dict[str, set[str]] = self._build_conflict_matrix()

        logger.debug(
            "InterlockingEngine создан для станции '%s'", config.station_id
        )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def request_route(self, request: RouteRequest) -> None:
        """Запросить открытие маршрута.

        Алгоритм:
        1. Проверить, что route_id существует в конфигурации.
        2. Проверить, что маршрут не конфликтует с уже открытыми.
        3. Перевести все стрелки маршрута в требуемые положения.
        4. Перевести статус маршрута в OPEN (или REQUESTED, если перевод асинхронный).

        Args:
            request: запрос с идентификатором и приоритетом маршрута.

        Raises:
            RouteNotFoundError: маршрут не найден в конфигурации.
            RouteConflictError: маршрут конфликтует с активным.
            SwitchOccupiedError: одна из стрелок заперта другим маршрутом.
        """
        route_id = request.route_id

        # 1. Проверка существования маршрута
        if route_id not in self._config.routes:
            raise RouteNotFoundError(
                f"Маршрут '{route_id}' не найден в конфигурации "
                f"станции '{self._config.station_id}'"
            )

        # Нельзя открыть уже открытый маршрут
        current_status = self._route_statuses[route_id]
        if current_status in (RouteStatus.OPEN, RouteStatus.REQUESTED):
            logger.warning(
                "Маршрут '%s' уже в состоянии %s, повторный запрос проигнорирован",
                route_id,
                current_status.name,
            )
            return

        # 2. Проверка конфликтов с активными маршрутами
        conflicts = self._check_conflicts(route_id)
        if conflicts:
            raise RouteConflictError(
                f"Маршрут '{route_id}' конфликтует с активными маршрутами: "
                f"{conflicts}"
            )

        route = self._config.routes[route_id]

        # 3. Перевести и запереть стрелки (проверит LOCKED/FAULT внутри)
        self._lock_switches(route)

        # 4. Маршрут открыт (синхронная модель — сразу OPEN)
        self._route_statuses[route_id] = RouteStatus.OPEN

        logger.info(
            "Маршрут '%s' (%s) открыт на станции '%s'",
            route_id,
            route.name,
            self._config.station_id,
        )

    def cancel_route(self, route_id: str) -> None:
        """Отменить (закрыть) открытый маршрут.

        После отмены все стрелки, входящие только в этот маршрут,
        возвращаются в нормальное положение. Стрелки, разделяемые
        с другим активным маршрутом, остаются заперты.

        Args:
            route_id: идентификатор маршрута для отмены.

        Raises:
            RouteNotFoundError: маршрут не найден в конфигурации.
            EngineError: маршрут не находится в состоянии OPEN или REQUESTED.
        """
        if route_id not in self._config.routes:
            raise RouteNotFoundError(
                f"Маршрут '{route_id}' не найден в конфигурации "
                f"станции '{self._config.station_id}'"
            )

        current_status = self._route_statuses[route_id]
        if current_status not in (RouteStatus.OPEN, RouteStatus.REQUESTED):
            raise EngineError(
                f"Невозможно отменить маршрут '{route_id}': текущий статус "
                f"{current_status.name}, ожидается OPEN или REQUESTED"
            )

        route = self._config.routes[route_id]

        # Отпереть стрелки (с учётом разделяемых с другими маршрутами)
        self._unlock_switches(route)

        self._route_statuses[route_id] = RouteStatus.CLOSED

        logger.info(
            "Маршрут '%s' (%s) отменён на станции '%s'",
            route_id,
            route.name,
            self._config.station_id,
        )

    def get_state(self) -> EngineState:
        """Вернуть снимок текущего состояния всех объектов станции.

        Returns:
            Объект :class:`EngineState` с актуальными статусами стрелок
            и маршрутов.
        """
        active = [
            route_id
            for route_id, status in self._route_statuses.items()
            if status in (RouteStatus.OPEN, RouteStatus.REQUESTED)
        ]

        return EngineState(
            switch_states=dict(self._switch_states),
            route_statuses=dict(self._route_statuses),
            active_routes=active,
        )

    def set_switch_fault(self, switch_id: str) -> None:
        """Отметить стрелку как неисправную (SwitchState.FAULT).

        Неисправная стрелка блокирует открытие всех маршрутов,
        в которые она входит.

        Args:
            switch_id: идентификатор стрелки.

        Raises:
            KeyError: стрелка с таким ID не найдена в конфигурации.
        """
        if switch_id not in self._switch_states:
            raise KeyError(
                f"Стрелка '{switch_id}' не найдена в конфигурации "
                f"станции '{self._config.station_id}'"
            )

        self._switch_states[switch_id] = SwitchState.FAULT

        logger.warning(
            "Стрелка '%s' помечена как неисправная (FAULT) на станции '%s'",
            switch_id,
            self._config.station_id,
        )

    def clear_switch_fault(self, switch_id: str) -> None:
        """Снять отметку неисправности стрелки.

        Возвращает стрелку в SwitchState.NORMAL (если она не заперта
        активным маршрутом).

        Args:
            switch_id: идентификатор стрелки.

        Raises:
            KeyError: стрелка с таким ID не найдена в конфигурации.
        """
        if switch_id not in self._switch_states:
            raise KeyError(
                f"Стрелка '{switch_id}' не найдена в конфигурации "
                f"станции '{self._config.station_id}'"
            )

        # Если стрелка заперта активным маршрутом — остаётся LOCKED
        if self._is_switch_in_active_route(switch_id):
            self._switch_states[switch_id] = SwitchState.LOCKED
            logger.info(
                "Неисправность стрелки '%s' снята, но стрелка остаётся LOCKED "
                "(входит в активный маршрут)",
                switch_id,
            )
        else:
            self._switch_states[switch_id] = SwitchState.NORMAL
            logger.info(
                "Неисправность стрелки '%s' снята, состояние → NORMAL",
                switch_id,
            )

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _build_conflict_matrix(self) -> dict[str, set[str]]:
        """Построить матрицу несовместимых маршрутов.

        Два маршрута конфликтуют, если:
        - они требуют перевода одной стрелки в разные положения, ИЛИ
        - они используют общие секции пути, ИЛИ
        - они объявлены в extra_conflicts конфигурации.

        Returns:
            Словарь ``{route_id: {конфликтующий_route_id, ...}}``.
        """
        routes = self._config.routes
        matrix: dict[str, set[str]] = {rid: set() for rid in routes}

        route_ids = list(routes.keys())
        for i, rid_a in enumerate(route_ids):
            route_a = routes[rid_a]
            for rid_b in route_ids[i + 1:]:
                route_b = routes[rid_b]

                # Конфликт по стрелкам: одна стрелка в разных положениях
                common_switches = set(route_a.switches) & set(route_b.switches)
                switch_conflict = any(
                    route_a.switches[sw_id] != route_b.switches[sw_id]
                    for sw_id in common_switches
                )

                # Конфликт по секциям: общие секции пути
                section_conflict = bool(
                    set(route_a.sections) & set(route_b.sections)
                )

                if switch_conflict or section_conflict:
                    # Матрица симметрична
                    matrix[rid_a].add(rid_b)
                    matrix[rid_b].add(rid_a)

        # Добавляем явные конфликты из extra_conflicts
        for route_a, route_b in self._config.extra_conflicts:
            matrix[route_a].add(route_b)
            matrix[route_b].add(route_a)

        logger.debug(
            "Матрица конфликтов построена: %d маршрутов, %d пар конфликтов",
            len(matrix),
            sum(len(v) for v in matrix.values()) // 2,
        )

        return matrix

    def _check_conflicts(self, route_id: str) -> list[str]:
        """Вернуть список ID активных маршрутов, конфликтующих с route_id.

        Args:
            route_id: маршрут, который планируется открыть.

        Returns:
            Список конфликтующих активных route_id. Пустой список —
            конфликтов нет.
        """
        conflicting = self._conflict_matrix.get(route_id, set())
        active_conflicts = [
            rid for rid in conflicting
            if self._route_statuses.get(rid) in (
                RouteStatus.OPEN, RouteStatus.REQUESTED,
            )
        ]
        return active_conflicts

    def _lock_switches(self, route: RouteConfig) -> None:
        """Перевести и запереть все стрелки маршрута.

        Args:
            route: конфиг открываемого маршрута.

        Raises:
            SwitchOccupiedError: хотя бы одна стрелка заперта другим маршрутом
                в другом положении или неисправна.
        """
        # Сначала проверяем все стрелки, только потом меняем состояние
        # (атомарность: либо все стрелки переводятся, либо ни одна)
        for sw_id, required_pos in route.switches.items():
            state = self._switch_states[sw_id]

            if state == SwitchState.FAULT:
                raise SwitchOccupiedError(
                    f"Стрелка '{sw_id}' неисправна (FAULT), "
                    f"маршрут '{route.route_id}' не может быть открыт"
                )

            if state == SwitchState.LOCKED:
                # Стрелка заперта другим маршрутом — проверяем,
                # совпадает ли требуемое положение
                if not self._switch_locked_in_position(sw_id, required_pos):
                    raise SwitchOccupiedError(
                        f"Стрелка '{sw_id}' заперта в другом положении, "
                        f"маршрут '{route.route_id}' требует '{required_pos.value}'"
                    )

        # Все проверки пройдены — переводим и запираем
        for sw_id, required_pos in route.switches.items():
            self._switch_states[sw_id] = SwitchState.LOCKED

            logger.debug(
                "Стрелка '%s' переведена в '%s' и заперта (маршрут '%s')",
                sw_id,
                required_pos.value,
                route.route_id,
            )

    def _unlock_switches(self, route: RouteConfig) -> None:
        """Отпереть стрелки маршрута и вернуть их в нормальное положение.

        Стрелки, разделяемые с другим активным маршрутом, остаются LOCKED.

        Args:
            route: конфиг закрываемого маршрута.
        """
        for sw_id in route.switches:
            # Проверяем, входит ли стрелка в другой активный маршрут
            # (кроме текущего закрываемого)
            used_by_other = self._is_switch_in_active_route(
                sw_id, exclude_route=route.route_id,
            )

            if used_by_other:
                logger.debug(
                    "Стрелка '%s' остаётся LOCKED — используется другим маршрутом",
                    sw_id,
                )
            else:
                self._switch_states[sw_id] = SwitchState.NORMAL
                logger.debug(
                    "Стрелка '%s' разблокирована → NORMAL",
                    sw_id,
                )

    def _is_switch_in_active_route(
        self,
        switch_id: str,
        exclude_route: str | None = None,
    ) -> bool:
        """Проверить, входит ли стрелка в какой-либо активный маршрут.

        Args:
            switch_id: идентификатор стрелки.
            exclude_route: маршрут, который исключается из проверки
                (например, при отмене — сам закрываемый маршрут).

        Returns:
            True, если стрелка входит хотя бы в один активный маршрут.
        """
        for route_id, status in self._route_statuses.items():
            if route_id == exclude_route:
                continue
            if status not in (RouteStatus.OPEN, RouteStatus.REQUESTED):
                continue
            route = self._config.routes[route_id]
            if switch_id in route.switches:
                return True
        return False

    def _switch_locked_in_position(
        self,
        switch_id: str,
        required_pos: SwitchPosition,
    ) -> bool:
        """Проверить, заперта ли стрелка в требуемом положении.

        Если стрелка заперта активным маршрутом и этот маршрут требует
        то же положение, что и новый — конфликта нет.

        Args:
            switch_id: идентификатор стрелки.
            required_pos: требуемое положение.

        Returns:
            True, если стрелка заперта именно в нужном положении.
        """
        for route_id, status in self._route_statuses.items():
            if status not in (RouteStatus.OPEN, RouteStatus.REQUESTED):
                continue
            route = self._config.routes[route_id]
            if switch_id in route.switches:
                if route.switches[switch_id] == required_pos:
                    return True
        return False
