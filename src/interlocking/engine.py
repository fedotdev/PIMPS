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
    ControlMode,
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

    def __init__(self, config: StationConfig, vc_methodology: str = "A", control_mode: str = "AB") -> None:
        self._config = config
        self._vc_methodology = vc_methodology
        self._control_mode = control_mode

        # Все стрелки в нормальном положении при инициализации
        self._switch_states: dict[str, SwitchState] = {
            sw_id: SwitchState.NORMAL
            for sw_id in config.switches
        }

        # Отслеживание освобождённых входных секций маршрутов (для VC)
        self._freed_sections: dict[str, set[str]] = {
            route_id: set() for route_id in config.routes
        }

        # Все маршруты закрыты при инициализации
        self._route_statuses: dict[str, RouteStatus] = {
            route_id: RouteStatus.CLOSED
            for route_id in config.routes
        }
        
        # Кто удерживает маршрут (train_id)
        self._route_acquirers: dict[str, set[str]] = {
            route_id: set() for route_id in config.routes
        }
        # Какой пакет зарезервировал маршрут (route_id -> platoon_id)
        self._route_reserved_by_platoon: dict[str, str] = {}

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
        train_id = request.train_id
        platoon_id = request.platoon_id
        is_method_b = (self._vc_methodology == "B" and platoon_id is not None)

        # 1. Проверка существования маршрута
        if route_id not in self._config.routes:
            raise RouteNotFoundError(
                f"Маршрут '{route_id}' не найден в конфигурации "
                f"станции '{self._config.station_id}'"
            )

        # Если маршрут уже открыт
        current_status = self._route_statuses[route_id]
        if current_status in (RouteStatus.OPEN, RouteStatus.REQUESTED):
            if train_id in self._route_acquirers[route_id]:
                # Запрос от того же поезда — игнорируем
                return
            if is_method_b and self._route_reserved_by_platoon.get(route_id) == platoon_id:
                # Пакетное присоединение
                self._route_acquirers[route_id].add(train_id)
                logger.debug("Поезд %s присоединился к пакетному маршруту %s", train_id, route_id)
                return
                
            # Секционное освобождение (только для режима VC)
            if self._control_mode == "VC":
                route = self._config.routes[route_id]
                if route.sections:
                    entry_section = route.sections[0]
                    if entry_section in self._freed_sections[route_id]:
                        self._route_acquirers[route_id].add(train_id)
                        # Занимаем входную секцию обратно
                        self._freed_sections[route_id].remove(entry_section)
                        logger.debug("Поезд %s попутно занял маршрут %s (секция %s свободна)", train_id, route_id, entry_section)
                        return
                    else:
                        logger.warning("Маршрут '%s' уже в состоянии OPEN (занято), запрос от %s отклонен", route_id, train_id)
                        raise RouteConflictError(f"Маршрут {route_id} уже занят и sectional release не готов.")

            logger.warning(
                "Маршрут '%s' уже в состоянии %s (занято), запрос от %s проигнорирован/отклонен",
                route_id, current_status.name, train_id
            )
            raise RouteConflictError(f"Маршрут {route_id} уже открыт другим поездом/пакетом")

        # Определяем, какие маршруты нужно перевести и запереть атомарно
        routes_to_open = [route_id]
        if is_method_b and request.batch_routes:
            # Для методики Б пытаемся занять все запрошенные маршруты пакета сразу
            routes_to_open = request.batch_routes

        # 2. Проверка конфликтов для всех открываемых маршрутов
        for rid in routes_to_open:
            if rid not in self._config.routes:
                raise RouteNotFoundError(f"Batch маршрут '{rid}' не найден")
                
            conflicts = self._check_conflicts(
                rid, 
                requester_platoon_id=platoon_id if is_method_b else None,
                requester_train_id=train_id
            )
            if conflicts:
                raise RouteConflictError(
                    f"Маршрут '{rid}' (часть запроса {route_id}) конфликтует с активными: {conflicts}"
                )
            
            # Также проверяем, не открыт ли батч-маршрут кем-то другим
            if self._route_statuses[rid] in (RouteStatus.OPEN, RouteStatus.REQUESTED):
                if not (is_method_b and self._route_reserved_by_platoon.get(rid) == platoon_id):
                    raise RouteConflictError(f"Маршрут '{rid}' уже занят другим поездом/пакетом")

        # 3. Перевести и запереть все требуемые стрелки
        for rid in routes_to_open:
            # Если батч-маршрут уже открыт нами же, пропускаем блокировку
            if self._route_statuses[rid] in (RouteStatus.OPEN, RouteStatus.REQUESTED):
                continue
            route = self._config.routes[rid]
            self._lock_switches(route)
            self._route_statuses[rid] = RouteStatus.OPEN
            if is_method_b and platoon_id:
                self._route_reserved_by_platoon[rid] = platoon_id
                
        # 4. Добавляем конкретный поезд в acquirers только для запрошенного маршрута
        self._route_acquirers[route_id].add(train_id)

        logger.info(
            "Маршруты %s открыты на станции '%s' (train=%s, platoon=%s)",
            routes_to_open, self._config.station_id, train_id, platoon_id
        )

    def cancel_route(self, route_id: str, train_id: str = "") -> None:
        """Отменить (закрыть) открытый маршрут.
        
        Если маршрут используют несколько поездов (пакет), он закрывается
        только когда его покинет последний поезд.

        Args:
            route_id: идентификатор маршрута для отмены.
            train_id: идентификатор поезда, освобождающего маршрут.


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

        if train_id in self._route_acquirers[route_id]:
            self._route_acquirers[route_id].remove(train_id)
            
        # Если маршрут всё ещё нужен другим поездам пакета - не закрываем!
        if len(self._route_acquirers[route_id]) > 0:
            logger.debug(
                "Маршрут '%s' освобожден поездом %s, но остается OPEN для других поездов пакета",
                route_id, train_id
            )
            return

        route = self._config.routes[route_id]

        # Отпереть стрелки (с учётом разделяемых с другими маршрутами)
        self._unlock_switches(route)

        self._route_statuses[route_id] = RouteStatus.CLOSED
        self._freed_sections[route_id].clear()
        if route_id in self._route_reserved_by_platoon:
            del self._route_reserved_by_platoon[route_id]

        logger.info(
            "Маршрут '%s' (%s) отменён (train=%s) на станции '%s'",
            route_id,
            route.name,
            train_id,
            self._config.station_id,
        )

    def release_section(self, route_id: str, section_id: str) -> None:
        """Пометить секцию маршрута как физически свободную (хвост поезда прошёл её)."""
        if route_id in self._freed_sections:
            self._freed_sections[route_id].add(section_id)
            logger.debug("На маршруте '%s' освобождена секция '%s'", route_id, section_id)

    def is_route_free(self, route_id: str, platoon_id: str | None = None) -> bool:
        """Проверить, можно ли открыть маршрут без конфликтов (не меняя состояние).
        
        Args:
            route_id: маршрут для проверки.
            platoon_id: пакет, запрашивающий маршрут.
            
        Returns:
            True, если маршрут свободен и доступен для открытия.
        """
        if route_id not in self._config.routes:
            return False
            
        current_status = self._route_statuses[route_id]
        is_method_b = (self._vc_methodology == "B" and platoon_id is not None)
        
        if current_status in (RouteStatus.OPEN, RouteStatus.REQUESTED):
            if is_method_b and self._route_reserved_by_platoon.get(route_id) == platoon_id:
                pass # Уже зарезервирован нашим же пакетом
            else:
                return False
                
        conflicts = self._check_conflicts(
            route_id, 
            requester_platoon_id=platoon_id if is_method_b else None,
            requester_train_id=None # Мы здесь проверяем только факт свободы
        )
        if conflicts:
            return False
            
        # Также проверяем стрелки на FAULT или запертость в другом положении
        route = self._config.routes[route_id]
        for sw_id, required_pos in route.switches.items():
            state = self._switch_states[sw_id]
            if state == SwitchState.FAULT:
                return False
            if state == SwitchState.LOCKED and not self._switch_locked_in_position(sw_id, required_pos):
                return False
                
        return True

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

    def _check_conflicts(
        self, 
        route_id: str, 
        requester_platoon_id: str | None = None,
        requester_train_id: str | None = None
    ) -> list[str]:
        """Вернуть список ID активных маршрутов, конфликтующих с route_id.

        Args:
            route_id: маршрут, который планируется открыть.
            requester_platoon_id: ID пакета, запрашивающего маршрут.
            requester_train_id: ID поезда, запрашивающего маршрут.

        Returns:
            Список конфликтующих активных route_id.
        """
        conflicting = self._conflict_matrix.get(route_id, set())
        active_conflicts = []
        for rid in conflicting:
            if self._route_statuses.get(rid) in (RouteStatus.OPEN, RouteStatus.REQUESTED):
                # Проверка: если режим VC и есть освобожденные секции, конфликт может быть исчерпан
                if self._control_mode == "VC" and self._freed_sections.get(rid):
                    route_a = self._config.routes[route_id]
                    route_b = self._config.routes[rid]
                    
                    # Оставшиеся (занятые) секции маршрута B
                    occupied_b = set(route_b.sections) - self._freed_sections[rid]
                    
                    # Если общих занятых секций нет и нет конфликта по стрелкам
                    common_sections = set(route_a.sections) & occupied_b
                    
                    # Проверка стрелок (стрелки не освобождаются секционно в этой версии)
                    common_switches = set(route_a.switches) & set(route_b.switches)
                    switch_conflict = any(
                        route_a.switches[sw_id] != route_b.switches[sw_id]
                        for sw_id in common_switches
                    )
                    
                    if not common_sections and not switch_conflict:
                        continue # Конфликт снят!

                # Если маршрут принадлежит тому же пакету — игнорируем!
                if requester_platoon_id and self._route_reserved_by_platoon.get(rid) == requester_platoon_id:
                    continue
                # Если маршрут удерживается этим же поездом — игнорируем!
                if requester_train_id and requester_train_id in self._route_acquirers.get(rid, set()):
                    continue
                active_conflicts.append(rid)
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
