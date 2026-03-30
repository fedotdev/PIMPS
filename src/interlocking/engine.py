from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from src.models import RouteConfig, StationConfig, SwitchConfig, SwitchPosition

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
# Вспомогательные типы состояния
# ---------------------------------------------------------------------------


class RouteStatus(Enum):
    """Статус маршрута в движке."""

    CLOSED = auto()       # маршрут не открыт
    REQUESTED = auto()    # запрос принят, стрелки ещё переводятся
    OPEN = auto()         # маршрут открыт, движение разрешено
    CANCELLING = auto()   # отмена в процессе


class SwitchState(Enum):
    """Текущее фактическое положение стрелки."""

    NORMAL = auto()       # нормальное положение
    REVERSE = auto()      # переведена
    MOVING = auto()       # в процессе перевода
    LOCKED = auto()       # заперта (входит в активный маршрут)
    FAULT = auto()        # неисправность


@dataclass
class EngineState:
    """Снимок текущего состояния всех объектов станции.

    Возвращается методом :meth:`InterlockingEngine.get_state`.
    Объект иммутабелен — изменения состояния отражаются только
    в следующем вызове ``get_state()``.
    """

    # TODO: уточнить состав полей по мере реализации engine
    switch_states: dict[str, SwitchState] = field(default_factory=dict)
    route_statuses: dict[str, RouteStatus] = field(default_factory=dict)
    active_routes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Запрос маршрута
# ---------------------------------------------------------------------------


@dataclass
class RouteRequest:
    """Запрос на открытие маршрута.

    Args:
        route_id: идентификатор маршрута из конфигурации станции.
        priority: приоритет запроса (выше — важнее при коллизиях).
            По умолчанию 0.
    """

    route_id: str
    priority: int = 0

    # TODO: добавить поле requested_by (идентификатор диспетчера/системы)
    # TODO: добавить timestamp


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
        # TODO: инициализировать _switch_states из config.switches
        #       (все стрелки → SwitchState.NORMAL по умолчанию)
        self._switch_states: dict[str, SwitchState] = {}

        # TODO: инициализировать _route_statuses из config.routes
        #       (все маршруты → RouteStatus.CLOSED)
        self._route_statuses: dict[str, RouteStatus] = {}

        # TODO: построить матрицу конфликтов через _build_conflict_matrix()
        self._conflict_matrix: dict[str, set[str]] = {}

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
        # TODO: реализовать логику открытия маршрута
        pass

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
        # TODO: реализовать логику закрытия маршрута
        pass

    def get_state(self) -> EngineState:
        """Вернуть снимок текущего состояния всех объектов станции.

        Returns:
            Объект :class:`EngineState` с актуальными статусами стрелок
            и маршрутов.
        """
        # TODO: собрать и вернуть EngineState из внутренних словарей
        pass

    def set_switch_fault(self, switch_id: str) -> None:
        """Отметить стрелку как неисправную (SwitchState.FAULT).

        Неисправная стрелка блокирует открытие всех маршрутов,
        в которые она входит.

        Args:
            switch_id: идентификатор стрелки.

        Raises:
            KeyError: стрелка с таким ID не найдена в конфигурации.
        """
        # TODO: реализовать установку неисправности стрелки
        pass

    def clear_switch_fault(self, switch_id: str) -> None:
        """Снять отметку неисправности стрелки.

        Возвращает стрелку в SwitchState.NORMAL (если она не заперта
        активным маршрутом).

        Args:
            switch_id: идентификатор стрелки.

        Raises:
            KeyError: стрелка с таким ID не найдена в конфигурации.
        """
        # TODO: реализовать снятие неисправности стрелки
        pass

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _build_conflict_matrix(self) -> dict[str, set[str]]:
        """Построить матрицу несовместимых маршрутов.

        Два маршрута конфликтуют, если:
        - они требуют перевода одной стрелки в разные положения, ИЛИ
        - они объявлены в extra_conflicts конфигурации.

        Returns:
            Словарь ``{route_id: {конфликтующий_route_id, ...}}``.
        """
        # TODO: перебрать все пары маршрутов из config.routes
        # TODO: для каждой пары проверить пересечение по стрелкам
        #       (разные SwitchPosition → конфликт)
        # TODO: добавить пары из config.extra_conflicts
        # TODO: матрица симметрична: если A конфликтует с B, то B с A
        pass

    def _check_conflicts(self, route_id: str) -> list[str]:
        """Вернуть список ID активных маршрутов, конфликтующих с route_id.

        Args:
            route_id: маршрут, который планируется открыть.

        Returns:
            Список конфликтующих активных route_id. Пустой список —
            конфликтов нет.
        """
        # TODO: пересечь _conflict_matrix[route_id] с активными маршрутами
        pass

    def _lock_switches(self, route: RouteConfig) -> None:
        """Перевести и запереть все стрелки маршрута.

        Args:
            route: конфиг открываемого маршрута.

        Raises:
            SwitchOccupiedError: хотя бы одна стрелка заперта другим маршрутом.
        """
        # TODO: для каждой стрелки в route.switches:
        #   - проверить, что она не LOCKED и не FAULT
        #   - перевести в нужное положение (с учётом transfer_time_s)
        #   - установить SwitchState.LOCKED
        pass

    def _unlock_switches(self, route: RouteConfig) -> None:
        """Отпереть стрелки маршрута и вернуть их в нормальное положение.

        Стрелки, разделяемые с другим активным маршрутом, остаются LOCKED.

        Args:
            route: конфиг закрываемого маршрута.
        """
        # TODO: для каждой стрелки в route.switches:
        #   - проверить, не входит ли она в другой активный маршрут
        #   - если нет — перевести в normal_position, установить NORMAL
        pass
