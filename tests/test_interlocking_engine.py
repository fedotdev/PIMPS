"""Тесты для src/interlocking/engine.py — логика управления маршрутами."""
from __future__ import annotations

import pytest

from src.interlocking.engine import (
    EngineError,
    InterlockingEngine,
    RouteConflictError,
    RouteNotFoundError,
    SwitchOccupiedError,
)
from src.models import (
    RouteConfig,
    RouteRequest,
    RouteStatus,
    RouteType,
    StationConfig,
    SwitchConfig,
    SwitchPosition,
    SwitchState,
)


@pytest.fixture
def base_config() -> StationConfig:
    """Базовый конфиг станции для тестирования.
    
    Содержит:
    - 2 стрелки (SW1, SW2)
    - 3 маршрута (R1, R2, R3)
    
    Логика конфликтов:
    - R1 и R2 конфликтуют по положению стрелки SW1 (PLUS vs MINUS)
    - R2 и R3 используют SW2 в одном положении (PLUS) и не имеют общих секций — совместимы
    - R1 и R3 не имеют общих объектов, но заданы как конфликтующие через extra_conflicts
    """
    switches = {
        "SW1": SwitchConfig(switch_id="SW1", normal_position=SwitchPosition.PLUS),
        "SW2": SwitchConfig(switch_id="SW2", normal_position=SwitchPosition.PLUS),
    }
    routes = {
        "R1": RouteConfig(
            route_id="R1",
            name="Маршрут 1",
            route_type=RouteType.ARRIVAL,
            sections=["SEC_A"],
            switches={"SW1": SwitchPosition.PLUS},
        ),
        "R2": RouteConfig(
            route_id="R2",
            name="Маршрут 2",
            route_type=RouteType.DEPARTURE,
            sections=["SEC_B"],
            switches={"SW1": SwitchPosition.MINUS, "SW2": SwitchPosition.PLUS},
        ),
        "R3": RouteConfig(
            route_id="R3",
            name="Маршрут 3",
            route_type=RouteType.PASSTHROUGH,
            sections=["SEC_C"],
            switches={"SW2": SwitchPosition.PLUS},
        ),
    }
    return StationConfig(
        station_id="ST_TEST",
        name="Тестовая Станция",
        routes=routes,
        switches=switches,
        extra_conflicts=[("R1", "R3")],
    )


@pytest.fixture
def engine(base_config: StationConfig) -> InterlockingEngine:
    return InterlockingEngine(base_config)


class TestEngineState:
    """Проверка базового состояния движка ЭЦ."""

    def test_initial_state(self, engine: InterlockingEngine):
        """При инициализации все маршруты закрыты, стрелки в нормальном положении."""
        state = engine.get_state()
        
        assert not state.active_routes
        
        assert state.route_statuses["R1"] is RouteStatus.CLOSED
        assert state.route_statuses["R2"] is RouteStatus.CLOSED
        assert state.route_statuses["R3"] is RouteStatus.CLOSED
        
        assert state.switch_states["SW1"] is SwitchState.NORMAL
        assert state.switch_states["SW2"] is SwitchState.NORMAL


class TestRouteLifecycle:
    """Жизненный цикл одного маршрута (открытие/закрытие)."""

    def test_request_route_success(self, engine: InterlockingEngine):
        """Успешное открытие маршрута запирает стрелки."""
        engine.request_route(RouteRequest("R1"))
        state = engine.get_state()
        
        assert "R1" in state.active_routes
        assert state.route_statuses["R1"] is RouteStatus.OPEN
        assert state.switch_states["SW1"] is SwitchState.LOCKED
        # SW2 не участвует в R1, должна остаться NORMAL
        assert state.switch_states["SW2"] is SwitchState.NORMAL

    def test_cancel_route_success(self, engine: InterlockingEngine):
        """Отмена маршрута возвращает стрелки в нормальное положение."""
        engine.request_route(RouteRequest("R1"))
        engine.cancel_route("R1")
        
        state = engine.get_state()
        assert not state.active_routes
        assert state.route_statuses["R1"] is RouteStatus.CLOSED
        assert state.switch_states["SW1"] is SwitchState.NORMAL

    def test_repeat_request_ignored(self, engine: InterlockingEngine):
        """Повторный запрос открытого маршрута игнорируется, не падая."""
        engine.request_route(RouteRequest("R1"))
        # Не должно вызывать исключений
        engine.request_route(RouteRequest("R1"))
        
        state = engine.get_state()
        assert state.route_statuses["R1"] is RouteStatus.OPEN

    def test_request_unknown_route(self, engine: InterlockingEngine):
        """Запрос несуществующего маршрута."""
        with pytest.raises(RouteNotFoundError, match="R_UNKNOWN"):
            engine.request_route(RouteRequest("R_UNKNOWN"))

    def test_cancel_unknown_route(self, engine: InterlockingEngine):
        """Отмена несуществующего маршрута."""
        with pytest.raises(RouteNotFoundError, match="R_UNKNOWN"):
            engine.cancel_route("R_UNKNOWN")

    def test_cancel_closed_route(self, engine: InterlockingEngine):
        """Попытка отменить маршрут, который уже закрыт."""
        with pytest.raises(EngineError, match="текущий статус CLOSED"):
            engine.cancel_route("R1")


class TestConflictsLogic:
    """Тестирование логики конфликтов (по топологии и extra_conflicts)."""

    def test_switch_conflict(self, engine: InterlockingEngine):
        """Конфликт из-за требуемого разного положения одной стрелки."""
        engine.request_route(RouteRequest("R1"))  # SW1 в PLUS
        
        with pytest.raises(RouteConflictError, match="R2"):
            engine.request_route(RouteRequest("R2"))  # SW1 требует MINUS

    def test_extra_conflict(self, engine: InterlockingEngine):
        """Конфликт, заданный явно через extra_conflicts."""
        engine.request_route(RouteRequest("R1"))
        
        with pytest.raises(RouteConflictError, match="R3"):
            engine.request_route(RouteRequest("R3"))

    def test_compatible_routes(self, engine: InterlockingEngine):
        """Совместимые маршруты могут быть открыты одновременно.
        
        R2 и R3 оба требуют SW2 в PLUS и не пересекаются по секциям.
        """
        engine.request_route(RouteRequest("R2"))
        engine.request_route(RouteRequest("R3"))
        
        state = engine.get_state()
        assert "R2" in state.active_routes
        assert "R3" in state.active_routes
        assert state.switch_states["SW2"] is SwitchState.LOCKED
        
    def test_cancel_with_shared_switches(self, engine: InterlockingEngine):
        """При отмене маршрута общие с другим маршрутом стрелки остаются LOCKED."""
        engine.request_route(RouteRequest("R2"))
        engine.request_route(RouteRequest("R3"))
        
        # Отменяем R2, но R3 всё ещё активен и использует SW2
        engine.cancel_route("R2")
        
        state = engine.get_state()
        assert state.route_statuses["R2"] is RouteStatus.CLOSED
        assert state.route_statuses["R3"] is RouteStatus.OPEN
        
        # SW1 входила только в R2, поэтому вернулась в NORMAL
        assert state.switch_states["SW1"] is SwitchState.NORMAL
        # SW2 используется R3, поэтому остаётся LOCKED
        assert state.switch_states["SW2"] is SwitchState.LOCKED


class TestSwitchFaults:
    """Тестирование неисправностей стрелок (FAULT)."""

    def test_fault_prevents_request(self, engine: InterlockingEngine):
        """Если стрелка в FAULT, открыть маршрут по ней нельзя."""
        engine.set_switch_fault("SW1")
        
        with pytest.raises(SwitchOccupiedError, match="неисправна \\(FAULT\\)"):
            engine.request_route(RouteRequest("R1"))

    def test_fault_leaves_other_routes_avail(self, engine: InterlockingEngine):
        """Неисправность одной стрелки не мешает маршрутам по другим стрелкам."""
        engine.set_switch_fault("SW1")
        
        # R3 использует только SW2, так что должен открыться
        engine.request_route(RouteRequest("R3"))
        state = engine.get_state()
        assert state.route_statuses["R3"] is RouteStatus.OPEN

    def test_clear_fault(self, engine: InterlockingEngine):
        """После снятия неисправности через clear_switch_fault маршрут можно открыть."""
        engine.set_switch_fault("SW1")
        engine.clear_switch_fault("SW1")
        
        engine.request_route(RouteRequest("R1"))
        state = engine.get_state()
        assert state.route_statuses["R1"] is RouteStatus.OPEN

    def test_fault_unknown_switch(self, engine: InterlockingEngine):
        """Попытка задать неисправность неизвестной стрелке."""
        with pytest.raises(KeyError, match="SW_UNKNOWN"):
            engine.set_switch_fault("SW_UNKNOWN")
