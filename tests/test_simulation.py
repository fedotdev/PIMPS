"""Тесты для движка симуляции SimPy (src/simulation.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.interlocking.engine import InterlockingEngine
from src.models import (
    PhysicsResult,
    RouteConfig,
    RouteSection,
    RouteType,
    ScenarioEntry,
    StationConfig,
    SwitchConfig,
    SwitchPosition,
)
from src.simulation import SimulationEngine


# ---------------------------------------------------------------------------
# Моки
# ---------------------------------------------------------------------------

class MockTractionCache:
    """Мок для TractionCache, чтобы избежать долгих расчетов."""
    
    def __init__(self, t_total_mock: float = 100.0, tail_delay_mock: float = 10.0):
        self.t_total_mock = t_total_mock
        self.tail_delay_mock = tail_delay_mock
        self.calls = 0

    def get_or_compute(
        self, train, sections, consist_id, route_id, v0_kmh, mode
    ) -> PhysicsResult:
        self.calls += 1
        return PhysicsResult(
            consist_id=consist_id,
            route_id=route_id,
            t_total_s=self.t_total_mock,
            v_profile=np.array([]),
            t_profile=np.array([]),
            s_points=np.array([]),
            head_to_tail_s=np.array([self.tail_delay_mock]),
        )


@pytest.fixture
def station_config() -> StationConfig:
    sw1 = SwitchConfig(
        switch_id="SW1", normal_position=SwitchPosition.PLUS
    )
    r1 = RouteConfig(
        route_id="R1",
        name="Маршрут 1",
        route_type=RouteType.ARRIVAL,
        sections=[],
        switches={"SW1": SwitchPosition.PLUS},
    )
    r2 = RouteConfig(
        route_id="R2",
        name="Маршрут 2",
        route_type=RouteType.ARRIVAL,
        sections=[],
        switches={"SW1": SwitchPosition.MINUS},
    )
    return StationConfig(
        station_id="TEST",
        name="Test Station",
        switches={"SW1": sw1},
        routes={"R1": r1, "R2": r2},
        extra_conflicts=frozenset(),
    )


@pytest.fixture
def interlocking(station_config) -> InterlockingEngine:
    return InterlockingEngine(station_config)


@pytest.fixture
def dummy_train():
    """Фейковый состав, методы которого не вызываются (используем MockTractionCache)."""
    class DummyTrain:
        consist_id = "C1"
    return DummyTrain()


@pytest.fixture
def sim_engine(interlocking) -> SimulationEngine:
    return SimulationEngine(
        interlocking=interlocking,
        traction_cache=MockTractionCache(), # type: ignore
        scenario_name="TestScenario",
        retry_interval_s=10.0,
    )


# ---------------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------------

def test_single_train_simulation(sim_engine, dummy_train):
    """Один поезд успешно запрашивает маршрут, едет и освобождает его."""
    entry = ScenarioEntry(
        train_id="T1",
        t_arrive_s=50.0,
        route_id="R1",
        train=dummy_train, # type: ignore
        sections=[RouteSection("S1", 0.0, 100.0, 0)],
        v0_kmh=0.0,
        dwell_s=5.0
    )
    
    sim_engine.load_scenario([entry])
    results = sim_engine.run()
    
    assert len(results) == 1
    res = results[0]
    
    assert res.train_id == "T1"
    assert res.route_id == "R1"
    assert res.t_arrive_s == 50.0
    assert res.t_wait_s == 0.0 # Сразу получил маршрут
    assert res.t_dwell_s == 5.0
    
    # Полное время = physics.t_total_s (100.0) + dwell (5.0) + tail_delay (10.0)
    assert res.t_total_s == 100.0 + 5.0 + 10.0
    
    # Отправление со станции = arrive + wait (0) + physics (100) + dwell (5) = 155
    assert res.t_depart_s == 155.0


def test_two_trains_conflict(sim_engine, dummy_train):
    """Второй поезд приезжает, пока первый на маршруте, и ждет освобождения."""
    # Первый поезд пребывает в 0.0, занимает R1
    entry1 = ScenarioEntry(
        train_id="T1",
        t_arrive_s=0.0,
        route_id="R1",
        train=dummy_train, # type: ignore
        sections=[RouteSection("S1", 0.0, 100.0, 0)]
    )
    
    # Второй поезд пребывает в 50.0, хочет на R2 (конфликт с R1)
    entry2 = ScenarioEntry(
        train_id="T2",
        t_arrive_s=50.0,
        route_id="R2",
        train=dummy_train, # type: ignore
        sections=[RouteSection("S2", 0.0, 100.0, 0)]
    )
    
    sim_engine.load_scenario([entry1, entry2])
    results = sim_engine.run()
    
    assert len(results) == 2
    res1 = next(r for r in results if r.train_id == "T1")
    res2 = next(r for r in results if r.train_id == "T2")
    
    # Первый поезд едет без ожиданий: 0.0 -> physics (100) + tail (10)
    assert res1.t_wait_s == 0.0
    assert res1.t_total_s == 110.0
    
    # Второй поезд приехал в 50.0. 
    # В 50.0 он попытался запросить R2, но конфликт, так как R1 освободится только в 110.0.
    # retry_interval_s = 10. Попытки 2: 50, 60, 70, 80, 90, 100, 110 (успех!).
    # Он ждет 60 секунд.
    assert res2.t_wait_s == 60.0 # 110.0 - 50.0
    
    # Общее время второго = t_wait + physics (100) + tail (10)
    assert res2.t_total_s == 60.0 + 100.0 + 10.0
    
    # TractionCache был вызван для обоих
    assert sim_engine.traction_cache.calls == 2
