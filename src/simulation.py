"""Модуль событийной симуляции движения поездов на станции."""
from __future__ import annotations

import logging

import simpy

from src.interlocking.engine import (
    InterlockingEngine,
    RouteConflictError,
    SwitchOccupiedError,
)
from src.models import (
    RouteRequest, 
    ScenarioEntry, 
    SimResult, 
    StationEvent, 
    EventType
)
from src.traction.dynamics import TractionCache

logger = logging.getLogger(__name__)

__all__ = ["SimulationEngine"]


class SimulationEngine:
    """Движок событийной симуляции (SimPy).

    Увязывает работу электрической централизации (маршруты) 
    и физики (тяговые расчёты движущихся поездов).
    """

    def __init__(
        self,
        interlocking: InterlockingEngine,
        traction_cache: TractionCache,
        scenario_name: str = "default",
        control_mode: str = "AB",
        retry_interval_s: float = 30.0,
    ) -> None:
        """
        Args:
            interlocking: Движок ЭЦ для управления маршрутами
            traction_cache: Кэш тяговых расчётов
            scenario_name: Имя сценария (для сохранения в SimResult)
            control_mode: Режим СИРДП ("AB" или "VC")
            retry_interval_s: Интервал повторного запроса маршрута (в сек)
        """
        self.env = simpy.Environment()
        self.interlocking = interlocking
        self.traction_cache = traction_cache
        self.scenario_name = scenario_name
        self.control_mode = control_mode
        self.retry_interval_s = retry_interval_s
        self.results: list[SimResult] = []
        self.events: list[StationEvent] = []

    def load_scenario(self, entries: list[ScenarioEntry]) -> None:
        """Загружает расписание. Для каждого поезда создаёт SimPy-процесс."""
        for entry in entries:
            self.env.process(self._train_process(entry))

    def run(self, until: float | None = None) -> list[SimResult]:
        """Запускает симуляцию. Возвращает список собранных результатов."""
        logger.info("Запуск симуляции (сценарий: %s)", self.scenario_name)
        self.env.run(until=until)
        logger.info("Симуляция завершена. Обработано поездов: %d", len(self.results))
        return self.results

    def _train_process(self, entry: ScenarioEntry):
        """Жизненный цикл одного поезда на станции (генератор SimPy)."""
        # 1. Поезд ожидает своего времени прибытия
        if entry.t_arrive_s > self.env.now:
            yield self.env.timeout(entry.t_arrive_s - self.env.now)

        t_real_arrive_s = self.env.now
        self.events.append(StationEvent(entry.train_id, EventType.ARRIVED, entry.route_id, "", t_real_arrive_s))

        logger.info(
            "[%7.1f] Поезд %s прибыл. Запрос маршрута '%s'",
            self.env.now,
            entry.train_id,
            entry.route_id,
        )

        t_wait_start = self.env.now
        self.events.append(StationEvent(entry.train_id, EventType.ROUTE_REQUESTED, entry.route_id, "", self.env.now))

        # 2. Поезд пытается захватить маршрут
        while True:
            try:
                self.interlocking.request_route(RouteRequest(route_id=entry.route_id))
                self.events.append(StationEvent(entry.train_id, EventType.ROUTE_ACQUIRED, entry.route_id, "", self.env.now))
                logger.info(
                    "[%7.1f] Поезд %s ЗАХВАТИЛ маршрут '%s'",
                    self.env.now,
                    entry.train_id,
                    entry.route_id,
                )
                break  # Успех
            except (RouteConflictError, SwitchOccupiedError) as e:
                # Маршрут занят (конфликт). Ждём и повторяем запрос.
                logger.debug(
                    "[%7.1f] Поезд %s ожидает маршрут '%s': %s",
                    self.env.now,
                    entry.train_id,
                    entry.route_id,
                    str(e),
                )
                yield self.env.timeout(self.retry_interval_s)

        t_wait_s = self.env.now - t_wait_start

        # 3. Физический расчёт (выполняется мгновенно или берется из кэша)
        physics = self.traction_cache.get_or_compute(
            train=entry.train,
            sections=entry.sections,
            consist_id=entry.train.consist_id,
            route_id=entry.route_id,
            v0_kmh=entry.v0_kmh,
            mode="traction",
        )

        # 4. Движение (ожидаем время хода)
        yield self.env.timeout(physics.t_total_s)

        # 5. Стоянка (dwell)
        if entry.dwell_s > 0:
            yield self.env.timeout(entry.dwell_s)

        t_depart_s = self.env.now

        self.events.append(StationEvent(entry.train_id, EventType.DEPARTED, entry.route_id, "", t_depart_s))

        # 6. Освобождение маршрута после того, как хвост покинет последнюю секцию.
        t_tail_delay_s = 0.0
        if len(physics.head_to_tail_s) > 0:
            t_tail_delay_s = float(physics.head_to_tail_s[-1])
            # Защита от NaN/Inf
            if t_tail_delay_s < 0 or not t_tail_delay_s == t_tail_delay_s:
                t_tail_delay_s = 0.0

        if t_tail_delay_s > 0:
            yield self.env.timeout(t_tail_delay_s)

        self.interlocking.cancel_route(entry.route_id)
        self.events.append(StationEvent(entry.train_id, EventType.ROUTE_RELEASED, entry.route_id, "", self.env.now))
        logger.info(
            "[%7.1f] Поезд %s ОСВОБОДИЛ маршрут '%s' (ушёл со станции)",
            self.env.now,
            entry.train_id,
            entry.route_id,
        )

        # Сбор статистики
        t_total_s = self.env.now - entry.t_arrive_s

        v_max_kmh = float(max(physics.v_profile)) if len(physics.v_profile) > 0 else 0.0
        v_avg_kmh = float(sum(physics.v_profile) / len(physics.v_profile)) if len(physics.v_profile) > 0 else 0.0
        
        delay_arrive_s = t_real_arrive_s - entry.t_arrive_s
        delay_depart_s = t_depart_s - entry.planned_depart_s

        res = SimResult(
            train_id=entry.train_id,
            route_id=entry.route_id,
            consist_id=entry.train.consist_id,
            scenario=self.scenario_name,
            control_mode=self.control_mode,
            t_arrive_s=t_real_arrive_s,
            t_depart_s=t_depart_s,
            t_wait_s=t_wait_s,
            t_dwell_s=entry.dwell_s,
            t_total_s=t_total_s,
            t_planned_arrive_s=entry.t_arrive_s,
            t_planned_depart_s=entry.planned_depart_s,
            delay_arrive_s=delay_arrive_s,
            delay_depart_s=delay_depart_s,
            v_avg_kmh=v_avg_kmh,
            v_max_kmh=v_max_kmh,
        )
        self.results.append(res)
