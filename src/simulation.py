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
    EventType,
    ControlMode
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
        vc_methodology: str = "A",
        vc_min_headway_s: float = 60.0,
        route_setup_time_s: float = 0.0,
    ) -> None:
        """
        Args:
            interlocking: Движок ЭЦ для управления маршрутами
            traction_cache: Кэш тяговых расчётов
            scenario_name: Имя сценария (для сохранения в SimResult)
            control_mode: Режим СИРДП ("AB" или "VC")
            retry_interval_s: Интервал повторного запроса маршрута (в сек)
            vc_methodology: Методика ("A" или "B")
            vc_min_headway_s: Минимальный интервал отправления пакета
            route_setup_time_s: Физическое время приготовления маршрута (с)
        """
        self.env = simpy.Environment()
        self.interlocking = interlocking
        self.traction_cache = traction_cache
        self.scenario_name = scenario_name
        self.control_mode = control_mode
        self.vc_methodology = vc_methodology
        self.vc_min_headway_s = vc_min_headway_s
        self.route_setup_time_s = route_setup_time_s
        
        # Синхронизация режима с движком ЭЦ
        self.interlocking._control_mode = control_mode
        
        self.results: list[SimResult] = []
        self.events: list[StationEvent] = []
        self.platoon_routes: dict[str, list[str]] = {}
        self.last_platoon_depart_time: dict[str, float] = {}
        
        # Диспетчер маршрутов
        self._request_counter = 0
        self.pending_requests: list[dict] = []
        self._dispatcher_trigger = self.env.event()
        self.env.process(self._dispatcher_process())

    def _trigger_dispatcher(self):
        """Пробуждает диспетчер маршрутов при новых запросах или освобождении."""
        if not self._dispatcher_trigger.triggered:
            self._dispatcher_trigger.succeed()

    def _dispatcher_process(self):
        """Центральный диспетчер очереди маршрутов со строгим FIFO-приоритетом."""
        while True:
            yield self._dispatcher_trigger
            self._dispatcher_trigger = self.env.event()
            
            if self.pending_requests:
                # Строгий порядок очереди: сначала по t_arrive_s, затем по стабильному счетчику
                self.pending_requests.sort(key=lambda x: (x["priority"], x["counter"]))
                
                to_remove = []
                for entry in self.pending_requests:
                    try:
                        logger.debug("Диспетчер пытается открыть маршрут %s для поезда %s", entry["request"].route_id, entry["request"].train_id)
                        self.interlocking.request_route(entry["request"])
                        # Маршрут успешно зарезервирован!
                        entry["event"].succeed()
                        to_remove.append(entry)
                    except (RouteConflictError, SwitchOccupiedError):
                        # Конфликт, маршрут пока недоступен — поезд ждет в очереди
                        pass
                    except Exception as e:
                        logger.exception("КРИТИЧЕСКАЯ ОШИБКА ДИСПЕТЧЕРА для поезда %s: %s", entry["request"].train_id, e)
                        raise # Останавливаем все, чтобы увидеть ошибку
                
                for r in to_remove:
                    self.pending_requests.remove(r)

    def _release_entry_section(self, delay_s: float, route_id: str, section_id: str):
        """Фоновый процесс: освобождает входную секцию после прохождения хвоста."""
        try:
            yield self.env.timeout(delay_s)
            self.interlocking.release_section(route_id, section_id)
            self._trigger_dispatcher()
        except Exception as e:
            logger.exception("ОШИБКА в фоновом процессе освобождения секции %s: %s", section_id, e)
        self._trigger_dispatcher()

    def _request_route_sync(self, req: RouteRequest, priority: float):
        """Ставит запрос в очередь диспетчера и ждет успеха."""
        self._request_counter += 1
        success_event = self.env.event()
        self.pending_requests.append({
            "request": req,
            "priority": priority,
            "counter": self._request_counter,
            "event": success_event,
        })
        self._trigger_dispatcher()
        return success_event

    def load_scenario(self, entries: list[ScenarioEntry]) -> None:
        """Загружает расписание. Для каждого поезда создаёт SimPy-процесс."""
        for entry in entries:
            if entry.platoon_id:
                if entry.platoon_id not in self.platoon_routes:
                    self.platoon_routes[entry.platoon_id] = []
                if entry.route_id not in self.platoon_routes[entry.platoon_id]:
                    self.platoon_routes[entry.platoon_id].append(entry.route_id)
                if entry.departure_route_id and entry.departure_route_id not in self.platoon_routes[entry.platoon_id]:
                    self.platoon_routes[entry.platoon_id].append(entry.departure_route_id)

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
        try:
            # 1. Ожидание времени прибытия
            target_arrive_s = entry.t_arrive_s + entry.delay_s
            if target_arrive_s > self.env.now:
                yield self.env.timeout(target_arrive_s - self.env.now)

            t_real_arrive_s = self.env.now
            self.events.append(StationEvent(entry.train_id, EventType.ARRIVED, entry.route_id, "", t_real_arrive_s))

            total_wait_s = 0.0
            
            # === ФАЗА 1: ПРИБЫТИЕ ===
            batch_routes = None
            if self.vc_methodology == "B" and entry.platoon_id:
                batch_routes = self.platoon_routes[entry.platoon_id]

            t_wait_start = self.env.now
            self.events.append(StationEvent(entry.train_id, EventType.ROUTE_REQUESTED, entry.route_id, "", self.env.now))

            req = RouteRequest(
                route_id=entry.route_id,
                train_id=entry.train_id,
                platoon_id=entry.platoon_id,
                batch_routes=batch_routes
            )
            
            # Ждём в очереди диспетчера
            yield self._request_route_sync(req, priority=entry.t_arrive_s)
            
            self.events.append(StationEvent(entry.train_id, EventType.ROUTE_ACQUIRED, entry.route_id, "", self.env.now))
            
            # Подготовка маршрута (перевод стрелок, замыкание)
            if self.route_setup_time_s > 0:
                yield self.env.timeout(self.route_setup_time_s)

            # Весь период от запроса до конца подготовки — это ожидание
            total_wait_s += self.env.now - t_wait_start
            t_real_arrive_s = self.env.now

            # Секционное освобождение (только для режима VC)
            if self.control_mode == ControlMode.VC:
                arr_route_config = self.interlocking._config.routes[entry.route_id]
                if arr_route_config.sections:
                    self.env.process(
                        self._release_entry_section(
                            self.vc_min_headway_s,
                            entry.route_id, 
                            arr_route_config.sections[0]
                        )
                    )

            physics_arr = self.traction_cache.get_or_compute(
                train=entry.train,
                sections=entry.sections,
                consist_id=entry.train.consist_id,
                route_id=entry.route_id,
                v0_kmh=entry.v0_kmh,
                mode="traction",
            )
            yield self.env.timeout(physics_arr.t_total_s)

            # === ФАЗА 2: СТОЯНКА ===
            if entry.dwell_s > 0:
                yield self.env.timeout(entry.dwell_s)

            # === ФАЗА 3: ОТПРАВЛЕНИЕ ===
            t_depart_s = self.env.now
            physics_dep = None
            t_tail_delay_s = 0.0

            if entry.departure_route_id and entry.departure_sections:
                t_wait_start = self.env.now
                if self.vc_methodology == "B" and entry.platoon_id:
                    # Забронировано, ждем интервал
                    last_depart = self.last_platoon_depart_time.get(entry.platoon_id, 0.0)
                    now = self.env.now
                    if now < last_depart + self.vc_min_headway_s:
                        yield self.env.timeout(last_depart + self.vc_min_headway_s - now)
                    
                    self.events.append(StationEvent(entry.train_id, EventType.ROUTE_REQUESTED, entry.departure_route_id, "", self.env.now))
                    
                    req = RouteRequest(
                        route_id=entry.departure_route_id,
                        train_id=entry.train_id,
                        platoon_id=entry.platoon_id,
                    )
                    yield self._request_route_sync(req, priority=entry.t_arrive_s)
                    self.events.append(StationEvent(entry.train_id, EventType.ROUTE_ACQUIRED, entry.departure_route_id, "", self.env.now))
                    
                    if self.route_setup_time_s > 0:
                        yield self.env.timeout(self.route_setup_time_s)

                else:
                    # Методика А (или АБ): честный запрос
                    self.events.append(StationEvent(entry.train_id, EventType.ROUTE_REQUESTED, entry.departure_route_id, "", self.env.now))
                    req = RouteRequest(
                        route_id=entry.departure_route_id,
                        train_id=entry.train_id,
                        platoon_id=entry.platoon_id,
                    )
                    yield self._request_route_sync(req, priority=entry.t_arrive_s)
                    self.events.append(StationEvent(entry.train_id, EventType.ROUTE_ACQUIRED, entry.departure_route_id, "", self.env.now))
                    
                    if self.route_setup_time_s > 0:
                        yield self.env.timeout(self.route_setup_time_s)
                
                # Весь период от Phase 3 start до готовности отправления
                total_wait_s += self.env.now - t_wait_start
                            
                # Фиксация времени отправления (после всех подготовок)
                t_depart_s = self.env.now
                if entry.platoon_id:
                    self.last_platoon_depart_time[entry.platoon_id] = t_depart_s
                    
                # Запуск секционного освобождения (только в VC)
                if self.control_mode == ControlMode.VC:
                    dep_route_config = self.interlocking._config.routes[entry.departure_route_id]
                    if dep_route_config.sections:
                        # Хвост освобождает входную секцию через vc_min_headway_s (интервал попутного следования)
                        self.env.process(
                            self._release_entry_section(
                                self.vc_min_headway_s,
                                entry.departure_route_id, 
                                dep_route_config.sections[0]
                            )
                        )

                physics_dep = self.traction_cache.get_or_compute(
                    train=entry.train,
                    sections=entry.departure_sections,
                    consist_id=entry.train.consist_id,
                    route_id=entry.departure_route_id,
                    v0_kmh=0.0,
                    mode="traction",
                )
                yield self.env.timeout(physics_dep.t_total_s)

            # Вычисляем задержку хвоста по последнему физическому расчету
            final_physics = physics_dep if physics_dep else physics_arr
            if len(final_physics.head_to_tail_s) > 0:
                t_tail_delay_s = float(final_physics.head_to_tail_s[-1])
                if t_tail_delay_s < 0 or not t_tail_delay_s == t_tail_delay_s:
                    t_tail_delay_s = 0.0

            if t_tail_delay_s > 0:
                yield self.env.timeout(t_tail_delay_s)

            # === ФАЗА 4: ОСВОБОЖДЕНИЕ ===
            # Освобождаем все занятые маршруты
            self.interlocking.cancel_route(entry.route_id, train_id=entry.train_id)
            self.events.append(StationEvent(entry.train_id, EventType.ROUTE_RELEASED, entry.route_id, "", self.env.now))
            
            if entry.departure_route_id:
                self.interlocking.cancel_route(entry.departure_route_id, train_id=entry.train_id)
                self.events.append(StationEvent(entry.train_id, EventType.ROUTE_RELEASED, entry.departure_route_id, "", self.env.now))

            self._trigger_dispatcher() # Оповещаем диспетчер на освобождение маршрутов

            self.events.append(StationEvent(entry.train_id, EventType.DEPARTED, entry.route_id, "", self.env.now))

            # Сбор статистики
            t_total_s = self.env.now - entry.t_arrive_s

            all_v = []
            if len(physics_arr.v_profile) > 0:
                all_v.extend(physics_arr.v_profile)
            if physics_dep and len(physics_dep.v_profile) > 0:
                all_v.extend(physics_dep.v_profile)

            v_max_kmh = float(max(all_v)) if all_v else 0.0
            v_avg_kmh = float(sum(all_v) / len(all_v)) if all_v else 0.0
            
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
                t_wait_s=total_wait_s, # сумма ожиданий при приеме и отправлении
                t_dwell_s=entry.dwell_s,
                t_total_s=t_total_s,
                t_planned_arrive_s=entry.t_arrive_s,
                t_planned_depart_s=entry.planned_depart_s,
                delay_arrive_s=delay_arrive_s,
                delay_depart_s=delay_depart_s,
                v_avg_kmh=v_avg_kmh,
                v_max_kmh=v_max_kmh,
                vc_methodology=self.vc_methodology,
                platoon_id=entry.platoon_id,
                departure_route_id=entry.departure_route_id,
            )
            self.results.append(res)
        except Exception as e:
            logger.exception("КРИТИЧЕСКАЯ ОШИБКА в процессе поезда %s: %s", entry.train_id, e)
            raise

    def _release_entry_section(self, delay_s: float, route_id: str, section_id: str):
        """Фоновый процесс: освобождает входную секцию после прохождения хвоста."""
        yield self.env.timeout(delay_s)
        self.interlocking.release_section(route_id, section_id)
        self._trigger_dispatcher()
