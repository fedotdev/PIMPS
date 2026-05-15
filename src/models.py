"""
Общие структуры данных (dataclasses), которые используются
сразу несколькими модулями: traction, interlocking, renderers, simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# ПЕРЕЧИСЛЕНИЯ — замена строковых литералов
# ---------------------------------------------------------------------------

class CurrentType(str, Enum):
    DC = "DC"
    AC = "AC"


class RouteType(str, Enum):
    ARRIVAL    = "arrival"
    DEPARTURE  = "departure"
    PASSTHROUGH = "passthrough"


class SwitchPosition(str, Enum):
    PLUS  = "plus"
    MINUS = "minus"


class ConflictReason(str, Enum):
    TOPOLOGY   = "topology"
    EXTRA_RULE = "extra_rule"
    OK         = "ok"


class EventType(str, Enum):
    """Типы событий для логирования работы станции."""
    ARRIVED = "arrived_at_station"
    ROUTE_REQUESTED = "route_requested"
    ROUTE_ACQUIRED = "route_acquired"
    DEPARTED = "departed_station"
    ROUTE_RELEASED = "route_released"


class ControlMode(str, Enum):
    """Режим СИРДП (Control mode): АБ или ВС."""
    AB = "AB"   # автоблокировка (фиксированные интервалы)
    VC = "VC"   # виртуальная сцепка (сокращённые интервалы)

    @property
    def label_ru(self) -> str:
        """Человекочитаемое название для отчёта."""
        return {
            "AB": "Автоблокировка (АБ)",
            "VC": "Виртуальная сцепка (ВС)",
        }[self.value]


class VCMethodology(str, Enum):
    """Методика работы станции в режиме ВС."""
    A = "A"   # базовая ВС
    B = "B"   # специальный режим (Бушуев-Голочалов)

    @property
    def label_ru(self) -> str:
        return {
            "A": "Методика А (Базовая ВС)",
            "B": "Методика Б (Спец. режим)",
        }[self.value]


# ---------------------------------------------------------------------------
# ТЯГА — traction/loader.py → traction/engine.py → traction/dynamics.py
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class LocomotiveConfig:
    """Параметры локомотива, загруженные из YAML."""
    loco_id: str
    name: str
    current_type: CurrentType         # DC | AC
    voltage: int                      # В: 3000 или 25000
    mass_t: float                     # масса локомотива, т
    loco_length_m: float              # длина локомотива, м
    num_axes: int
    v_table: np.ndarray               # скорости для тяговой характеристики, км/ч
    fk_table: np.ndarray              # сила тяги Fk(v), кН
    wox_table: np.ndarray             # удельное сопротивление движению wox(v)
    bt_table: np.ndarray              # удельная тормозная сила bT(v)
    v_max: float                      # конструкционная скорость, км/ч

    def __post_init__(self) -> None:
        if self.voltage not in {3000, 25000}:
            raise ValueError(
                f"voltage должен быть 3000 или 25000, получено {self.voltage}"
            )
        if self.mass_t <= 0:
            raise ValueError(f"mass_t должна быть > 0, получено {self.mass_t}")
        n = len(self.v_table)
        for name, arr in (
            ("fk_table", self.fk_table),
            ("wox_table", self.wox_table),
            ("bt_table", self.bt_table),
        ):
            if len(arr) != n:
                raise ValueError(
                    f"Длина {name} ({len(arr)}) не совпадает с v_table ({n})"
                )

    @property
    def locolengthm(self) -> float:
        """Compatibility alias for legacy PTR-style formulas."""
        return self.loco_length_m

    @property
    def bttable(self) -> np.ndarray:
        """Compatibility alias for locomotive braking table."""
        return self.bt_table


@dataclass
class TrainConfig:
    """Параметры состава (локомотив + вагоны) для тягового расчёта."""
    consist_id: str
    loco: LocomotiveConfig
    num_wagons: int
    wagon_mass_t: float               # масса одного гружёного вагона, т
    wagon_length_m: float             # длина одного вагона, м
    q0: float                         # нагрузка от оси на рельс, т/ось
    wagon_type: int                   # тип вагона по ПТР (4, 6, ...)
    bt_wagons_table: np.ndarray | None = None  # удельная тормозная сила вагонов bT(v)

    def __post_init__(self) -> None:
        if self.num_wagons <= 0:
            raise ValueError(
                f"num_wagons должен быть > 0, получено {self.num_wagons}"
            )
        if self.wagon_mass_t <= 0:
            raise ValueError(
                f"wagon_mass_t должна быть > 0, получено {self.wagon_mass_t}"
            )
        if self.wagon_type not in {4, 6, 8}:
            raise ValueError(
                f"wagon_type должен быть 4, 6 или 8, получено {self.wagon_type}"
            )
        if self.bt_wagons_table is not None and len(self.bt_wagons_table) != len(self.loco.v_table):
            raise ValueError(
                "Длина bt_wagons_table "
                f"({len(self.bt_wagons_table)}) не совпадает с v_table "
                f"({len(self.loco.v_table)})"
            )

    @property
    def train_mass_t(self) -> float:
        """Полная масса состава, т."""
        return self.loco.mass_t + self.num_wagons * self.wagon_mass_t

    @property
    def train_length_m(self) -> float:
        """Полная длина состава: локомотив + вагоны, м."""
        return self.loco.loco_length_m + self.num_wagons * self.wagon_length_m

    @property
    def trainlengthm(self) -> float:
        """Compatibility alias requested by PTR-facing calculations."""
        return self.train_length_m

    @property
    def btwagons_bttable(self) -> np.ndarray | None:
        """Compatibility alias for wagon braking table."""
        return self.bt_wagons_table


@dataclass
class RouteSection:
    """Один участок маршрута: координаты, уклон, кривая, ограничение скорости."""
    section_id: str
    s_start: float                    # начало участка, м
    s_end: float                      # конец участка, м
    grade: float                      # уклон i, ‰ (+ подъём, − спуск)
    radius: float = 0.0               # радиус кривой R, м (0 = прямая)
    v_limit: float = 120.0            # ограничение скорости, км/ч

    def __post_init__(self) -> None:
        if self.s_start >= self.s_end:
            raise ValueError(
                f"s_start ({self.s_start}) должен быть < s_end ({self.s_end})"
            )
        if self.radius < 0:
            raise ValueError(
                f"radius должен быть >= 0, получено {self.radius}"
            )
        if self.v_limit <= 0:
            raise ValueError(
                f"v_limit должен быть > 0, получено {self.v_limit}"
            )

    @property
    def length_m(self) -> float:
        """Длина участка, м."""
        return self.s_end - self.s_start


@dataclass(eq=False)
class PhysicsResult:
    """
    Результат интегрирования уравнения движения (dynamics.py → solveivp).
    Хранится в TractionCache и передаётся в SimPy-процесс.
    """
    consist_id: str
    route_id: str
    t_total_s: float                  # полное время хода по маршруту, с
    v_profile: np.ndarray             # V(s), км/ч
    t_profile: np.ndarray             # t(s), с — время прибытия головы
    s_points: np.ndarray              # точки s, м — общая ось абсцисс
    head_to_tail_s: np.ndarray        # Δt(s): хвост минус голова, с


# ---------------------------------------------------------------------------
# ЭЛЕКТРИЧЕСКАЯ ЦЕНТРАЛИЗАЦИЯ — interlocking/
# ---------------------------------------------------------------------------

@dataclass
class SwitchConfig:
    """Стрелочный перевод."""
    switch_id: str
    normal_position: SwitchPosition
    transfer_time_s: float = 4.0     # время перевода, с


@dataclass
class RouteConfig:
    """Маршрут приёма/отправления из YAML станции."""
    route_id: str
    name: str
    route_type: RouteType
    sections: list[str]               # id секций в порядке занятия
    switches: dict[str, SwitchPosition]  # {switch_id: SwitchPosition}
    v_limit: float = 60.0            # скорость по маршруту, км/ч


@dataclass
class JointConfig:
    """Стык путевого развития станции."""
    joint_id: str
    kind: str = ""
    interval_base_min: float = 0.0
    interval_vc_min: float = 0.0


@dataclass
class StationConfig:
    """Конфигурация станции, загруженная из YAML."""
    station_id: str
    name: str
    routes: dict[str, RouteConfig]
    switches: dict[str, SwitchConfig]
    extra_conflicts: list[tuple[str, str]] = field(default_factory=list)
    joints: dict[str, JointConfig] = field(default_factory=dict)


@dataclass
class RouteConflictResult:
    """Результат проверки враждебности двух маршрутов."""
    route_a: str
    route_b: str
    is_hostile: bool
    reason: ConflictReason


class RouteStatus(Enum):
    """Статус маршрута в движке."""

    CLOSED = "CLOSED"       # маршрут не открыт
    REQUESTED = "REQUESTED" # запрос принят, стрелки ещё переводятся
    OPEN = "OPEN"           # маршрут открыт, движение разрешено
    CANCELLING = "CANCELLING" # отмена в процессе


class SwitchState(Enum):
    """Текущее фактическое положение стрелки."""

    NORMAL = "NORMAL"       # нормальное положение
    REVERSE = "REVERSE"     # переведена
    MOVING = "MOVING"       # в процессе перевода
    LOCKED = "LOCKED"       # заперта (входит в активный маршрут)
    FAULT = "FAULT"         # неисправность


@dataclass
class EngineState:
    """Снимок текущего состояния всех объектов станции.

    Возвращается методом :meth:`InterlockingEngine.get_state`.
    Объект иммутабелен — изменения состояния отражаются только
    в следующем вызове ``get_state()``.
    """

    switch_states: dict[str, SwitchState] = field(default_factory=dict)
    route_statuses: dict[str, RouteStatus] = field(default_factory=dict)
    active_routes: list[str] = field(default_factory=list)


@dataclass
class RouteRequest:
    """Запрос на открытие маршрута.

    Args:
        route_id: идентификатор маршрута из конфигурации станции.
        train_id: идентификатор поезда, запрашивающего маршрут.
        priority: приоритет запроса (выше — важнее при коллизиях).
            По умолчанию 0.
        platoon_id: идентификатор пакета (опционально).
    """

    route_id: str
    train_id: str = ""
    priority: int = 0
    platoon_id: str | None = None
    batch_routes: list[str] | None = None  # все маршруты пакета для предварительного резервирования


# ---------------------------------------------------------------------------
# СИМУЛЯЦИЯ — simulation.py / SimPy
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    """Текущее положение поезда в SimPy-процессе."""
    train_id: str
    x_head: float                     # координата головы, м
    x_tail: float                     # координата хвоста, м
    v_kmh: float = 0.0               # текущая скорость, км/ч

    @classmethod
    def from_physics(
        cls,
        train_id: str,
        s: float,
        physics: PhysicsResult,
        train_length_m: float,
    ) -> TrainState:
        x_head = float(s)             # координата головы = позиция по пути, м
        v_kmh  = float(np.interp(s, physics.s_points, physics.v_profile))
        return cls(
            train_id=train_id,
            x_head=x_head,
            x_tail=x_head - train_length_m,
            v_kmh=v_kmh,
        )


@dataclass
class StationEvent:
    """Одно событие по поезду на станции в ходе симуляции."""
    train_id: str
    event_type: EventType
    route_id: str
    section_id: str                   # Может быть пустым, если событие на уровне маршрута
    t_event_s: float
    phase: str = ""
    reason: str = ""
    queue_length: int = 0
    route_status: str = ""


@dataclass
class SimResult:
    """Итоговая запись по одному поезду за одну симуляцию."""
    train_id: str
    route_id: str
    consist_id: str
    scenario: str                     # "Demo-AB" | "Demo-VC"
    t_arrive_s: float                 # время прибытия на станцию фактически, с
    t_depart_s: float                 # время отправления фактически, с
    t_wait_s: float                   # ожидание ресурса (горловина), с
    t_dwell_s: float                  # стоянка фактическая, с
    t_total_s: float                  # полное время от входа до выхода, с
    control_mode: str = "AB"          # режим СИРДП: "AB" | "VC"
    t_planned_arrive_s: float = 0.0   # плановое время прибытия
    t_planned_depart_s: float = 0.0   # плановое время отправления
    delay_arrive_s: float = 0.0       # задержка по прибытию
    delay_depart_s: float = 0.0       # задержка по отправлению
    v_avg_kmh: float = 0.0            # средняя скорость по маршруту (из профиля тяги)
    v_max_kmh: float = 0.0            # максимальная скорость по маршруту (из профиля тяги)
    vc_methodology: str = "A"         # методика ВС: "A" или "B"
    platoon_id: str | None = None     # ID пакета (если есть)
    departure_route_id: str | None = None # маршрут отправления
    t_scheduled_arrive_s: float = 0.0
    t_actual_arrive_s: float = 0.0
    t_route_acquired_arrive_s: float = 0.0
    t_arrival_route_clear_s: float = 0.0
    t_depart_ready_s: float = 0.0
    t_actual_depart_s: float = 0.0
    t_final_clear_s: float = 0.0
    wait_arrival_route_s: float = 0.0
    wait_departure_route_s: float = 0.0
    route_setup_wait_s: float = 0.0
    arrival_run_s: float = 0.0
    departure_run_s: float = 0.0
    tail_clearance_s: float = 0.0


@dataclass
class ScenarioEntry:
    """Один поезд в сценарии симуляции.

    Описывает: когда поезд прибывает, какой состав, по какому маршруту
    и через какие физические секции движется.
    """
    train_id: str
    t_arrive_s: float                 # момент появления поезда в симуляции, с
    route_id: str                     # маршрут в StationConfig
    train: TrainConfig                # состав (локомотив + вагоны)
    sections: list[RouteSection]      # физические секции для тягового расчёта (или только для приёма)
    v0_kmh: float = 0.0              # начальная скорость, км/ч
    dwell_s: float = 0.0             # время стоянки на станции, с
    platoon_id: str | None = None    # ID пакета (None = одиночный поезд)
    departure_route_id: str | None = None # маршрут отправления (для методики Б)
    departure_sections: list[RouteSection] | None = None # секции для отправления
    delay_s: float = 0.0
    
    @property
    def planned_depart_s(self) -> float:
        """Плановое время отправления без учета задержек (но без учета времени хода, так как оно неизвестно до расчёта, 
        считаем как прибытие + стоянка, хотя точнее было бы считать с учетом времени следования до места стоянки).
        Пока принимаем t_arrive_s + dwell_s как базовую метрику для отправления, чтобы считать задержку относительно нее."""
        return self.t_arrive_s + self.dwell_s
