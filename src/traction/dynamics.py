"""
Интегрирование уравнения движения поезда по ПТР РЖД 2016.

Модуль не знает ни о SimPy, ни о станции — только физика и SciPy.
Публичный API: solve_route, TractionCache, head_to_tail_profile.
"""
from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp

from src.models import PhysicsResult, RouteSection, TrainConfig

logger = logging.getLogger(__name__)

# Режим движения: тяга / выбег / торможение
DriveMode = Literal["traction", "coasting", "braking"]

# Перевод единиц
_KMH_TO_MS: float = 1.0 / 3.6
_MS_TO_KMH: float = 3.6

# Ускорение свободного падения, м/с²
_G: float = 9.81

# Коэффициент инерции вращающихся масс для грузового состава (ПТР РЖД 2016, §1.1)
_GAMMA_FREIGHT: float = 0.06


# ---------------------------------------------------------------------------
# Блок 1 — Силовые функции
# ---------------------------------------------------------------------------

def _fk_kn(v_kmh: float, train: TrainConfig) -> float:
    """
    Расчётная сила тяги Fk(v), кН.

    Интерполирует по fk_table локомотива.
    При v >= v_max → 0.0 (локомотив не может тянуть быстрее конструкционной скорости).
    При v <= 0     → fk_table[0] (значение при трогании с места).

    fk_table уже является min(тяговая, сцепная) — это сделано в loader.py.
    ПТР РЖД 2016, §1.1.
    """
    if v_kmh <= 0.0:
        return float(train.loco.fk_table[0])
    if v_kmh >= train.loco.v_max:
        return 0.0
    return float(np.interp(v_kmh, train.loco.v_table, train.loco.fk_table))


def _wox_kn(v_kmh: float, train: TrainConfig) -> float:
    """
    Основное удельное сопротивление движению локомотива wox(v), Н/кН.

    Интерполирует по wox_table. Коэффициенты таблицы рассчитаны в loader.py
    по формулам 19–20 ПТР РЖД 2016, §1.2, табл. 2.
    """
    return float(np.interp(v_kmh, train.loco.v_table, train.loco.wox_table))


# Коэффициенты формул удельного сопротивления вагонов (ПТР РЖД 2016, §1.2, табл. 5–6).
# Формула: wo' = a + (b + c·v + d·v²) / q0
# "heavy": q0 >= 6 т/ось (гружёный), "light": q0 < 6 т/ось (порожний).
_WO_WAGON_COEFFS: dict[int, dict[str, tuple[float, float, float, float]]] = {
    4: {
        "heavy": (0.7,  3.0, 0.1,  0.0025),  # ПТР РЖД 2016, §1.2, табл. 5, форм. 3 (гружёный, ПК)
        "light": (0.7,  8.0, 0.1,  0.0025),  # ПТР РЖД 2016, §1.2, табл. 5, форм. 1 (порожний, ПК)
    },
    6: {
        "heavy": (0.7,  4.0, 0.1,  0.0020),  # ПТР РЖД 2016, §1.2, табл. 5, форм. 5 (6-осн., гружёный)
        "light": (0.7,  9.0, 0.1,  0.0020),  # ПТР РЖД 2016, §1.2, табл. 5, форм. 4 (6-осн., порожний)
    },
    8: {
        "heavy": (0.7,  4.5, 0.12, 0.0018),  # ПТР РЖД 2016, §1.2, табл. 6, форм. 6 (8-осн., гружёный)
        "light": (0.7,  9.5, 0.12, 0.0018),  # ПТР РЖД 2016, §1.2, табл. 6, форм. 6 (8-осн., порожний)
    },
}


def _wo_wagons_kn(v_kmh: float, train: TrainConfig) -> float:
    """
    Основное удельное сопротивление движению вагонов wo'(v), Н/кН.

    Формулы ПТР РЖД 2016, §1.2, табл. 5–6 (формулы 1–6):
        wo' = a + (b + c·v + d·v²) / q0

    Коэффициенты a, b, c, d зависят от типа вагона (wagon_type) и нагрузки:
        q0 >= 6 т/ось — гружёный (\"heavy\"),
        q0  < 6 т/ось — порожний (\"light\").

    Поддерживаемые wagon_type: 4, 6, 8 (по числу осей).
    """
    if train.wagon_type not in _WO_WAGON_COEFFS:
        raise ValueError(
            f"Неизвестный тип вагона wagon_type={train.wagon_type}. "
            f"Допустимые значения: {sorted(_WO_WAGON_COEFFS)}"
        )
    bucket = "heavy" if train.q0 >= 6.0 else "light"
    a, b, c, d = _WO_WAGON_COEFFS[train.wagon_type][bucket]
    v = v_kmh
    return a + (b + c * v + d * v ** 2) / train.q0


def _wi_kn(grade_per_mill: float, train_mass_t: float) -> float:
    """
    Сила сопротивления от уклона Wi, кН.

    ПТР РЖД 2016, §1.2.6, формула 67:
        wi = i  [Н/кН]  (i — уклон в ‰, g в формулу не входит)
        Wi = wi · (P + Q) / 1000  [кН]

    Знак Wi совпадает со знаком уклона i:
        i > 0 — подъём (сопротивление),
        i < 0 — спуск (ускоряющая сила, добавляется со знаком минус).
    """
    return grade_per_mill * train_mass_t / 1000.0


def _wr_kn(radius_m: float, train_mass_t: float) -> float:
    """
    Сила сопротивления от кривой Wr, кН.

    ПТР РЖД 2016, §1.2.7, формула 68:
        wr = 700 / R  [Н/кН]  (коэффициент 700 уже включает g)
        Wr = wr · (P + Q) / 1000  [кН]

    При radius_m == 0 или None (прямой путь) возвращает 0.
    """
    if not radius_m:
        return 0.0
    return (700.0 / radius_m) * train_mass_t / 1000.0


def _w_full_kn(
    v_kmh: float,
    train: TrainConfig,
    section: RouteSection,
) -> float:
    """
    Полное сопротивление движению W, кН.

    W = Wox + Wo_wagons + Wi + Wr.

    Пересчёт удельных сопротивлений (Н/кН) в абсолютные силы (кН):
        Wox      = wox · P / 1000
        Wo_wagon = wo' · Q / 1000
        Wi       = i · (P+Q) / 1000         (формула 67)
        Wr       = (700/R) · (P+Q) / 1000   (формула 68)

    Где P — масса локомотива (т), Q — масса вагонов (т).
    """
    wagon_mass_total_t = train.num_wagons * train.wagon_mass_t

    wox  = _wox_kn(v_kmh, train)       * train.loco.mass_t      / 1000.0
    wo_w = _wo_wagons_kn(v_kmh, train) * wagon_mass_total_t      / 1000.0
    wi   = _wi_kn(section.grade,  train.train_mass_t)
    wr   = _wr_kn(section.radius, train.train_mass_t)
    return wox + wo_w + wi + wr


def _bt_full_kn(v_kmh: float, train: TrainConfig) -> float:
    """
    Расчётная тормозная сила Bt(v), кН.

    Интерполирует bt_table по текущей скорости.
    При v <= 0 поезд уже стоит — тормозная сила равна нулю.
    ПТР РЖД 2016, §2.2.

    ВНИМАНИЕ: учитывается только тормозная сила локомотива.
    Тормозная сила вагонов (ПТР РЖД 2016, §2.2) не реализована — TODO.
    """
    if v_kmh <= 0.0:
        return 0.0
    return float(np.interp(v_kmh, train.loco.v_table, train.loco.bt_table))


# ---------------------------------------------------------------------------
# Блок 2 — ODE и интегрирование
# ---------------------------------------------------------------------------

def _current_section(
    s_m: float,
    sections: list[RouteSection],
    s_ends: list[float],
) -> RouteSection:
    """
    Возвращает RouteSection, которому принадлежит координата s_m.

    Поиск через bisect_right по предрассчитанному списку s_ends — O(log n).
    Граничные случаи:
        s_m <  sections[0].s_start → WARNING + возвращает sections[0],
        s_m >= sections[-1].s_end  → WARNING (если вышел за конец) + sections[-1].

    Параметр s_ends передаётся из solve_route (предрассчитан один раз),
    чтобы избежать аллокации на каждом шаге ODE.
    """
    if s_m < sections[0].s_start:
        logger.warning("_current_section: s=%.2f вне маршрута (до начала)", s_m)
        return sections[0]
    idx = bisect.bisect_right(s_ends, s_m)
    if idx >= len(sections):
        if s_m > s_ends[-1]:
            logger.warning("_current_section: s=%.2f вне маршрута (после конца)", s_m)
        return sections[-1]
    return sections[idx]


def _ode(
    t: float,  # noqa: ARG001 — время нужно для совместимости с solve_ivp API
    y: list[float],
    train: TrainConfig,
    sections: list[RouteSection],
    mode: DriveMode,
    s_ends: list[float],
) -> list[float]:
    """
    Правая часть уравнения движения поезда для scipy.integrate.solve_ivp.

    Вектор состояния: y = [v_ms, s_m].
    Возвращает: [dv/dt, ds/dt].

    Уравнение движения (ПТР РЖД 2016, §1.1, формула 1):
        dv/dt = (Fk − W − Bt) / (M · (1 + γ))

    Где:
        Fk — сила тяги, кН      (0 при выбеге и торможении, а также при v >= v_limit)
        W  — полное сопротивление, кН
        Bt — тормозная сила, кН (0 при тяге и выбеге)
        M  — масса состава, т   (1 кН/т = 1 м/с²)
        γ  = 0,06 — коэффициент инерции вращающихся масс (§1.1)

    ds/dt = v_ms.

    Ограничение скорости секции:
        Если v_kmh >= section.v_limit — тяга обнуляется.
        Поезд выкатывается выбегом до v < v_limit, после чего тяга возобновляется.
        Это имитирует работу САУТ/КЛУБ без явного регулятора скорости.
    """
    v_ms  = max(y[0], 0.0)  # защита от численного ухода в отрицательные скорости
    s_m   = y[1]
    v_kmh = v_ms * _MS_TO_KMH

    section = _current_section(s_m, sections, s_ends)

    # Ограничитель скорости секции: обнуляем тягу, если скорость достигла v_limit.
    # Поле v_limit присутствует в RouteSection (models.py, дефолт 120 км/ч).
    over_limit = (mode == "traction") and (v_kmh >= section.v_limit)

    fk = _fk_kn(v_kmh, train) if (mode == "traction" and not over_limit) else 0.0
    bt = _bt_full_kn(v_kmh, train) if mode == "braking"  else 0.0
    w  = _w_full_kn(v_kmh, train, section)

    m_eff = train.train_mass_t * (1.0 + _GAMMA_FREIGHT)  # т (эффективная масса)

    # кН / т = м/с²  (1 кН = 1000 Н = 1000 кг·м/с², 1 т = 1000 кг)
    dv_dt = (fk - w - bt) / m_eff
    ds_dt = v_ms
    return [dv_dt, ds_dt]


def _make_events(sections: list[RouteSection]) -> list[Callable]:
    """
    Формирует список терминальных событий для solve_ivp.

    Событие 1 — конец маршрута: s >= s_end последнего участка.
    Событие 2 — остановка:      v <= 0.

    Оба события терминальные (terminal=True), интегрирование прекращается.

    Ограничения скорости секций реализованы непосредственно в _ode
    через RouteSection.v_limit — отдельное терминальное событие не нужно:
    обнуление Fk достаточно для контроля скорости в задачах ВКР.
    """
    s_route_end = sections[-1].s_end

    def event_end_of_route(t: float, y: list[float]) -> float:  # noqa: ARG001
        return y[1] - s_route_end

    event_end_of_route.terminal  = True   # type: ignore[attr-defined]
    event_end_of_route.direction = 1      # type: ignore[attr-defined]  # только при росте s

    def event_stop(t: float, y: list[float]) -> float:  # noqa: ARG001
        return y[0]  # = v_ms; пересекает 0 при остановке

    event_stop.terminal  = True   # type: ignore[attr-defined]
    event_stop.direction = -1     # type: ignore[attr-defined]  # только при убывании v

    return [event_end_of_route, event_stop]


def solve_route(
    train: TrainConfig,
    sections: list[RouteSection],
    consist_id: str,
    route_id: str,
    v0_kmh: float = 0.0,
    mode: DriveMode = "traction",
    t_max_s: float = 7200.0,
) -> PhysicsResult:
    """
    Интегрирует уравнение движения по маршруту, возвращает PhysicsResult.

    Параметры:
        train      — конфигурация состава
        sections   — участки маршрута (непрерывная последовательность)
        consist_id — идентификатор состава (для кэша и логов)
        route_id   — идентификатор маршрута
        v0_kmh     — начальная скорость, км/ч (0 = трогание с места)
        mode       — режим: тяга / выбег / торможение
        t_max_s    — верхняя граница интегрирования, с (защита от зависания)

    Возвращает:
        PhysicsResult с заполненными v_profile, t_profile, s_points, head_to_tail_s.

    Метод: RK45 (scipy.integrate.solve_ivp).
    """
    _validate_sections(sections)

    y0     = [v0_kmh * _KMH_TO_MS, sections[0].s_start]
    events = _make_events(sections)

    # Предрассчитываем s_ends один раз, чтобы не аллоцировать список
    # на каждом шаге ODE (_current_section вызывается тысячи раз).
    s_ends_cache = [sec.s_end for sec in sections]

    sol = solve_ivp(
        fun          = lambda t, y: _ode(t, y, train, sections, mode, s_ends_cache),
        t_span       = (0.0, t_max_s),
        y0           = y0,
        method       = "RK45",
        events       = events,
        dense_output = False,
        max_step     = 1.0,     # шаг ≤1 с для достаточной точности v-профиля
        rtol         = 1e-4,
        atol         = 1e-6,
    )

    if sol.status == -1:
        raise ValueError(
            f"solve_ivp не сошёлся для consist='{consist_id}', route='{route_id}': "
            f"{sol.message}"
        )

    # sol.status == 0 — достигнут t_max без события (нештатно, но не ошибка интегратора)
    # sol.status == 1 — остановлено событием (штатный конец маршрута или остановка)
    s_points  = sol.y[1]
    v_profile = sol.y[0] * _MS_TO_KMH
    t_profile = sol.t
    t_total_s = float(t_profile[-1])

    # Профиль задержки хвоста относительно головы
    ht_profile = head_to_tail_profile(
        # временный объект только для удобства сигнатуры
        PhysicsResult(
            consist_id=consist_id,
            route_id=route_id,
            t_total_s=t_total_s,
            v_profile=v_profile,
            t_profile=t_profile,
            s_points=s_points,
            head_to_tail_s=np.zeros(0),  # пустой массив вместо zeros_like
        ),
        train.train_length_m,
    )

    result = PhysicsResult(
        consist_id     = consist_id,
        route_id       = route_id,
        t_total_s      = t_total_s,
        v_profile      = v_profile,
        t_profile      = t_profile,
        s_points       = s_points,
        head_to_tail_s = ht_profile,
    )

    logger.info(
        "solve_route: consist='%s', route='%s', mode=%s, "
        "t_total=%.1f с, точек=%d, v_max=%.1f км/ч",
        consist_id, route_id, mode,
        t_total_s, len(s_points), float(np.max(v_profile)),
    )
    return result


# ---------------------------------------------------------------------------
# Блок 3 — Кэш предрассчитанных результатов
# ---------------------------------------------------------------------------

@dataclass
class TractionCache:
    """
    Кэш результатов solve_route.

    Один экземпляр создаётся в main.py и передаётся в SimPy-процессы.
    Ключ: "{consist_id}:{route_id}:{mode}:{v0_kmh}" — учитывает все параметры,
    влияющие на результат интегрирования.
    Повторный вызов get_or_compute с теми же параметрами не перезапускает ODE.
    """
    _store: dict[str, PhysicsResult] = field(default_factory=dict)

    def get_or_compute(
        self,
        train: TrainConfig,
        sections: list[RouteSection],
        consist_id: str,
        route_id: str,
        v0_kmh: float = 0.0,
        mode: DriveMode = "traction",
    ) -> PhysicsResult:
        """
        Возвращает кэшированный результат или вычисляет через solve_route.

        При промахе логирует INFO с ключом и временем расчёта.
        """
        key = f"{consist_id}:{route_id}:{mode}:{v0_kmh}"
        if key not in self._store:
            logger.info("TractionCache: miss — вычисляем '%s'", key)
            t0 = time.perf_counter()
            self._store[key] = solve_route(
                train, sections, consist_id, route_id, v0_kmh, mode
            )
            elapsed = time.perf_counter() - t0
            logger.info("TractionCache: '%s' посчитан за %.3f с", key, elapsed)
        return self._store[key]

    def invalidate(self, consist_id: str, route_id: str) -> None:
        """
        Удаляет все записи с данной парой consist_id / route_id.

        Молча игнорирует отсутствующие ключи. Используется при смене
        конфигурации состава без перезапуска симуляции.
        """
        prefix = f"{consist_id}:{route_id}:"
        keys_to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._store[k]
        if keys_to_delete:
            logger.debug(
                "TractionCache: удалено %d записей для '%s:%s'",
                len(keys_to_delete), consist_id, route_id,
            )

    def clear(self) -> None:
        """Очищает весь кэш."""
        self._store.clear()
        logger.debug("TractionCache: кэш очищен.")

    def __len__(self) -> int:
        """Количество закэшированных результатов."""
        return len(self._store)


# ---------------------------------------------------------------------------
# Блок 4 — Профиль голова / хвост
# ---------------------------------------------------------------------------

def head_to_tail_profile(
    physics: PhysicsResult,
    train_length_m: float,
) -> np.ndarray:
    """
    Строит массив Δt(s) — запаздывание хвоста поезда относительно головы, с.

    Δt(s) = t_head(s) − t_head(s − L),

    где L = train_length_m — полная длина состава (м),
    t_head(s) — момент прохождения головой точки s.

    Используется в SimPy-процессе для определения момента освобождения секции:
    секция освобождается, когда хвост поезда прошёл её конец.

    При s − L < s_points[0] (хвост ещё не вошёл на маршрут) полагаем
    t_tail = 0, т.е. Δt = t_head(s) — хвост задерживается на всё время
    хода головы от начала маршрута.

    Возвращает np.ndarray той же длины, что physics.s_points.
    """
    s = physics.s_points
    t = physics.t_profile

    # Координата хвоста в момент, когда голова находится в точке s[i]
    s_tail = s - train_length_m

    # left=0.0: при s_tail < s[0] хвост ещё не вошёл на маршрут → t_tail = 0
    t_tail  = np.interp(s_tail, s, t, left=0.0)
    delta_t = t - t_tail
    return delta_t


# ---------------------------------------------------------------------------
# Вспомогательная проверка непрерывности маршрута
# ---------------------------------------------------------------------------

def _validate_sections(sections: list[RouteSection]) -> None:
    """
    Проверяет, что секции образуют непрерывный маршрут без разрывов.

    Бросает ValueError если:
    - список пуст,
    - конец i-й секции не совпадает (с точностью 1e-6 м) с началом (i+1)-й.
    """
    if not sections:
        raise ValueError("Список секций пуст.")
    for i in range(len(sections) - 1):
        if not np.isclose(sections[i].s_end, sections[i + 1].s_start, atol=1e-6):
            raise ValueError(
                f"Разрыв между секциями {i} и {i + 1}: "
                f"{sections[i].s_end} != {sections[i + 1].s_start}"
            )
