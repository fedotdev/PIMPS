"""Тесты для src/traction/dynamics.py — физика тягового расчёта."""
from __future__ import annotations

import numpy as np
import pytest

from src.models import (
    CurrentType,
    LocomotiveConfig,
    PhysicsResult,
    RouteSection,
    TrainConfig,
)
from src.traction.dynamics import (
    TractionCache,
    _bt_full_kn,
    _validate_sections,
    _wi_kn,
    _wo_wagons_kn,
    _wr_kn,
    apply_speed_limit,
    head_to_tail_profile,
    solve_route,
)


@pytest.fixture
def test_loco() -> LocomotiveConfig:
    """Фикстура локомотива с простыми таблицами характеристик."""
    return LocomotiveConfig(
        loco_id="TEST",
        name="Тяговый тест",
        current_type=CurrentType.AC,
        voltage=25000,
        mass_t=100.0,
        loco_length_m=20.0,
        num_axes=4,
        v_table=np.array([0.0, 50.0, 100.0]),
        fk_table=np.array([300.0, 200.0, 100.0]),
        wox_table=np.array([2.0, 3.0, 4.0]),
        bt_table=np.array([50.0, 50.0, 50.0]),
        v_max=100.0,
    )


@pytest.fixture
def test_train(test_loco: LocomotiveConfig) -> TrainConfig:
    """Фикстура состава: 10 вагонов по 50 т."""
    return TrainConfig(
        consist_id="T1",
        loco=test_loco,
        num_wagons=10,
        wagon_mass_t=50.0,
        wagon_length_m=15.0,
        q0=12.5,  # гружёный (>6 т/ось)
        wagon_type=4,
    )


# ---------------------------------------------------------------------------
# Тесты вспомогательных физических функций
# ---------------------------------------------------------------------------

class TestPhysicsFunctions:
    """Модульные тесты физических формул ПТР и валидации."""

    def test_validate_sections_empty(self):
        """Пустой список секций → ValueError."""
        with pytest.raises(ValueError, match="пуст"):
            _validate_sections([])

    def test_validate_sections_gap(self):
        """Разрыв координат между секциями → ValueError."""
        s1 = RouteSection("S1", 0.0, 100.0, 0.0)
        s2 = RouteSection("S2", 100.5, 200.0, 0.0)  # Разрыв 0.5 метра
        with pytest.raises(ValueError, match="Разрыв"):
            _validate_sections([s1, s2])

    def test_wi_kn(self):
        """Сопротивление от уклона Wi."""
        # Уклон 10 ‰, масса 1000 т -> 10 * 1000 * g / 1000 = 98.1 кН
        assert _wi_kn(10.0, 1000.0) == pytest.approx(98.1)
        # Спуск -5 ‰, масса 1000 т -> -49.05 кН
        assert _wi_kn(-5.0, 1000.0) == pytest.approx(-49.05)

    def test_wi_zero_grade(self):
        """Нулевой уклон не дает дополнительного сопротивления."""
        assert _wi_kn(0.0, 1000.0) == 0.0

    def test_wr_kn(self):
        """Сопротивление от кривой Wr."""
        # 700 / R * M * g / 1000
        # радиус 350 м, масса 1000 т -> 700/350 * 9.81 = 19.62 кН
        assert _wr_kn(350.0, 1000.0) == pytest.approx(19.62)
        # Без радиуса -> 0
        assert _wr_kn(0.0, 1000.0) == 0.0

    def test_wr_infinite_radius(self):
        """Бесконечный радиус эквивалентен прямому пути."""
        assert _wr_kn(float("inf"), 1000.0) == 0.0

    def test_wowagons_all_types(self, test_train: TrainConfig):
        """Сопротивление вагонов поддерживает 4-, 6- и 8-осные типы."""
        for wagon_type in (4, 6, 8):
            test_train.wagon_type = wagon_type
            test_train.q0 = 8.0
            assert _wo_wagons_kn(50.0, test_train) > 0.0
            test_train.q0 = 5.0
            assert _wo_wagons_kn(50.0, test_train) > 0.0

    def test_wo_wagons_unknown_type(self, test_train: TrainConfig):
        """Неизвестный тип вагона → ValueError."""
        test_train.wagon_type = 10
        with pytest.raises(ValueError, match="Неизвестный тип вагона"):
            _wo_wagons_kn(50.0, test_train)

    def test_btfull_loco_plus_wagons(self, test_train: TrainConfig):
        """Полное торможение складывает локомотивную и вагонную составляющие."""
        test_train.bt_wagons_table = np.array([2.0, 2.0, 2.0])
        expected = 50.0 + 2.0 * (10 * 50.0) * 9.81 / 1000.0
        assert _bt_full_kn(50.0, test_train) == pytest.approx(expected)

    def test_speed_limit_cuts_fk(self):
        """Тяга обнуляется при достижении ограничения скорости."""
        assert apply_speed_limit(100.0, v_ms=20.0, v_limit_ms=20.0) == 0.0
        assert apply_speed_limit(100.0, v_ms=19.9, v_limit_ms=20.0) == 100.0

    def test_trainlength_property(self, test_train: TrainConfig):
        """Совместимое свойство trainlengthm возвращает полную длину поезда."""
        assert test_train.trainlengthm == pytest.approx(test_train.train_length_m)


# ---------------------------------------------------------------------------
# Интеграционные тесты (ODE Solver)
# ---------------------------------------------------------------------------

class TestSolveRoute:
    """Тестирование интегрирования уравнения движения."""

    def test_solve_route_basic(self, test_train: TrainConfig):
        """Базовый успешный проезд участка длиннее состава."""
        s1 = RouteSection("S1", 0.0, 200.0, 0.0)
        res = solve_route(
            train=test_train,
            sections=[s1],
            consist_id=test_train.consist_id,
            route_id="R1",
            t_max_s=60.0,  # Защита от долгого интегрирования
        )
        
        assert res.consist_id == "T1"
        assert res.route_id == "R1"
        assert res.t_total_s > 0.0
        assert len(res.s_points) >= 2
        
        # Размеры массивов профиля совпадают
        assert len(res.v_profile) == len(res.s_points)
        assert len(res.t_profile) == len(res.s_points)
        assert len(res.head_to_tail_s) == len(res.s_points)
        
        # Интегрирование остановило событие конца маршрута (s ≈ 200.0)
        assert res.s_points[-1] == pytest.approx(200.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Тесты кэша
# ---------------------------------------------------------------------------

class TestTractionCache:
    """Проверка кэширования тяговых расчётов."""

    def test_cache_miss_and_hit(self, test_train: TrainConfig):
        """Сначала промах (вычисление), потом попадание (тот же объект)."""
        cache = TractionCache()
        s1 = RouteSection("S1", 0.0, 200.0, 0.0)
        
        # Miss
        res1 = cache.get_or_compute(test_train, [s1], test_train.consist_id, "R1")
        assert len(cache) == 1
        
        # Hit — должен быть тот же самый объект памяти (is)
        res2 = cache.get_or_compute(test_train, [s1], test_train.consist_id, "R1")
        assert res1 is res2
        assert len(cache) == 1
        
    def test_invalidate_and_clear(self, test_train: TrainConfig):
        """Инвалидация по ключу и полная очистка кэша."""
        cache = TractionCache()
        s1 = RouteSection("S1", 0.0, 200.0, 0.0)
        cache.get_or_compute(test_train, [s1], "T1", "R1")
        cache.get_or_compute(test_train, [s1], "T1", "R2")
        
        # Сброс только R1 для T1
        cache.invalidate("T1", "R1")
        assert len(cache) == 1
        
        # Полная очистка
        cache.clear()
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Прочие функции
# ---------------------------------------------------------------------------

class TestHeadToTailProfile:
    """Точечная проверка функции head_to_tail_profile."""

    def test_head_to_tail_calculation(self):
        """Расчёт дельты времени с учетом хвоста по интерполяции профиля."""
        # Искусственный простой профиль: поезд едет с ускорением/замедлением
        s_points = np.array([0.0, 10.0, 20.0, 30.0])
        t_profile = np.array([0.0,  1.0,  5.0,  10.0])
        
        mock_result = PhysicsResult(
            consist_id="C",
            route_id="R",
            t_total_s=10.0,
            v_profile=np.zeros_like(s_points),
            t_profile=t_profile,
            s_points=s_points,
            head_to_tail_s=np.zeros_like(s_points)
        )
        
        # Длина поезда = 10 м. 
        # Значит хвост в s=10 был когда голова была в s=20.
        ht = head_to_tail_profile(mock_result, train_length_m=10.0)
        
        # s_tail для точек = [-10, 0, 10, 20]
        # t(0) = 0.0
        # t_tail(s=-10) = 0.0 (интерполяция слева) -> delta = 0.0 - 0.0
        # t_tail(s=0) = 0.0 -> delta = t(10) - 0.0 = 1.0 - 0.0
        # t_tail(s=10) = t(1) = 1.0 -> delta = 5.0 - 1.0 = 4.0
        # t_tail(s=20) = t(2) = 5.0 -> delta = 10.0 - 5.0 = 5.0
        
        np.testing.assert_array_almost_equal(ht, [
             0.0 - 0.0,  # голова в 0, хвост ещё не вошёл
             1.0 - 0.0,  # голова в 10, хвост в 0
             5.0 - 1.0,  # голова в 20, хвост в 10
            10.0 - 5.0   # голова в 30, хвост в 20
        ])
