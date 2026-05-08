"""Тесты для src/traction/loader.py — загрузка локомотивов и составов."""
from __future__ import annotations

import pytest
import numpy as np

from src.models import CurrentType
from src.traction.loader import (
    ConfigError,
    LocomotiveConfigError,
    TrainConfigError,
    load_locomotive,
    load_train,
)


MINIMAL_LOCO_YAML = """\
loco_id: 2ES5K
name: 2ЭС5К
current_type: AC
voltage: 25000
mass_t: 192.0
loco_length_m: 35.0
num_axes: 8
v_max: 110.0
traction_curve:
  - {v: 0, Fk: 600}
  - {v: 50, Fk: 400}
  - {v: 120, Fk: 200}
adhesion_curve:
  - {v: 0, Fk: 550}
  - {v: 50, Fk: 350}
  - {v: 120, Fk: 150}
wo_coeffs:
  a: 6.4
  b: 0.089
  c: 0.0022
bt_coeffs:
  K_P: 100.0
  phi_0: 10.0
  phi_1: 20.0
"""

MINIMAL_TRAIN_YAML = """\
consist_id: C1
num_wagons: 50
wagon_mass_t: 80.0
wagon_length_m: 14.5
q0: 20.0
wagon_type: 6
"""


class TestLoadLocomotiveSuccess:
    """Успешная загрузка локомотива."""

    def test_minimal_loco(self, write_yaml):
        path = write_yaml(MINIMAL_LOCO_YAML)
        loco = load_locomotive(path)

        assert loco.loco_id == "2ES5K"
        assert loco.name == "2ЭС5К"
        assert loco.current_type is CurrentType.AC
        assert loco.voltage == 25000
        assert loco.mass_t == 192.0
        assert loco.v_max == 110.0

        # Таблицы должны быть из 3 точек
        assert len(loco.v_table) == 3
        np.testing.assert_array_almost_equal(loco.v_table, [0.0, 50.0, 120.0])

        # fk_table = min(traction_curve, adhesion_curve)
        np.testing.assert_array_almost_equal(loco.fk_table, [550.0, 350.0, 150.0])

        assert len(loco.wox_table) == 3
        assert len(loco.bt_table) == 3

    def test_loco_only_traction_curve(self, write_yaml):
        """Если adhesion_curve нет, YAML считается невалидным."""
        yaml_text = """\
        loco_id: L1
        name: Тест
        current_type: DC
        voltage: 3000
        mass_t: 100.0
        loco_length_m: 20.0
        num_axes: 4
        v_max: 100.0
        traction_curve:
          - {v: 0, Fk: 300}
          - {v: 100, Fk: 100}
        wo_coeffs: {a: 2, b: 0.1, c: 0.01}
        bt_coeffs: {K_P: 10, phi_0: 1, phi_1: 1}
        """
        with pytest.raises(LocomotiveConfigError, match="'adhesion_curve' отсутствует"):
            load_locomotive(write_yaml(yaml_text))

    def test_loco_bt_curve_instead_of_coeffs(self, write_yaml):
        """Успешная загрузка, если вместо bt_coeffs задана bt_curve."""
        yaml_text = """\
        loco_id: L1
        name: Тест
        current_type: DC
        voltage: 3000
        mass_t: 100.0
        loco_length_m: 20.0
        num_axes: 4
        v_max: 100.0
        traction_curve:
          - {v: 0, Fk: 300}
          - {v: 100, Fk: 100}
        adhesion_curve:
          - {v: 0, Fk: 250}
          - {v: 100, Fk: 80}
        wo_coeffs: {a: 2, b: 0.1, c: 0.01}
        bt_curve:
          - {v: 0, bt: 10}
          - {v: 100, bt: 5}
        """
        loco = load_locomotive(write_yaml(yaml_text))
        np.testing.assert_array_almost_equal(loco.bt_table, [10.0, 5.0])


class TestLoadLocomotiveErrors:
    """Ошибки парсинга локомотива."""

    def test_missing_traction_curve(self, write_yaml):
        yaml_text = """\
        loco_id: 2ES5K
        name: 2ЭС5К
        current_type: AC
        voltage: 25000
        mass_t: 192.0
        loco_length_m: 35.0
        num_axes: 8
        v_max: 110.0
        wo_coeffs: {a: 0, b: 0, c: 0}
        bt_coeffs: {K_P: 1, phi_0: 1, phi_1: 1}
        """
        with pytest.raises(LocomotiveConfigError, match="'traction_curve' отсутствует"):
            load_locomotive(write_yaml(yaml_text))

    def test_missing_both_bt_configs(self, write_yaml):
        yaml_text = """\
        loco_id: L1
        name: 2ЭС5К
        current_type: AC
        voltage: 25000
        mass_t: 192.0
        loco_length_m: 35.0
        num_axes: 8
        v_max: 110.0
        traction_curve: [{v: 0, Fk: 100}, {v: 100, Fk: 50}]
        adhesion_curve: [{v: 0, Fk: 90}, {v: 100, Fk: 40}]
        wo_coeffs: {a: 0, b: 0, c: 0}
        """
        with pytest.raises(LocomotiveConfigError, match="'bt_curve' или 'bt_coeffs'"):
            load_locomotive(write_yaml(yaml_text))

    def test_adhesion_not_intersecting(self, write_yaml):
        yaml_text = """\
        loco_id: L1
        name: Тест
        current_type: DC
        voltage: 3000
        mass_t: 100.0
        loco_length_m: 20.0
        num_axes: 4
        v_max: 100.0
        traction_curve:
          - {v: 0, Fk: 300}
          - {v: 50, Fk: 100}
        adhesion_curve:
          - {v: 60, Fk: 300}
          - {v: 100, Fk: 100}
        wo_coeffs: {a: 2, b: 0.1, c: 0.01}
        bt_coeffs: {K_P: 10, phi_0: 1, phi_1: 1}
        """
        with pytest.raises(LocomotiveConfigError, match="не пересекаются"):
            load_locomotive(write_yaml(yaml_text))

    def test_bt_phi1_zero_or_negative(self, write_yaml):
        for bad_phi1 in [0.0, -5.0]:
            yaml_text = f"""\
            loco_id: L1
            name: Тест
            current_type: DC
            voltage: 3000
            mass_t: 100.0
            loco_length_m: 20.0
            num_axes: 4
            v_max: 100.0
            traction_curve:
              - {{v: 0, Fk: 300}}
              - {{v: 100, Fk: 100}}
            adhesion_curve:
              - {{v: 0, Fk: 250}}
              - {{v: 100, Fk: 80}}
            wo_coeffs: {{a: 2, b: 0.1, c: 0.01}}
            bt_coeffs: {{K_P: 10, phi_0: 1, phi_1: {bad_phi1}}}
            """
            with pytest.raises(LocomotiveConfigError, match="phi_1 должен быть > 0"):
                load_locomotive(write_yaml(yaml_text))

    def test_negative_f_k(self, write_yaml):
        yaml_text = """\
        loco_id: L1
        name: Тест
        current_type: DC
        voltage: 3000
        mass_t: 100.0
        loco_length_m: 20.0
        num_axes: 4
        v_max: 100.0
        traction_curve:
          - {v: 0, Fk: 300}
          - {v: 100, Fk: -100}
        adhesion_curve:
          - {v: 0, Fk: 250}
          - {v: 100, Fk: 80}
        wo_coeffs: {a: 2, b: 0.1, c: 0.01}
        bt_coeffs: {K_P: 10, phi_0: 1, phi_1: 1}
        """
        with pytest.raises(LocomotiveConfigError, match="содержит отрицательные значения"):
            load_locomotive(write_yaml(yaml_text))

    def test_missing_required_fields(self, write_yaml):
        """Если loco_id отсутствует — KeyError -> LocomotiveConfigError."""
        yaml_text = """\
        name: 2ЭС5К
        current_type: AC
        voltage: 25000
        mass_t: 192.0
        loco_length_m: 35.0
        num_axes: 8
        v_max: 110.0
        traction_curve: [{v: 0, Fk: 100}, {v: 100, Fk: 50}]
        adhesion_curve: [{v: 0, Fk: 90}, {v: 100, Fk: 40}]
        wo_coeffs: {a: 0, b: 0, c: 0}
        bt_coeffs: {K_P: 1, phi_0: 1, phi_1: 1}
        """
        with pytest.raises(LocomotiveConfigError, match="'loco_id'"):
            load_locomotive(write_yaml(yaml_text))


class TestLoadTrain:
    """Успешная загрузка и ошибки конфигурации состава."""

    @pytest.fixture
    def loco(self, write_yaml) -> LocomotiveConfig:
        return load_locomotive(write_yaml(MINIMAL_LOCO_YAML))

    def test_load_train_success_from_path(self, write_yaml, loco):
        path = write_yaml(MINIMAL_TRAIN_YAML)
        train = load_train(loco, path)

        assert train.consist_id == "C1"
        assert train.num_wagons == 50
        assert train.train_mass_t == loco.mass_t + 50 * 80.0
        assert train.train_length_m == loco.loco_length_m + 50 * 14.5

    def test_load_train_success_from_dict(self, loco):
        raw = {
            "consist_id": "C2",
            "num_wagons": 10,
            "wagon_mass_t": 50.0,
            "wagon_length_m": 14.0,
            "q0": 15.0,
            "wagon_type": 4,
        }
        train = load_train(loco, raw)
        assert train.consist_id == "C2"
        assert train.num_wagons == 10
        np.testing.assert_array_almost_equal(train.bt_wagons_table, np.zeros_like(loco.v_table))

    def test_load_train_wagon_bt_coeffs(self, loco):
        raw = {
            "consist_id": "C3",
            "num_wagons": 10,
            "wagon_mass_t": 50.0,
            "wagon_length_m": 14.0,
            "q0": 15.0,
            "wagon_type": 4,
            "bt_coeffs": {"KP": 2.0, "phi0": 1.0, "phi1": 0.22},
        }
        train = load_train(loco, raw)
        expected = 2.0 * (1.0 + 0.22 * loco.v_table / 2.2)
        np.testing.assert_array_almost_equal(train.bt_wagons_table, expected)

    def test_override_consist_id(self, loco):
        raw = {
            "consist_id": "C2",
            "num_wagons": 10,
            "wagon_mass_t": 50.0,
            "wagon_length_m": 14.0,
            "q0": 15.0,
            "wagon_type": 4,
        }
        train = load_train(loco, raw, consist_id="NEW_C")
        assert train.consist_id == "NEW_C"

    def test_train_invalid_values(self, write_yaml, loco):
        """Невалидные значения ( <= 0) вызывают ValueError в dataclass, который оборачивается в TrainConfigError."""
        yaml_text = """\
        consist_id: C1
        num_wagons: 0
        wagon_mass_t: 80.0
        wagon_length_m: 14.5
        q0: 20.0
        wagon_type: 6
        """
        with pytest.raises(TrainConfigError, match="num_wagons должен быть > 0"):
            load_train(loco, write_yaml(yaml_text))

    def test_train_invalid_wagon_type(self, loco):
        raw = {
            "consist_id": "C_BAD",
            "num_wagons": 10,
            "wagon_mass_t": 50.0,
            "wagon_length_m": 14.0,
            "q0": 15.0,
            "wagon_type": 5,
        }
        with pytest.raises(TrainConfigError, match="wagon_type должен быть 4, 6 или 8"):
            load_train(loco, raw)
