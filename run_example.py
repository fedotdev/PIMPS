from __future__ import annotations

"""Пример запуска симуляции PIMPS: сценарии АБ и ВС (Методики А и Б)."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.interlocking.engine import InterlockingEngine
from src.interlocking.loader import load_station
from src.models import ScenarioEntry, RouteSection, ControlMode, SimResult, StationEvent, VCMethodology
from src.renderers.metrics import (
    calculate_summary_metrics,
    export_sim_results,
    export_events_log,
    export_delays_table,
    export_physics_data,
    export_scenario_comparison,
)
from src.renderers.excel_reporter import export_xlsx
from src.renderers.plots import (
    plot_physics_profile,
    plot_station_occupancy,
    plot_throughput_comparison,
    plot_methodology_comparison,
)
from src.simulation import SimulationEngine
from src.traction.dynamics import TractionCache
from src.traction.loader import load_locomotive, load_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pimps_demo")


# ---------------------------------------------------------------------------
# Вспомогательные функции для генерации секций
# ---------------------------------------------------------------------------

TRACK_LENGTH_M  = 850.0
THROAT_NGP_M    = 150.0
THROAT_CHG_M    = 150.0


def get_arr_sections(track_id: int) -> list[RouteSection]:
    """Секции маршрута прибытия: входная горловина → станционный путь.

    0 → THROAT_NGP_M                       : горловина НГП
    THROAT_NGP_M → THROAT_NGP_M + TRACK_LENGTH_M : тело пути
    """
    return [
        RouteSection(
            section_id="STR_ENTER",
            s_start=0.0, s_end=THROAT_NGP_M,
            grade=0.0, radius=300.0, v_limit=40.0,
        ),
        RouteSection(
            section_id=f"TRK_P{track_id}",
            s_start=THROAT_NGP_M, s_end=THROAT_NGP_M + TRACK_LENGTH_M,
            grade=-3.5, radius=0.0, v_limit=40.0,
        ),
    ]


def get_dep_sections(track_id: int) -> list[RouteSection]:
    """Секции маршрута отправления: тело пути → выходная горловина ЧГП.

    0 → TRACK_LENGTH_M                      : тело пути
    TRACK_LENGTH_M → TRACK_LENGTH_M + THROAT_CHG_M : горловина ЧГП
    """
    return [
        RouteSection(
            section_id=f"TRK_P{track_id}",
            s_start=0.0, s_end=TRACK_LENGTH_M,
            grade=3.5, radius=0.0, v_limit=40.0,
        ),
        RouteSection(
            section_id="STR_EXIT",
            s_start=TRACK_LENGTH_M, s_end=TRACK_LENGTH_M + THROAT_CHG_M,
            grade=0.0, radius=300.0, v_limit=40.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Генерация сценариев
# ---------------------------------------------------------------------------

PLATOON_DEFS = [
    ("PLT-1", 2, 3),
    ("PLT-2", 3, 3),
    ("PLT-3", 5, 2),
]

ARRIVAL_ROUTE_BY_TRACK = {
    1: "route_N_1P",
    2: "route_N_2P",
    3: "route_N_3P",
    5: "route_N_5P",
}

DEPARTURE_ROUTE_BY_TRACK = {
    1: "route_1P_B",
    2: "route_2P_B",
    3: "route_3P_B",
    5: "route_5P_B",
}

INTER_PLATOON_GAP_S = 300.0


def build_vc_entries(
    train,
    intra_interval_s: float,
    platoon_mode: bool = True,
) -> list[ScenarioEntry]:
    """Формирует расписание: 8 поездов, 3 пакета (3+3+2).

    platoon_mode=False — все поезда считаются одиночными (режим АБ).
    """
    entries: list[ScenarioEntry] = []
    train_idx = 0
    platoon_start = 0.0

    for platoon_id, track_id, count in PLATOON_DEFS:
        for j in range(count):
            train_idx += 1
            entries.append(ScenarioEntry(
                train_id=f"Freight-{train_idx:03d}",
                t_arrive_s=platoon_start + j * intra_interval_s,
                route_id=ARRIVAL_ROUTE_BY_TRACK[track_id],
                train=train,
                sections=get_arr_sections(track_id),
                v0_kmh=40.0,
                dwell_s=120.0,
                platoon_id=platoon_id if platoon_mode else None,
                departure_route_id=DEPARTURE_ROUTE_BY_TRACK[track_id],
                departure_sections=get_dep_sections(track_id),
            ))
        platoon_start += INTER_PLATOON_GAP_S

    return entries


def build_packet_split_entries(train) -> list[ScenarioEntry]:
    """СЦЕНАРИЙ 3: ВС с разделением пакета на подходе (3 поезда, интервал 180 с)."""
    return [
        ScenarioEntry(
            train_id="Freight-301", t_arrive_s=0,
            route_id="route_N_2P", train=train, sections=get_arr_sections(2),
            v0_kmh=40.0, dwell_s=120.0, platoon_id="PLT-SPLIT",
            departure_route_id="route_2P_B", departure_sections=get_dep_sections(2),
        ),
        ScenarioEntry(
            train_id="Freight-302", t_arrive_s=180,
            route_id="route_N_3P", train=train, sections=get_arr_sections(3),
            v0_kmh=40.0, dwell_s=120.0, platoon_id="PLT-SPLIT",
            departure_route_id="route_3P_B", departure_sections=get_dep_sections(3),
        ),
        ScenarioEntry(
            train_id="Freight-303", t_arrive_s=360,
            route_id="route_N_2P", train=train, sections=get_arr_sections(2),
            v0_kmh=40.0, dwell_s=120.0, platoon_id="PLT-SPLIT",
            departure_route_id="route_2P_B", departure_sections=get_dep_sections(2),
        ),
    ]


def build_recovery_entries(
    train,
    interval_s: float,
    platoon_mode: bool,
) -> list[ScenarioEntry]:
    """СЦЕНАРИЙ 4: восстановление графика после сбоя.

    6 поездов. Поезд №3 прибывает с задержкой 480 с.
    """
    pid = "PLT-REC" if platoon_mode else None
    delays = {3: 480.0}
    return [
        ScenarioEntry(
            train_id=f"Freight-40{i}",
            t_arrive_s=(i - 1) * interval_s,
            route_id="route_N_2P", train=train, sections=get_arr_sections(2),
            v0_kmh=40.0, dwell_s=120.0, platoon_id=pid,
            departure_route_id="route_2P_B", departure_sections=get_dep_sections(2),
            delay_s=delays.get(i, 0.0),
        )
        for i in range(1, 7)
    ]


def run_scenario(
    station_config,
    train,
    traction_cache: TractionCache,
    entries: list[ScenarioEntry],
    scenario_name: str,
    control_mode: ControlMode,
    vc_methodology: str,
    vc_min_headway_s: float,
    output_dir: Path,
    planned_interval_s: float = 0.0,
) -> tuple[list[SimResult], list[StationEvent], dict[str, float]]:
    """Запускает один сценарий и экспортирует все артефакты."""
    logger.info("═" * 60)
    logger.info("Сценарий: %s  |  Режим: %s  |  Методика: %s  |  Поездов: %d",
                scenario_name, control_mode.value, vc_methodology, len(entries))
    logger.info("═" * 60)

    interlocking = InterlockingEngine(
        station_config,
        vc_methodology=vc_methodology,
        control_mode=control_mode.value,
    )
    sim = SimulationEngine(
        interlocking=interlocking,
        traction_cache=traction_cache,
        scenario_name=scenario_name,
        control_mode=control_mode.value,
        vc_methodology=vc_methodology,
        vc_min_headway_s=vc_min_headway_s,
        route_setup_time_s=10.0,
    )
    sim.load_scenario(entries)
    results = sim.run()
    events = sim.events

    profiles_dir = output_dir / "profiles"
    data_dir     = output_dir / "data"
    station_dir  = output_dir / "station"
    for d in (profiles_dir, data_dir, station_dir):
        d.mkdir(parents=True, exist_ok=True)

    export_sim_results(results, data_dir / f"sim_results_{scenario_name}.csv")
    export_events_log(events,   data_dir / f"events_log_{scenario_name}.csv")
    export_delays_table(results, data_dir / f"delays_{scenario_name}.csv")

    metrics = calculate_summary_metrics(
        results, events=events,
        vc_min_headway_s=vc_min_headway_s,
        planned_interval_s=planned_interval_s,
        vc_gap_threshold_s=2.0 * vc_min_headway_s,
    )

    # График занятости маршрутов
    plot_station_occupancy(
        events, scenario_name,
        station_dir / f"occupancy_{scenario_name}.png",
    )

    # Тяговые профили
    seen_profiles: set[str] = set()
    for r in results:
        key = f"{r.consist_id}:{r.route_id}"
        if key in seen_profiles:
            continue
        seen_profiles.add(key)
        physics = next(
            (p for p in traction_cache._store.values()
             if p.consist_id == r.consist_id and p.route_id == r.route_id),
            None,
        )
        if physics:
            safe_id = r.train_id.replace(" ", "_").replace("(", "").replace(")", "")
            plot_physics_profile(
                physics, r.train_id,
                profiles_dir / f"profile_{scenario_name}_{safe_id}_{r.route_id}.png",
                v_limit_kmh=40.0,
                section_boundaries_m=[THROAT_NGP_M],
            )
            export_physics_data(
                physics,
                data_dir / f"physics_{scenario_name}_{safe_id}_{r.route_id}.csv",
            )

    return results, events, metrics


# ---------------------------------------------------------------------------
# Итоговый вывод в консоли
# ---------------------------------------------------------------------------

_SUMMARY_ROWS = [
    ("throughput_trains_per_hour", "Пропускная способность, п/ч",       ".2f"),
    ("headway_avg_s",              "Ср. интервал отправл., с",           ".1f"),
    ("mean_wait_time_s",           "Ср. ожидание маршрута, с",          ".1f"),
    ("delay_depart_avg_s",         "Ср. задержка отправл., с",          ".1f"),
    ("throat_utilization",         "Загрузка маршрутов горловины",       ".1%"),
    ("packet_integrity_ratio",     "Сохранность пакетов",                ".0%"),
]


def _print_summary_table(
    metrics_ab:    dict[str, float],
    metrics_vc_a:  dict[str, float],
    metrics_vc_b:  dict[str, float],
    all_scenarios: dict[str, dict[str, float]],
) -> None:
    """Печатает оформленную итоговую таблицу для консольного вывода."""

    def _fv(v: float, fmt: str) -> str:
        if v != v:
            return "  N/A"
        return f"{v:{fmt}}"

    W = 82
    print()
    print("┌" + "─" * W + "┐")
    print("│" + " PIMPS — ИТОГОВЫЕ РЕЗУЛЬТАТЫ СИМУЛЯЦИОННОГО МОДЕЛИРОВАНИЯ".center(W) + "│")
    print("│" + " Станция: МИИТовская | Локомотив: 2ЭС-5к | Состав: 80 вагонов".center(W) + "│")
    print("╞" + "═" * W + "╡")
    print("│" + f" {'':38} {'  АБ':>10} {'  ВС-А':>10} {'  ВС-Б':>10} {'  Дельта':>8} ".ljust(W) + "│")
    print("│" + f" {'':38} {'автоблок.':>10} {'мет. А':>10} {'мет. Б':>10} {'(Б−АБ)':>8} ".ljust(W) + "│")
    print("╞" + "─" * W + "╡")

    for key, label, fmt in _SUMMARY_ROWS:
        ab  = metrics_ab.get(key, float("nan"))
        va  = metrics_vc_a.get(key, float("nan"))
        vb  = metrics_vc_b.get(key, float("nan"))
        delta = vb - ab if (vb == vb and ab == ab) else float("nan")
        arrow = ("↑" if delta > 0 else "↓") if delta == delta and abs(delta) > 1e-9 else " "
        print("│" + f"  {label:<38} {_fv(ab, fmt):>10} {_fv(va, fmt):>10} {_fv(vb, fmt):>10} {arrow}{_fv(delta, fmt):>7} ".ljust(W) + "│")

    print("╞" + "═" * W + "╡")
    print("│" + " Дополнительные сценарии:".ljust(W) + "│")
    print("│" + " (↓ — улучшение относительно АБ-базы)".ljust(W) + "│")

    extra_keys = [
        ("VC-Packet-Split", "Сценарий: разделение пакета ВС",
          "packet_split_delay_s",  "задержка разделения",  ".1f"),
        ("AB-Recovery",     "Сценарий: восстановление АБ",
          "recovery_time_s",       "время восстановления",    ".1f"),
        ("VC-Recovery",     "Сценарий: восстановление ВС",
          "recovery_time_s",       "время восстановления",    ".1f"),
    ]
    for sc_key, sc_label, metric_key, metric_label, fmt in extra_keys:
        m = all_scenarios.get(sc_key, {})
        val = m.get(metric_key, float("nan"))
        print("│" + f"  {sc_label:<42} {metric_label}: {_fv(val, fmt):>8} с".ljust(W) + "│")

    print("╞" + "═" * W + "╡")
    print("│" + " Примечание: загрузка горловины >100% — норма при пакетном движении".ljust(W) + "│")
    print("│" + " (одновременно задействованы несколько параллельных маршрутов).".ljust(W) + "│")
    print("└" + "─" * W + "┘")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VC_MIN_HEADWAY_S = 60.0
INTRA_INTERVAL_S = 40.0
AB_INTERVAL_S    = 360.0


def main() -> None:
    logger.info("Инициализация данных...")

    station_config = load_station(Path("stations/miitovskaya_station.yaml"))
    locomotive     = load_locomotive(Path("config/2ES5k.yaml"))
    train          = load_train(locomotive, Path("config/demo_train.yaml"))
    traction_cache = TractionCache()

    output_dir = Path("output")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    entries_ab = build_vc_entries(train, AB_INTERVAL_S,    platoon_mode=False)
    entries_vc = build_vc_entries(train, INTRA_INTERVAL_S, platoon_mode=True)

    # Сценарий 1 — АБ
    _, _, metrics_ab = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_ab, scenario_name="Demo-AB",
        control_mode=ControlMode.AB, vc_methodology="A",
        vc_min_headway_s=AB_INTERVAL_S, output_dir=output_dir,
        planned_interval_s=AB_INTERVAL_S,
    )

    # Сценарий 2 — ВС Методика А
    _, _, metrics_vc_a = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_vc, scenario_name="Demo-VC-A",
        control_mode=ControlMode.VC, vc_methodology="A",
        vc_min_headway_s=VC_MIN_HEADWAY_S, output_dir=output_dir,
        planned_interval_s=INTRA_INTERVAL_S,
    )

    # Сценарий 3 — ВС Методика Б
    _, _, metrics_vc_b = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_vc, scenario_name="Demo-VC-B",
        control_mode=ControlMode.VC, vc_methodology="B",
        vc_min_headway_s=VC_MIN_HEADWAY_S, output_dir=output_dir,
        planned_interval_s=INTRA_INTERVAL_S,
    )

    # Сценарий — Разделение пакета
    entries_split = build_packet_split_entries(train)
    _, _, metrics_split = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_split, scenario_name="VC-Packet-Split",
        control_mode=ControlMode.VC, vc_methodology="A",
        vc_min_headway_s=180.0, output_dir=output_dir,
        planned_interval_s=180.0,
    )

    # Сценарий 4 — Восстановление графика
    entries_rec_ab = build_recovery_entries(train, interval_s=AB_INTERVAL_S, platoon_mode=False)
    entries_rec_vc = build_recovery_entries(train, interval_s=180.0, platoon_mode=True)

    _, _, metrics_rec_ab = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_rec_ab, scenario_name="AB-Recovery",
        control_mode=ControlMode.AB, vc_methodology="A",
        vc_min_headway_s=AB_INTERVAL_S, output_dir=output_dir,
        planned_interval_s=AB_INTERVAL_S,
    )
    _, _, metrics_rec_vc = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_rec_vc, scenario_name="VC-Recovery",
        control_mode=ControlMode.VC, vc_methodology="A",
        vc_min_headway_s=180.0, output_dir=output_dir,
        planned_interval_s=180.0,
    )

    # Сборные метрики всех сценариев (без дублей — только именованные сценарии)
    scenario_metrics: dict[str, dict[str, float]] = {
        "Demo-AB":         metrics_ab,
        "Demo-VC-A":       metrics_vc_a,
        "Demo-VC-B":       metrics_vc_b,
        "VC-Packet-Split": metrics_split,
        "AB-Recovery":     metrics_rec_ab,
        "VC-Recovery":     metrics_rec_vc,
    }

    # Графики и экспорт
    export_scenario_comparison(scenario_metrics, data_dir / "throughput_comparison.csv")
    plot_throughput_comparison(scenario_metrics, output_dir / "throughput_benchmark.png")
    if metrics_vc_a and metrics_vc_b:
        plot_methodology_comparison(
            metrics_a=metrics_vc_a,
            metrics_b=metrics_vc_b,
            metrics_ab=metrics_ab,
            out_path=output_dir / "methodology_comparison_chart.png",
        )
    export_xlsx(scenario_metrics, output_dir)

    # Манифест запуска
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "station_config": "stations/miitovskaya_station.yaml",
        "locomotive_config": "config/2ES5k.yaml",
        "train_config": "config/demo_train.yaml",
        "parameters": {
            "vc_min_headway_s":   VC_MIN_HEADWAY_S,
            "intra_interval_s":   INTRA_INTERVAL_S,
            "ab_interval_s":      AB_INTERVAL_S,
            "inter_platoon_gap_s": INTER_PLATOON_GAP_S,
        },
        "scenarios": list(scenario_metrics.keys()),
        "outputs": {
            "data":     "output/data",
            "station":  "output/station",
            "profiles": "output/profiles",
            "root":     "output",
        },
    }
    with open(data_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Оформленная итоговая таблица
    _print_summary_table(metrics_ab, metrics_vc_a, metrics_vc_b, scenario_metrics)

    logger.info("Артефакты записаны в output/:")
    logger.info("  • Данные CSV:      output/data/")
    logger.info("  • Диаграммы Ганта:  output/station/")
    logger.info("  • Тяговые профили: output/profiles/")
    logger.info("  • Сводные графики: output/")
    logger.info("  • Excel-отчёт:    output/simulation_results.xlsx")


if __name__ == "__main__":
    main()
