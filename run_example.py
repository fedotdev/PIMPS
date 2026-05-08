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
    generate_markdown_report,
    generate_methodology_comparison_report,
)
from src.renderers.plots import (
    plot_physics_profile, 
    plot_station_occupancy, 
    plot_throughput_comparison,
    plot_methodology_comparison,
)
from src.simulation import SimulationEngine
from src.traction.dynamics import TractionCache
from src.traction.loader import load_locomotive, load_train

# Настраиваем логирование, чтобы видеть процесс
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pimps_demo")


# ---------------------------------------------------------------------------
# Вспомогательные функции для генерации секций
# ---------------------------------------------------------------------------

def get_arr_sections(track_id: int) -> list[RouteSection]:
    """Секции маршрута прибытия (входная горловина -> станционный путь)."""
    return [
        RouteSection(section_id="STR_ENTER", s_start=0.0, s_end=150.0, grade=0.0, radius=300.0, v_limit=40.0),
        RouteSection(section_id=f"TRK_P{track_id}", s_start=150.0, s_end=1400.0, grade=-3.5, radius=0.0, v_limit=40.0),
    ]

def get_dep_sections() -> list[RouteSection]:
    """Секции маршрута отправления (станционный путь -> выходная горловина)."""
    return [
        RouteSection(section_id="STR_EXIT", s_start=1400.0, s_end=1550.0, grade=0.0, radius=300.0, v_limit=40.0),
    ]


# ---------------------------------------------------------------------------
# Генерация сценариев
# ---------------------------------------------------------------------------

# Описание пакетов: (platoon_id, track_id, кол-во поездов)
PLATOON_DEFS = [
    ("PLT-1", 2, 3),  # Пакет 1: 3 поезда по пути 2
    ("PLT-2", 3, 3),  # Пакет 2: 3 поезда по пути 3
    ("PLT-3", 5, 2),  # Пакет 3: 2 поезда по пути 5
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

# Смещение (в секундах) между началом пакетов.
# Должно быть достаточным, чтобы горловина (STR_ENTER, SW1) успевала
# освободиться от предыдущего пакета.
INTER_PLATOON_GAP_S = 300.0


def build_vc_entries(
    train,
    intra_interval_s: float,
    platoon_mode: bool = True,
) -> list[ScenarioEntry]:
    """Формирует расписание: 8 поездов, 3 пакета (3+3+2).

    Каждый пакет следует по своему станционному пути (P2, P3, P4).
    Внутри пакета поезда идут с интервалом intra_interval_s.
    Между пакетами — фиксированный зазор INTER_PLATOON_GAP_S.

    При platoon_mode=False все поезда считаются одиночными (АБ-режим).
    """
    entries: list[ScenarioEntry] = []
    train_idx = 0
    platoon_start = 0.0

    for platoon_id, track_id, count in PLATOON_DEFS:
        arrival_route_id = ARRIVAL_ROUTE_BY_TRACK[track_id]
        departure_route_id = DEPARTURE_ROUTE_BY_TRACK[track_id]
        for j in range(count):
            train_idx += 1
            pid = platoon_id if platoon_mode else None
            t_arr = platoon_start + j * intra_interval_s

            entries.append(ScenarioEntry(
                train_id=f"Freight-{train_idx:03d}",
                t_arrive_s=t_arr,
                route_id=arrival_route_id,
                train=train,
                sections=get_arr_sections(track_id),
                v0_kmh=40.0,
                dwell_s=120.0,
                platoon_id=pid,
                departure_route_id=departure_route_id,
                departure_sections=get_dep_sections(),
            ))
        platoon_start += INTER_PLATOON_GAP_S

    return entries


def build_packet_split_entries(train) -> list[ScenarioEntry]:
    """СЦЕНАРИЙ 3: ВС с разделением пакета на подходе
    Пакет из 3 поездов (PLT-SPLIT) интервал 180с. 
    Поезд #1 -> P1, #2 -> P3 (чтобы разойтись), #3 -> P1 (снова на старый путь). 
    Здесь используется маршрут прибытия на путь и отправления с пути
    в терминах текущей ЭЦ.
    Длительность стоянки = 0 (пропуск).
    """
    return [
        ScenarioEntry(
            train_id="Freight-301", t_arrive_s=0,
            route_id="route_N_2P", train=train, sections=get_arr_sections(2),
            v0_kmh=40.0, dwell_s=120.0, platoon_id="PLT-SPLIT",
            departure_route_id="route_2P_B", departure_sections=get_dep_sections()
        ),
        ScenarioEntry(
            train_id="Freight-302", t_arrive_s=180,
            route_id="route_N_3P", train=train, sections=get_arr_sections(3),
            v0_kmh=40.0, dwell_s=120.0, platoon_id="PLT-SPLIT",
            departure_route_id="route_3P_B", departure_sections=get_dep_sections()
        ),
        ScenarioEntry(
            train_id="Freight-303", t_arrive_s=360,
            route_id="route_N_2P", train=train, sections=get_arr_sections(2),
            v0_kmh=40.0, dwell_s=120.0, platoon_id="PLT-SPLIT",
            departure_route_id="route_2P_B", departure_sections=get_dep_sections()
        )
    ]

def build_recovery_entries(train, interval_s: float, platoon_mode: bool) -> list[ScenarioEntry]:
    """СЦЕНАРИЙ 4: Восстановление графика после сбоя
    6 поездов. Поезд 3 отправлен с задержкой 480 с.
    """
    entries = []
    pid = "PLT-REC" if platoon_mode else None
    
    delays = {3: 480.0} # задержка для 3-го поезда (Freight-403)
    
    for i in range(1, 7):
        t_arr = (i - 1) * interval_s
        delay = delays.get(i, 0.0)
        entries.append(ScenarioEntry(
            train_id=f"Freight-40{i}", t_arrive_s=t_arr,
            route_id="route_N_2P", train=train, sections=get_arr_sections(2),
            v0_kmh=40.0, dwell_s=120.0, platoon_id=pid,
            departure_route_id="route_2P_B", departure_sections=get_dep_sections(),
            delay_s=delay
        ))
    return entries


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

    logger.info("=" * 60)
    logger.info("Запуск сценария: %s (режим: %s, методика: %s, поездов: %d)",
                scenario_name, control_mode.value, vc_methodology, len(entries))
    logger.info("=" * 60)

    # Пересоздаём движок ЭЦ для чистого состояния
    interlocking = InterlockingEngine(
        station_config, 
        vc_methodology=vc_methodology, 
        control_mode=control_mode.value
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

    # --- Экспорт артефактов сценария ---
    profiles_dir = output_dir / "profiles"
    data_dir = output_dir / "data"
    station_dir = output_dir / "station"
    summary_dir = output_dir / "summary"

    for d in [profiles_dir, data_dir, station_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # CSV
    export_sim_results(results, data_dir / f"sim_results_{scenario_name}.csv")
    export_events_log(events, station_dir / f"events_log_{scenario_name}.csv")
    export_delays_table(results, data_dir / f"delays_{scenario_name}.csv")

    # Метрики
    metrics = calculate_summary_metrics(
        results, events=events, vc_min_headway_s=vc_min_headway_s,
        planned_interval_s=planned_interval_s,
        vc_gap_threshold_s=2.0 * vc_min_headway_s,
    )
    logger.info("Метрики [%s]: %s", scenario_name, metrics)

    # Графики станции
    plot_station_occupancy(events, scenario_name, station_dir / f"occupancy_{scenario_name}.png")

    # Markdown-отчёт
    generate_markdown_report(results, metrics, summary_dir / f"report_{scenario_name}.md")

    # Профили тяги (по одному на уникальную комбинацию consist + route)
    seen_profiles: set[str] = set()
    for r in results:
        profile_key = f"{r.consist_id}:{r.route_id}"
        if profile_key in seen_profiles:
            continue
        seen_profiles.add(profile_key)
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
            )
            export_physics_data(
                physics,
                data_dir / f"physics_{scenario_name}_{safe_id}_{r.route_id}.csv",
            )

    return results, events, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VC_MIN_HEADWAY_S = 60.0   # минимальный интервал ВС (настраиваемый)
INTRA_INTERVAL_S = 40.0   # интервал внутри пакета (сокращён для ВС)
AB_INTERVAL_S    = 360.0  # интервал для АБ (6 мин)


def main():
    logger.info("Инициализация данных...")

    # 1. Загружаем конфигурацию станции
    station_config = load_station(Path("stations/miitovskaya_station.yaml"))

    # 2. Загружаем локомотив и состав
    locomotive = load_locomotive(Path("config/2ES5k.yaml"))
    train = load_train(locomotive, Path("config/demo_train.yaml"))

    # 3. Общий кэш тяговых расчётов (переиспользуется между сценариями)
    traction_cache = TractionCache()

    output_dir = Path("output")
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Генерация расписаний
    # =========================================================
    # Те же 8 поездов, 3 пакета — один набор используется для А и Б.
    entries_ab  = build_vc_entries(train, AB_INTERVAL_S,    platoon_mode=False)
    entries_vc  = build_vc_entries(train, INTRA_INTERVAL_S, platoon_mode=True)

    # =========================================================
    # Сценарий 1: АБ (автоблокировка) — классические интервалы
    # =========================================================
    _, _, metrics_ab = run_scenario(
        station_config=station_config,
        train=train,
        traction_cache=traction_cache,
        entries=entries_ab,
        scenario_name="Demo-AB",
        control_mode=ControlMode.AB,
        vc_methodology="A",
        vc_min_headway_s=AB_INTERVAL_S,
        output_dir=output_dir,
        planned_interval_s=AB_INTERVAL_S,
    )

    # =========================================================
    # Сценарий 2: ВС — Методика А (базовая ВС, без координации)
    # =========================================================
    _, _, metrics_vc_a = run_scenario(
        station_config=station_config,
        train=train,
        traction_cache=traction_cache,
        entries=entries_vc,
        scenario_name="Demo-VC-A",
        control_mode=ControlMode.VC,
        vc_methodology="A",
        vc_min_headway_s=VC_MIN_HEADWAY_S,
        output_dir=output_dir,
        planned_interval_s=INTRA_INTERVAL_S,
    )

    # =========================================================
    # Сценарий 3: ВС — Методика Б (координированная ВС)
    # =========================================================
    _, _, metrics_vc_b = run_scenario(
        station_config=station_config,
        train=train,
        traction_cache=traction_cache,
        entries=entries_vc,
        scenario_name="Demo-VC-B",
        control_mode=ControlMode.VC,
        vc_methodology="B",
        vc_min_headway_s=VC_MIN_HEADWAY_S,
        output_dir=output_dir,
        planned_interval_s=INTRA_INTERVAL_S,
    )

    # =========================================================
    # Сценарий 3: ВС - Разделение пакета на подходе (VC-Packet-Split)
    # =========================================================
    entries_split = build_packet_split_entries(train)
    _, _, metrics_split = run_scenario(
        station_config=station_config,
        train=train,
        traction_cache=traction_cache,
        entries=entries_split,
        scenario_name="VC-Packet-Split",
        control_mode=ControlMode.VC,
        vc_methodology="A", # Тут не работает B, т.к. маршруты разные
        vc_min_headway_s=180.0,
        output_dir=output_dir,
        planned_interval_s=180.0,
    )

    # =========================================================
    # Сценарий 4: Восстановление графика после сбоя
    # =========================================================
    entries_rec_ab = build_recovery_entries(train, interval_s=AB_INTERVAL_S, platoon_mode=False)
    entries_rec_vc = build_recovery_entries(train, interval_s=180.0, platoon_mode=True)

    _, _, metrics_rec_ab = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_rec_ab, scenario_name="AB-Recovery",
        control_mode=ControlMode.AB, vc_methodology="A", vc_min_headway_s=AB_INTERVAL_S,
        output_dir=output_dir,
        planned_interval_s=AB_INTERVAL_S,
    )

    _, _, metrics_rec_vc = run_scenario(
        station_config=station_config, train=train, traction_cache=traction_cache,
        entries=entries_rec_vc, scenario_name="VC-Recovery",
        control_mode=ControlMode.VC, vc_methodology="A", vc_min_headway_s=180.0,
        output_dir=output_dir,
        planned_interval_s=180.0,
    )

    # =========================================================
    # Сравнение сценариев
    # =========================================================
    scenario_metrics = {
        "АБ (Baseline)": metrics_ab,
        "ВС — Методика А": metrics_vc_a,
        "ВС — Методика Б": metrics_vc_b,
        "ВС — Packet Split": metrics_split,
        "АБ — Recovery": metrics_rec_ab,
        "ВС — Recovery": metrics_rec_vc,
    }

    export_scenario_comparison(scenario_metrics, summary_dir / "throughput_comparison.csv")
    plot_throughput_comparison(scenario_metrics, summary_dir / "throughput_benchmark.png")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "station_config": "stations/miitovskaya_station.yaml",
        "locomotive_config": "config/2ES5k.yaml",
        "train_config": "config/demo_train.yaml",
        "parameters": {
            "vc_min_headway_s": VC_MIN_HEADWAY_S,
            "intra_interval_s": INTRA_INTERVAL_S,
            "ab_interval_s": AB_INTERVAL_S,
            "inter_platoon_gap_s": INTER_PLATOON_GAP_S,
        },
        "scenarios": list(scenario_metrics.keys()),
        "outputs": {
            "data": "output/data",
            "station": "output/station",
            "profiles": "output/profiles",
            "summary": "output/summary",
        },
    }
    with open(summary_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # Расширенный отчёт Методика А vs Методика Б
    generate_methodology_comparison_report(
        metrics_a=metrics_vc_a, 
        metrics_b=metrics_vc_b, 
        out_path=summary_dir / "methodology_comparison_report.md"
    )
    if metrics_vc_a and metrics_vc_b:
        plot_methodology_comparison(
            metrics_a=metrics_vc_a,
            metrics_b=metrics_vc_b,
            metrics_ab=metrics_ab,
            out_path=summary_dir / "methodology_comparison_chart.png",
        )

    logger.info("=" * 60)
    logger.info("Готово! Проверьте папку 'output/'.")
    for label, m in scenario_metrics.items():
        integrity = m.get("packet_integrity_ratio", float("nan"))
        integrity_s = f"{integrity:.0%}" if integrity == integrity else "N/A"
        
        info_str = f"  {label}: integrity={integrity_s}, wait={m.get('mean_wait_time_s', 0):.0f} с, throughput={m.get('throughput_trains_per_hour', 0):.1f} п/ч"
        
        if m.get("max_queue_length", 0) > 0:
            info_str += f", queue={m.get('max_queue_length', 0):.0f}"
        if m.get("packet_split_delay_s", 0) > 0:
            info_str += f", split_delay={m.get('packet_split_delay_s', 0):.1f} с"
        if m.get("cascade_delay_s", 0) > 0:
            info_str += f", cascade={m.get('cascade_delay_s', 0):.1f} с"
        if m.get("recovery_time_s", 0) > 0:
            info_str += f", recovery={m.get('recovery_time_s', 0):.1f} с"
            
        logger.info(info_str)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
