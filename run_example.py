"""Пример запуска симуляции PIMPS: два сценария (АБ и ВС)."""
import logging
from pathlib import Path

from src.interlocking.engine import InterlockingEngine
from src.interlocking.loader import load_station
from src.models import ScenarioEntry, RouteSection, ControlMode, SimResult, StationEvent
from src.renderers.metrics import (
    calculate_summary_metrics,
    export_sim_results,
    export_events_log,
    export_delays_table,
    export_physics_data,
    export_scenario_comparison,
    generate_markdown_report,
)
from src.renderers.plots import (
    plot_physics_profile, 
    plot_station_occupancy, 
    plot_throughput_comparison,
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
# Секции маршрутов (общие для обоих сценариев)
# ---------------------------------------------------------------------------

# Секции для приема на 2-й путь
ARR_P2_SECTIONS = [
    RouteSection(section_id="STR_ENTER", s_start=0.0, s_end=150.0, grade=0.0, radius=300.0, v_limit=40.0),
    RouteSection(section_id="TRK_P2", s_start=150.0, s_end=1400.0, grade=-3.5, radius=0.0, v_limit=40.0),
]

# Секции для прохода по главному пути
PASS_P1_SECTIONS = [
    RouteSection(section_id="STR_ENTER", s_start=0.0, s_end=150.0, grade=0.0, radius=0.0, v_limit=80.0),
    RouteSection(section_id="TRK_P1", s_start=150.0, s_end=1400.0, grade=-1.5, radius=0.0, v_limit=80.0),
    RouteSection(section_id="STR_EXIT", s_start=1400.0, s_end=1550.0, grade=0.0, radius=0.0, v_limit=80.0),
]


def _make_scenario_entries(
    train, 
    control_mode: ControlMode,
    interval_s: float,
) -> list[ScenarioEntry]:
    """Формирует расписание поездов для одного сценария.
    
    В режиме ВС интервал между поездами сокращается,
    что моделирует повышенную пропускную способность.
    """
    entries = [
        # Поезд 1: транзитный проход по главному пути
        ScenarioEntry(
            train_id="Freight-101",
            t_arrive_s=0.0,
            route_id="PASS_P1",
            train=train,
            sections=PASS_P1_SECTIONS,
            v0_kmh=60.0,
            dwell_s=0.0,
        ),
        # Поезд 2: приём на 2-й путь со стоянкой
        ScenarioEntry(
            train_id="Freight-102",
            t_arrive_s=interval_s,
            route_id="ARR_P2",
            train=train,
            sections=ARR_P2_SECTIONS,
            v0_kmh=40.0,
            dwell_s=180.0,
        ),
        # Поезд 3: второй транзитный — прибывает после второго
        ScenarioEntry(
            train_id="Freight-103",
            t_arrive_s=interval_s * 2,
            route_id="PASS_P1",
            train=train,
            sections=PASS_P1_SECTIONS,
            v0_kmh=60.0,
            dwell_s=0.0,
        ),
    ]
    return entries


def run_scenario(
    station_config,
    train,
    traction_cache: TractionCache,
    scenario_name: str,
    control_mode: ControlMode,
    interval_s: float,
    output_dir: Path,
) -> tuple[list[SimResult], list[StationEvent], dict[str, float]]:
    """Запускает один сценарий и экспортирует все артефакты."""
    
    logger.info("=" * 60)
    logger.info("Запуск сценария: %s (режим: %s, интервал: %.0f с)", 
                scenario_name, control_mode.value, interval_s)
    logger.info("=" * 60)
    
    # Пересоздаём движок ЭЦ для чистого состояния
    interlocking = InterlockingEngine(station_config)
    
    sim = SimulationEngine(
        interlocking=interlocking,
        traction_cache=traction_cache,
        scenario_name=scenario_name,
        control_mode=control_mode.value,
    )
    
    entries = _make_scenario_entries(train, control_mode, interval_s)
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
    metrics = calculate_summary_metrics(results, events=events, headway_route_id="PASS_P1")
    logger.info("Метрики [%s]: %s", scenario_name, metrics)
    
    # Графики станции
    plot_station_occupancy(events, scenario_name, station_dir / f"occupancy_{scenario_name}.png")
    
    # Markdown-отчёт
    generate_markdown_report(results, metrics, summary_dir / f"report_{scenario_name}.md")
    
    # Профили тяги
    for r in results:
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
    # Сценарий 1: АБ (автоблокировка) — классические интервалы
    # =========================================================
    _, _, metrics_ab = run_scenario(
        station_config=station_config,
        train=train,
        traction_cache=traction_cache,
        scenario_name="Demo-AB",
        control_mode=ControlMode.AB,
        interval_s=600.0,         # 10 минут между поездами
        output_dir=output_dir,
    )
    
    # =========================================================
    # Сценарий 2: ВС (виртуальная сцепка) — сокращённые интервалы
    # =========================================================
    _, _, metrics_vc = run_scenario(
        station_config=station_config,
        train=train,
        traction_cache=traction_cache,
        scenario_name="Demo-VC",
        control_mode=ControlMode.VC,
        interval_s=180.0,         # 3 минуты между поездами (сокращённый интервал)
        output_dir=output_dir,
    )
    
    # =========================================================
    # Сравнение сценариев
    # =========================================================
    scenario_metrics = {
        "Demo-AB (АБ)": metrics_ab,
        "Demo-VC (ВС)": metrics_vc,
    }
    
    export_scenario_comparison(scenario_metrics, summary_dir / "throughput_comparison.csv")
    plot_throughput_comparison(scenario_metrics, summary_dir / "throughput_benchmark.png")
    
    logger.info("=" * 60)
    logger.info("Готово! Проверьте папку 'output/'.")
    logger.info("  АБ: throughput=%.2f поездов/час, headway=%.1f с",
                metrics_ab.get("throughput_trains_per_hour", 0),
                metrics_ab.get("headway_avg_s", 0))
    logger.info("  ВС: throughput=%.2f поездов/час, headway=%.1f с",
                metrics_vc.get("throughput_trains_per_hour", 0),
                metrics_vc.get("headway_avg_s", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
