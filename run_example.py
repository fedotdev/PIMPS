"""Пример запуска симуляции PIMPS напрямую через API (без CLI)."""
import logging
from pathlib import Path

from src.interlocking.engine import InterlockingEngine
from src.interlocking.loader import load_station
from src.models import ScenarioEntry, RouteSection
from src.renderers.metrics import (
    calculate_summary_metrics,
    export_sim_results,
    generate_markdown_report,
)
from src.renderers.plots import plot_physics_profile
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


def main():
    logger.info("Инициализация данных...")
    
    # 1. Загружаем конфигурацию станции
    station_path = Path("stations/demo_station.yaml")
    station_config = load_station(station_path)
    
    # 2. Загружаем локомотив и состав
    loco_path = Path("config/demo_loco.yaml")
    locomotive = load_locomotive(loco_path)
    
    train_path = Path("config/demo_train.yaml")
    train = load_train(locomotive, train_path)
    
    # 3. Подготавливаем движки
    interlocking = InterlockingEngine(station_config)
    traction_cache = TractionCache()
    
    sim = SimulationEngine(
        interlocking=interlocking,
        traction_cache=traction_cache,
        scenario_name="Демо-Сценарий"
    )
    
    # Поезд 1 прибывает в момент времени 0 и идет по главному пути
    train_1 = ScenarioEntry(
        train_id="Поезд-1001",
        t_arrive_s=0.0,
        route_id="MAIN_PASS",
        train=train,
        sections=[
            RouteSection(section_id="TRK_1", s_start=0.0, s_end=5000.0, grade=2.0, radius=0.0, v_limit=80.0)
        ],
        v0_kmh=40.0,
        dwell_s=0.0
    )
    
    # Поезд 2 прибывает чуть позже на боковой путь
    train_2 = ScenarioEntry(
        train_id="Поезд-2002",
        t_arrive_s=300.0,  # Через 5 минут
        route_id="SIDE_ARRIVAL",
        train=train,
        sections=[
            RouteSection(section_id="TRK_2", s_start=0.0, s_end=5000.0, grade=0.0, radius=500.0, v_limit=40.0)
        ],
        v0_kmh=40.0,
        dwell_s=120.0  # Стоянка 2 минуты
    )
    
    # Загружаем поезда в симулятор
    sim.load_scenario([train_1, train_2])
    
    # 5. Запуск!
    logger.info("Старт симуляции SimPy...")
    results = sim.run()
    
    # 6. Рендеринг (Выходные данные)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Генерация отчетов...")
    # а) Экспорт CSV
    export_sim_results(results, output_dir / "sim_results.csv")
    
    # б) Сводные метрики
    metrics = calculate_summary_metrics(results)
    logger.info("Сводные метрики: %s", metrics)
    
    # в) Красивый Markdown-отчёт
    logger.info("Генерация Markdown-отчета...")
    generate_markdown_report(results, metrics, output_dir / "report.md")
    
    # в) Графики профилей для проехавших поездов (берем из кэша)
    # Так как кэш хранит PhysicsResult по ключам consist_id:route_id:mode:v0
    for key, physics in traction_cache._store.items():
        # Формируем имя файла из ключа, заменяя недопустимые символы
        safe_name = key.replace(":", "_")
        plot_path = output_dir / f"profile_{safe_name}.png"
        plot_physics_profile(physics, plot_path)
        
    logger.info("Готово! Проверьте папку 'output/'.")

if __name__ == "__main__":
    main()
