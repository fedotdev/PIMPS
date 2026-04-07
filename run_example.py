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
    station_path = Path("stations/miitovskaya_station.yaml")
    station_config = load_station(station_path)
    
    # 2. Загружаем локомотив и состав
    loco_path = Path("config/2ES5k.yaml")
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
    
    # 4. Создаем расписание (Scenario) -- Прием поезда в Миитовской
    
    # Секции для приема на 2-й путь
    arr_p2_sections = [
        RouteSection(section_id="STR_ENTER", s_start=0.0, s_end=150.0, grade=0.0, radius=300.0, v_limit=40.0),
        RouteSection(section_id="TRK_P2", s_start=150.0, s_end=1400.0, grade=-3.5, radius=0.0, v_limit=40.0)
    ]
    
    # Секции для прохода по главному пути
    pass_p1_sections = [
        RouteSection(section_id="STR_ENTER", s_start=0.0, s_end=150.0, grade=0.0, radius=0.0, v_limit=80.0),
        RouteSection(section_id="TRK_P1", s_start=150.0, s_end=1400.0, grade=-1.5, radius=0.0, v_limit=80.0),
        RouteSection(section_id="STR_EXIT", s_start=1400.0, s_end=1550.0, grade=0.0, radius=0.0, v_limit=80.0)
    ]
    
    # Поезд 1: Транзитный проход по главному пути
    train_1 = ScenarioEntry(
        train_id="Freight-101",
        t_arrive_s=0.0,
        route_id="PASS_P1",
        train=train,
        sections=pass_p1_sections,
        v0_kmh=60.0,
        dwell_s=0.0
    )
    
    # Поезд 2: ПРИЕМ на 2-й путь со стоянкой (dwell_s)
    train_2 = ScenarioEntry(
        train_id="Freight-102 (Arrival)",
        t_arrive_s=600.0,  # Через 10 минут после первого
        route_id="ARR_P2",
        train=train,
        sections=arr_p2_sections,
        v0_kmh=40.0,
        dwell_s=180.0  # Стоянка 3 минуты
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
