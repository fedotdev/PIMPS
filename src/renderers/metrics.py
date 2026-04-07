"""Модуль вычисления и экспорта метрик симуляции."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import SimResult, PhysicsResult, StationEvent, ControlMode

logger = logging.getLogger(__name__)

__all__ = [
    "export_sim_results", 
    "export_physics_data",
    "export_events_log",
    "export_delays_table",
    "export_scenario_comparison",
    "calculate_summary_metrics", 
    "generate_markdown_report"
]


def generate_markdown_report(
    results: list[SimResult], 
    metrics: dict[str, float], 
    out_path: Path | str
) -> None:
    """Генерирует наглядный отчет о работе станции в формате Markdown."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trains_total = int(metrics.get('trains_total', len(results)))

    # Определяем режим СИРДП (из первого результата)
    control_mode_str = "N/A"
    if results:
        try:
            cm = ControlMode(results[0].control_mode)
            control_mode_str = cm.label_ru
        except ValueError:
            control_mode_str = results[0].control_mode

    lines = [
        "# 🚉 Отчет по имитации станционной работы",
        f"\n**Сценарий:** {results[0].scenario if results else 'N/A'}",
        f"\n**Режим СИРДП:** {control_mode_str}",
        "\n## 📊 Сводная статистика",
        f"- **Принято поездов:** {trains_total}",
        f"- **Среднее время ожидания (t_wait):** {metrics.get('mean_wait_time_s', 0):.1f} с",
        f"- **Макс. время ожидания (t_wait):** {metrics.get('max_wait_time_s', 0):.1f} с",
        f"- **Средняя задержка отправления:** {metrics.get('delay_depart_avg_s', 0):.1f} с",
        f"- **Среднее время хода по станции:** {metrics.get('mean_travel_time_s', 0):.1f} с",
        f"- **Средний простой (dwell):** {metrics.get('dwell_avg_s', 0):.1f} с",
        f"- **Макс. простой (dwell):** {metrics.get('dwell_max_s', 0):.1f} с",
        f"- **Пропускная способность:** {metrics.get('throughput_trains_per_hour', 0):.2f} поездов/час",
        f"- **Средний интервал отправления (headway):** {metrics.get('headway_avg_s', 0):.1f} с",
        f"- **Использование горловины:** {metrics.get('throat_utilization', 0):.2%}",
        "\n## 📋 Журнал прибытия (Прием поездов)",
        "| Поезд | Маршрут | Состав | Прибытие (с) | Ожидание (с) | Задержка отпр. (с) | Стоянка (с) | Всего (с) |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|",
    ]

    for r in results:
        lines.append(
            f"| {r.train_id} | {r.route_id} | {r.consist_id} | {r.t_arrive_s:.0f} | "
            f"{r.t_wait_s:.1f} | {r.delay_depart_s:.1f} | {r.t_dwell_s:.0f} | {r.t_total_s:.1f} |"
        )

    lines.append("\n---")
    lines.append(f"\n*Отчет сгенерирован автоматически PIMPS Simulation Engine.*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info("Markdown-отчёт сформирован: %s", path.absolute())


def export_sim_results(results: list[SimResult], out_path: Path | str) -> None:
    """Сохраняет результаты каждого поезда в CSV-файл."""
    if not results:
        logger.warning("Нет результатов для экспорта в CSV.")
        return

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "train_id": r.train_id,
            "route_id": r.route_id,
            "consist_id": r.consist_id,
            "scenario": r.scenario,
            "t_arrive_s": round(r.t_arrive_s, 1),
            "t_depart_s": round(r.t_depart_s, 1),
            "t_total_s": round(r.t_total_s, 1),
            "v_avg_kmh": round(r.v_avg_kmh, 1),
            "v_max_kmh": round(r.v_max_kmh, 1),
        }
        for r in results
    ]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Результаты симуляции сохранены: %s", path.absolute())


def export_physics_data(physics: PhysicsResult, out_path: Path | str) -> None:
    """Сохраняет точки тягового расчёта (профиля движения) в CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # В physics есть v_profile (V(s)), t_profile (t(s)), s_points (S)
    data = {"s_m": physics.s_points, "t_s": physics.t_profile, "v_kmh": physics.v_profile}
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.debug("Traction profile data saved: %s", path)


def export_events_log(events: list[StationEvent], out_path: Path | str) -> None:
    """Сохраняет лог событий по станции в CSV."""
    if not events:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [{
        "train_id": e.train_id, 
        "event_type": e.event_type.value, 
        "route_id": e.route_id, 
        "section_id": e.section_id, 
        "t_event_s": round(e.t_event_s, 1)
    } for e in events]
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Лог событий сохранен: %s", path.absolute())


def export_delays_table(results: list[SimResult], out_path: Path | str) -> None:
    """Сохраняет таблицу задержек поездов."""
    if not results:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = [{
        "train_id": r.train_id,
        "scenario": r.scenario,
        "t_planned_arrive_s": round(r.t_planned_arrive_s, 1),
        "t_real_arrive_s": round(r.t_arrive_s, 1),
        "delay_arrive_s": round(r.delay_arrive_s, 1),
        "t_planned_depart_s": round(r.t_planned_depart_s, 1),
        "t_real_depart_s": round(r.t_depart_s, 1),
        "delay_depart_s": round(r.delay_depart_s, 1)
    } for r in results]
    
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Таблица задержек сохранена: %s", path.absolute())


def calculate_summary_metrics(
    results: list[SimResult],
    events: list[StationEvent] | None = None,
    headway_route_id: str | None = None
) -> dict[str, float]:
    """Агрегирует основные метрики (среднее ожидание, пропускная способность, задержки)."""
    if not results:
        return {}

    df = pd.DataFrame([r.__dict__ for r in results])

    metrics = {
        "trains_total": float(len(results)),
        "mean_wait_time_s": float(df["t_wait_s"].mean() if not df["t_wait_s"].isna().all() else 0.0),
        "max_wait_time_s": float(df["t_wait_s"].max() if not df["t_wait_s"].isna().all() else 0.0),
        "mean_travel_time_s": float(df["t_total_s"].mean() if not df["t_total_s"].isna().all() else 0.0),
        "total_span_s": float((df["t_depart_s"].max() - df["t_arrive_s"].min()) if len(df) > 0 else 0.0),
        "dwell_avg_s": float(df["t_dwell_s"].mean() if not df["t_dwell_s"].isna().all() else 0.0),
        "dwell_max_s": float(df["t_dwell_s"].max() if not df["t_dwell_s"].isna().all() else 0.0),
        "delay_arrive_avg_s": float(df["delay_arrive_s"].mean() if "delay_arrive_s" in df and not df["delay_arrive_s"].isna().all() else 0.0),
        "delay_depart_avg_s": float(df["delay_depart_s"].mean() if "delay_depart_s" in df and not df["delay_depart_s"].isna().all() else 0.0),
    }

    # Поездов в час (приближенно)
    hours = metrics["total_span_s"] / 3600.0 if metrics["total_span_s"] > 0 else 0
    metrics["throughput_trains_per_hour"] = len(results) / hours if hours > 0 else 0.0

    # Вычисляем headway (интервал отправления)
    # Сначала пробуем по ключевому маршруту, если задан
    target_results = results
    if headway_route_id:
        filtered = [r for r in results if r.route_id == headway_route_id]
        # Если по ключевому маршруту >= 2 поездов, считаем по нему;
        # иначе fallback на всех поездов сценария
        if len(filtered) >= 2:
            target_results = filtered
        
    if len(target_results) >= 2:
        departs = sorted([r.t_depart_s for r in target_results])
        headways = [departs[i] - departs[i-1] for i in range(1, len(departs))]
        metrics["headway_avg_s"] = sum(headways) / len(headways)
    else:
        # Один поезд — интервал не определён
        metrics["headway_avg_s"] = float('nan')

    # Вычисляем throat_utilization, если есть события
    metrics["throat_utilization"] = 0.0
    if events and metrics["total_span_s"] > 0:
        occupancies = [] # [(start, end)]
        acquired_times = {}
        
        for e in events:
            if e.event_type.value == "route_acquired":
                acquired_times[e.train_id] = e.t_event_s
            elif e.event_type.value == "route_released" and e.train_id in acquired_times:
                occupancies.append((acquired_times[e.train_id], e.t_event_s))
                
        # Считаем объединение интервалов занятости
        if occupancies:
            occupancies.sort()
            merged = [occupancies[0]]
            for current in occupancies[1:]:
                last = merged[-1]
                if current[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged.append(current)
            
            total_occupied_s = sum(end - start for start, end in merged)
            metrics["throat_utilization"] = total_occupied_s / metrics["total_span_s"]

    return metrics


def export_scenario_comparison(scenario_metrics: dict[str, dict[str, float]], out_path: Path | str) -> None:
    """Сохраняет таблицу сравнения сценариев."""
    if not scenario_metrics:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for scenario, metrics in scenario_metrics.items():
        row = {"scenario": scenario}
        row.update(metrics)
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # Порядок колонок, если они есть
    cols = ["scenario", "trains_total", "throughput_trains_per_hour", "headway_avg_s", "dwell_avg_s", "dwell_max_s", "throat_utilization"]
    exist_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols and c != "scenario"]
    
    df[exist_cols].to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Сравнение сценариев сохранено: %s", path.absolute())
