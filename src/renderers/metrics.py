"""Модуль вычисления и экспорта метрик симуляции."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.models import SimResult

logger = logging.getLogger(__name__)

__all__ = ["export_sim_results", "calculate_summary_metrics", "generate_markdown_report"]


def generate_markdown_report(
    results: list[SimResult], 
    metrics: dict[str, float], 
    out_path: Path | str
) -> None:
    """Генерирует наглядный отчет о работе станции в формате Markdown."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 🚉 Отчет по имитации станционной работы",
        f"\n**Сценарий:** {results[0].scenario if results else 'N/A'}",
        "\n## 📊 Сводная статистика",
        f"- **Принято поездов:** {metrics.get('trains_count', 0)}",
        f"- **Среднее время ожидания:** {metrics.get('mean_wait_time_s', 0):.1f} с",
        f"- **Макс. задержка:** {metrics.get('max_wait_time_s', 0):.1f} с",
        f"- **Среднее время хода по станции:** {metrics.get('mean_travel_time_s', 0):.1f} с",
        f"- **Пропускная способность:** {metrics.get('throughput_trains_per_hour', 0):.2f} поездов/час",
        "\n## 📋 Журнал прибытия (Прием поездов)",
        "| Поезд | Маршрут | Состав | Прибытие (с) | Ожидание (с) | Стоянка (с) | Всего (с) |",
        "|:---|:---|:---|:---|:---|:---|:---|",
    ]

    for r in results:
        lines.append(
            f"| {r.train_id} | {r.route_id} | {r.consist_id} | {r.t_arrive_s:.0f} | "
            f"{r.t_wait_s:.1f} | {r.t_dwell_s:.0f} | {r.t_total_s:.1f} |"
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
            "t_wait_s": round(r.t_wait_s, 1),
            "t_dwell_s": round(r.t_dwell_s, 1),
            "t_total_s": round(r.t_total_s, 1),
        }
        for r in results
    ]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Результаты симуляции сохранены: %s", path.absolute())


def calculate_summary_metrics(results: list[SimResult]) -> dict[str, float]:
    """Агрегирует основные метрики (среднее ожидание, пропускная способность)."""
    if not results:
        return {}

    df = pd.DataFrame([r.__dict__ for r in results])

    metrics = {
        "trains_count": len(results),
        "mean_wait_time_s": df["t_wait_s"].mean(),
        "max_wait_time_s": df["t_wait_s"].max(),
        "mean_travel_time_s": df["t_total_s"].mean(),
        "total_span_s": df["t_depart_s"].max() - df["t_arrive_s"].min(),
    }

    # Поездов в час (приближенно)
    hours = metrics["total_span_s"] / 3600.0 if metrics["total_span_s"] > 0 else 0
    metrics["throughput_trains_per_hour"] = len(results) / hours if hours > 0 else 0.0

    return metrics
