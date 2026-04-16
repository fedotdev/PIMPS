"""Визуализация результатов с помощью Matplotlib."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models import PhysicsResult, StationEvent

logger = logging.getLogger(__name__)

__all__ = ["plot_physics_profile", "plot_station_occupancy", "plot_throughput_comparison", "plot_methodology_comparison"]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_physics_profile(physics: PhysicsResult, train_id: str, out_path: Path | str) -> None:
    """Генерирует график скорости v(s) и времени t(s) по дистанции."""
    if len(physics.s_points) == 0:
        logger.warning("Нет точек для отрисовки профиля маршрута %s", physics.route_id)
        return

    path = Path(out_path)
    _ensure_dir(path)

    # Задаем размер графика (12x8 дюймов)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Ось 1: Скорость (км/ч) от расстояния
    color_v = 'tab:red'
    ax1.set_xlabel('Путь $S$, м')
    ax1.set_ylabel('Скорость $V$, км/ч', color=color_v)
    ax1.plot(physics.s_points, physics.v_profile, color=color_v, linewidth=2, label="$V(S)$")
    ax1.tick_params(axis='y', labelcolor=color_v)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Ось 2: Время (с) от расстояния
    ax2 = ax1.twinx()
    color_t = 'tab:blue'
    ax2.set_ylabel('Время $T$, с', color=color_t)
    ax2.plot(physics.s_points, physics.t_profile, color=color_t, linewidth=2, linestyle='--', label="$T(S)$")
    ax2.tick_params(axis='y', labelcolor=color_t)

    # Добавляем общий заголовок и легенду
    plt.title(f"Поезд: {train_id} | Маршрут: {physics.route_id}\nВремя: {physics.t_total_s:.1f} с", pad=15)
    
    # Объединяем легенды с двух осей
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    
    # Сохраняем и закрываем фигуру
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Профиль сохранен: %s", path.absolute())


def plot_station_occupancy(events: list[StationEvent], scenario_name: str, out_path: Path | str) -> None:
    """Строит диаграмму Ганта (занятость маршрутов по времени)."""
    if not events:
        logger.warning("Нет событий для отрисовки диаграммы Ганта.")
        return
        
    path = Path(out_path)
    _ensure_dir(path)
    
    occupancies: list[dict] = []
    acquired = {}
    
    # Сортируем события по времени
    sorted_events = sorted(events, key=lambda x: x.t_event_s)
    
    for e in sorted_events:
        if e.event_type.value == "route_acquired":
            acquired[(e.train_id, e.route_id)] = e.t_event_s
        elif e.event_type.value == "route_released":
            key = (e.train_id, e.route_id)
            if key in acquired:
                start_time = acquired.pop(key)
                duration = e.t_event_s - start_time
                occupancies.append({
                    "train_id": e.train_id,
                    "route_id": e.route_id,
                    "start": start_time,
                    "duration": duration
                })
                
    if not occupancies:
        logger.warning("Нет интервалов занятости для графика Ганта.")
        return
        
    routes = list(set(o["route_id"] for o in occupancies))
    routes.sort()
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(routes) * 0.8)))
    colors = plt.cm.tab20.colors
    
    for i, route in enumerate(routes):
        route_occs = [o for o in occupancies if o["route_id"] == route]
        for occ in route_occs:
            # Выберем цвет стабильный относительно имени поезда
            color = colors[hash(occ["train_id"]) % len(colors)]
            ax.broken_barh(
                [(occ["start"], occ["duration"])], 
                (i - 0.4, 0.8), 
                facecolors=color, 
                alpha=0.8, 
                edgecolor='black'
            )
            
            # Подпись внутри/рядом с блоком
            text_x = occ["start"] + occ["duration"] / 2
            # Если блок слишком узкий, поворачиваем текст
            rotation = 90 if occ["duration"] < 50 else 0
            ax.text(
                text_x, i, occ["train_id"], 
                ha='center', va='center', rotation=rotation, fontsize=8
            )
            
    ax.set_yticks(range(len(routes)))
    ax.set_yticklabels(routes)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Маршруты")
    ax.set_title(f"Диаграмма занятости маршрутов (Сценарий: {scenario_name})")
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("Диаграмма занятости сохранена: %s", path.absolute())


def plot_throughput_comparison(scenario_metrics: dict[str, dict[str, float]], out_path: Path | str) -> None:
    """Строит столбчатую диаграмму сравнения пропускной способности сценариев."""
    if not scenario_metrics:
        return
        
    path = Path(out_path)
    _ensure_dir(path)
    
    scenarios = list(scenario_metrics.keys())
    throughput = [scenario_metrics[s].get("throughput_trains_per_hour", 0.0) for s in scenarios]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(scenarios, throughput, color='tab:green', alpha=0.8, edgecolor='black')
    
    ax.set_ylabel("Пропускная способность, поездов/час")
    ax.set_title("Сравнение пропускной способности по сценариям")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  
            textcoords="offset points",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
                    
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("График сравнения сценариев сохранён: %s", path.absolute())


# ---------------------------------------------------------------------------
# Сравнение методик А vs Б (мультибарплот)
# ---------------------------------------------------------------------------

_COMPARE_KEYS = [
    ("throughput_trains_per_hour", "Пропускная\nспособность\n(п/ч)"),
    ("headway_avg_s",             "Ср. интервал\nотправл. (с)"),
    ("mean_wait_time_s",          "Ср. ожидание\nмаршрута (с)"),
    ("throat_utilization_pct",    "Использование\nгорловины (%)"),
    ("packet_integrity_pct",      "Сохранность\nпакетов (%)"),
    ("max_intra_packet_gap_s",    "Макс. разрыв\nв пакете (с)"),
]


def _prepare_compare_values(metrics: dict[str, float]) -> dict[str, float]:
    """Преобразует некоторые ключи для удобства отображения."""
    out = dict(metrics)
    out["throat_utilization_pct"] = metrics.get("throat_utilization", 0.0) * 100.0
    pi = metrics.get("packet_integrity_ratio", float("nan"))
    out["packet_integrity_pct"] = pi * 100.0 if pi == pi else 0.0
    return out


def plot_methodology_comparison(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    metrics_ab: dict[str, float] | None = None,
    out_path: Path | str = "",
) -> None:
    """Строит столбчатую диаграмму сравнения Методик А, Б и АБ по 6 метрикам."""
    path = Path(out_path)
    _ensure_dir(path)

    va = _prepare_compare_values(metrics_a)
    vb = _prepare_compare_values(metrics_b)
    vab = _prepare_compare_values(metrics_ab) if metrics_ab else None

    labels = [lbl for _, lbl in _COMPARE_KEYS]
    vals_a = [va.get(k, 0.0) for k, _ in _COMPARE_KEYS]
    vals_b = [vb.get(k, 0.0) for k, _ in _COMPARE_KEYS]
    vals_ab = [vab.get(k, 0.0) for k, _ in _COMPARE_KEYS] if vab else None

    x = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if vab:
        width = 0.25
        bars_ab = ax.bar(x - width, vals_ab, width, label="АБ (Базлайн)", color="#a5a5a5", edgecolor="black")
        bars_a = ax.bar(x, vals_a, width, label="Методика А", color="#5b9bd5", edgecolor="black")
        bars_b = ax.bar(x + width, vals_b, width, label="Методика Б", color="#ed7d31", edgecolor="black")
        bar_groups = (bars_ab, bars_a, bars_b)
    else:
        width = 0.35
        bars_a = ax.bar(x - width / 2, vals_a, width, label="Методика А", color="#5b9bd5", edgecolor="black")
        bars_b = ax.bar(x + width / 2, vals_b, width, label="Методика Б", color="#ed7d31", edgecolor="black")
        bar_groups = (bars_a, bars_b)

    # Подписи значений над столбцами
    for bar_group in bar_groups:
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Сравнение Методик (АБ, ВС-А, ВС-Б)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("График сравнения методик сохранён: %s", path.absolute())
