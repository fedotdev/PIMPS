"""Визуализация результатов с помощью Matplotlib."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models import PhysicsResult

logger = logging.getLogger(__name__)

__all__ = ["plot_physics_profile"]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_physics_profile(physics: PhysicsResult, out_path: Path | str) -> None:
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
    plt.title(f"Поезд: {physics.consist_id} | Маршрут: {physics.route_id}\nВремя: {physics.t_total_s:.1f} с", pad=15)
    
    # Объединяем легенды с двух осей
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    
    # Сохраняем и закрываем фигуру
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info("Профиль сохранен: %s", path.absolute())
