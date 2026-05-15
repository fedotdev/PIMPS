"""Визуализация результатов с помощью Matplotlib."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models import PhysicsResult, StationEvent

logger = logging.getLogger(__name__)

__all__ = [
    "plot_physics_profile",
    "plot_station_occupancy",
    "plot_throughput_comparison",
    "plot_methodology_comparison",
]

# ---------------------------------------------------------------------------
# Единая цветовая палитра сценариев (используется во всех графиках)
# ---------------------------------------------------------------------------
SCENARIO_COLORS: dict[str, str] = {
    "Demo-AB":         "#a5a5a5",  # серый        — базовый АБ
    "Baseline":        "#c8c8c8",  # светло-серый
    "Demo-VC-A":       "#5b9bd5",  # синий        — ВС методика А
    "Demo-VC-B":       "#ed7d31",  # оранжевый    — ВС методика Б
    "VC-Packet-Split": "#ffc000",  # жёлтый       — разделение пакета
    "AB-Recovery":     "#b0b0b0",  # серый-2      — восстановление АБ
    "VC-Recovery":     "#70ad47",  # зелёный      — восстановление ВС
}
_DEFAULT_COLOR = "#4472c4"

# ---------------------------------------------------------------------------
# Словарь отображаемых имён сценариев
# Все функции графиков обязаны использовать _get_scenario_display_name()
# вместо прямого вывода идентификатора сценария.
# ---------------------------------------------------------------------------
_SCENARIO_DISPLAY_NAMES: dict[str, str] = {
    "Demo-AB":         "АБ (базовый)",
    "Baseline":        "АБ (базовый)",
    "Demo-VC-A":       "ВС — Методика А",
    "Demo-VC-B":       "ВС — Методика Б",
    "VC-Packet-Split": "ВС — Разделение пакета",
    "AB-Recovery":     "АБ — Восстановление",
    "VC-Recovery":     "ВС — Восстановление",
}


def _get_scenario_display_name(scenario_id: str) -> str:
    """Возвращает читаемое название сценария для подписей осей и заголовков.

    Если идентификатор отсутствует в словаре — возвращает его как есть,
    заменяя дефисы на пробелы для минимальной читаемости.
    """
    return _SCENARIO_DISPLAY_NAMES.get(scenario_id, scenario_id.replace("-", " "))


# ---------------------------------------------------------------------------
# Словарь отображаемых имён маршрутов горловины
#
# Правило именования (по нотации СЦБ):
#   Маршрут задаётся от открывающего светофора до пути назначения.
#   Приём:       «Н → 2П»   (входной светофор Н открывается, поезд принимается на 2П)
#   Отправление: «Н2 → А»   (маршрутный светофор Н2 открывается, поезд уходит на перегон А)
#   Сквозной:    «Н → Ч»    (сквозной пропуск, открываются оба конца)
#
# Ключи словаря соответствуют полю route_id в stations/miitovskaya_station.yaml.
# ---------------------------------------------------------------------------
_ROUTE_DISPLAY_NAMES: dict[str, str] = {
    # ── Сквозной пропуск ────────────────────────────────────────────────────
    "route_pass_1P":    "Сквозной: Н → 1П → Ч",

    # ── Приём со стороны А (нечётная горловина, входной светофор Н) ─────────
    "route_N_1P":       "Приём Н → 1П",
    "route_N_2P":       "Приём Н → 2П",
    "route_N_3P":       "Приём Н → 3П",
    "route_N_5P":       "Приём Н → 5П",

    # ── Приём со стороны Б (чётная горловина, входной светофор Ч) ───────────
    "route_CH_1P":      "Приём Ч → 1П",
    "route_CH_2P":      "Приём Ч → 2П",
    "route_CH_3P":      "Приём Ч → 3П",
    "route_CH_5P":      "Приём Ч → 5П",
    "route_CH_1P_var":  "Приём Ч → 1П (вариант.)",

    # ── Отправление в сторону А (маршрутные светофоры Н1, Н2, Н3, Н5) ──────
    "route_1P_A":       "Отпр. Н1 → перег. А",
    "route_2P_A":       "Отпр. Н2 → перег. А",
    "route_3P_A":       "Отпр. Н3 → перег. А",
    "route_5P_A":       "Отпр. Н5 → перег. А",
    "route_2P_A_var":   "Отпр. Н2 → перег. А (вариант.)",

    # ── Отправление в сторону Б (маршрутные светофоры Ч1, Ч2, Ч3, Ч5) ──────
    "route_1P_B":       "Отпр. Ч1 → перег. Б",
    "route_2P_B":       "Отпр. Ч2 → перег. Б",
    "route_3P_B":       "Отпр. Ч3 → перег. Б",
    "route_5P_B":       "Отпр. Ч5 → перег. Б",
    "route_2P_B_var":   "Отпр. Ч2 → перег. Б (вариант.)",
}


def _get_route_display_name(route_id: str) -> str:
    """Возвращает читаемое название маршрута для подписей осей и заголовков.

    Если route_id есть в словаре — возвращает готовую строку.
    Иначе применяет эвристику: «route_X_Y» → «X → Y».
    """
    if route_id in _ROUTE_DISPLAY_NAMES:
        return _ROUTE_DISPLAY_NAMES[route_id]
    # Эвристика для неизвестных идентификаторов
    cleaned = re.sub(r"^route_", "", route_id)
    return cleaned.replace("_", " → ")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Тяговый профиль V(s) и T(s)
# ---------------------------------------------------------------------------

def plot_physics_profile(
    physics: PhysicsResult,
    train_id: str,
    out_path: Path | str,
    v_limit_kmh: float | None = None,
    section_boundaries_m: list[float] | None = None,
) -> None:
    """Генерирует график скорости V(s) и времени T(s) по дистанции.

    Parameters
    ----------
    physics : PhysicsResult
        Результат тягового расчёта.
    train_id : str
        Идентификатор поезда (для заголовка).
    out_path : Path | str
        Путь сохранения PNG.
    v_limit_kmh : float | None
        Допустимая скорость (км/ч); если задана — отображается пунктиром.
    section_boundaries_m : list[float] | None
        Координаты (м) границ секций маршрута; рисуются вертикальными линиями.
    """
    if len(physics.s_points) == 0:
        logger.warning("Нет точек для отрисовки профиля маршрута %s", physics.route_id)
        return

    path = Path(out_path)
    _ensure_dir(path)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Ось 1: скорость ---
    color_v = "#c0392b"
    ax1.set_xlabel("Путь $S$, м", fontsize=11)
    ax1.set_ylabel("Скорость $V$, км/ч", color=color_v, fontsize=11)
    ax1.plot(
        physics.s_points, physics.v_profile,
        color=color_v, linewidth=2.2, label="$V(S)$", zorder=3,
    )
    ax1.tick_params(axis="y", labelcolor=color_v)
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=0)

    # Линия допустимой скорости
    if v_limit_kmh is not None and v_limit_kmh > 0:
        ax1.axhline(
            v_limit_kmh, color="#e67e22", linestyle="--", linewidth=1.6,
            label=f"$V_{{доп}}$ = {v_limit_kmh:.0f} км/ч", zorder=2,
        )

    # Аннотация максимальной скорости
    if len(physics.v_profile) > 0:
        v_max_val = float(np.max(physics.v_profile))
        v_max_idx = int(np.argmax(physics.v_profile))
        s_at_vmax = physics.s_points[v_max_idx]
        ax1.annotate(
            f"$V_{{max}}$ = {v_max_val:.1f} км/ч",
            xy=(s_at_vmax, v_max_val),
            xytext=(s_at_vmax + max(30, physics.s_points[-1] * 0.03), v_max_val - 4),
            arrowprops=dict(arrowstyle="->", color=color_v, lw=1.2),
            fontsize=9, color=color_v, zorder=5,
        )

    # Вертикальные линии границ секций
    if section_boundaries_m:
        for i, s_bound in enumerate(section_boundaries_m):
            ax1.axvline(
                s_bound, color="#7f8c8d", linestyle=":", linewidth=1.2,
                alpha=0.7, zorder=1,
                label="Граница секции" if i == 0 else None,
            )

    # --- Ось 2: время ---
    ax2 = ax1.twinx()
    color_t = "#2980b9"
    ax2.set_ylabel("Время $T$, с", color=color_t, fontsize=11)
    ax2.plot(
        physics.s_points, physics.t_profile,
        color=color_t, linewidth=2.0, linestyle="--", label="$T(S)$", zorder=3,
    )
    ax2.tick_params(axis="y", labelcolor=color_t)

    # Заголовок с читаемым именем маршрута
    t_min = physics.t_total_s / 60.0
    route_display = _get_route_display_name(physics.route_id)
    fig.suptitle(
        "Тяговый расчёт движения поезда",
        fontsize=13, fontweight="bold", y=1.01,
    )
    ax1.set_title(
        f"Поезд: {train_id}   |   Маршрут: {route_display}\n"
        f"Время хода: {physics.t_total_s:.1f} с ({t_min:.1f} мин)",
        fontsize=10, pad=8,
    )

    # Объединённая легенда
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper left", fontsize=9, framealpha=0.85,
    )

    fig.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Профиль сохранен: %s", path.absolute())


# ---------------------------------------------------------------------------
# Диаграмма Ганта — занятость горловины
# ---------------------------------------------------------------------------

def plot_station_occupancy(
    events: list[StationEvent],
    scenario_name: str,
    out_path: Path | str,
) -> None:
    """Строит диаграмму Ганта занятости горловины (маршрутов приёма/отправления)."""
    if not events:
        logger.warning("Нет событий для отрисовки диаграммы Ганта.")
        return

    path = Path(out_path)
    _ensure_dir(path)

    occupancies: list[dict] = []
    acquired: dict = {}

    for e in sorted(events, key=lambda x: x.t_event_s):
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
                    "duration": duration,
                })

    if not occupancies:
        logger.warning("Нет интервалов занятости для графика Ганта.")
        return

    routes = sorted(set(o["route_id"] for o in occupancies))
    lane_height = 0.35
    route_gap = 0.3
    route_layouts: list[dict] = []
    current_y = 0.0

    for route in routes:
        route_occs = sorted(
            [o for o in occupancies if o["route_id"] == route],
            key=lambda o: (o["start"], o["start"] + o["duration"], o["train_id"]),
        )
        lane_ends: list[float] = []
        assigned_occs: list[dict] = []

        for occ in route_occs:
            start = occ["start"]
            end = occ["start"] + occ["duration"]
            lane_idx = None
            for idx, lane_end in enumerate(lane_ends):
                if start >= lane_end:
                    lane_idx = idx
                    lane_ends[idx] = end
                    break
            if lane_idx is None:
                lane_idx = len(lane_ends)
                lane_ends.append(end)
            assigned = dict(occ)
            assigned["lane"] = lane_idx
            assigned_occs.append(assigned)

        lane_count = max(1, len(lane_ends))
        route_height = lane_count * lane_height
        route_layouts.append({
            "route": route,
            "occs": assigned_occs,
            "base_y": current_y,
            "height": route_height,
            "center_y": current_y + route_height / 2,
        })
        current_y += route_height + route_gap

    total_rendered_height = max(lane_height, current_y - route_gap)
    time_start = min(o["start"] for o in occupancies)
    time_end = max(o["start"] + o["duration"] for o in occupancies)
    time_span = time_end - time_start
    major_step = 60.0 if time_span <= 720.0 else 120.0
    minor_step = major_step / 2.0
    x_min = np.floor(time_start / major_step) * major_step
    x_max = np.ceil(time_end / major_step) * major_step

    palette = list(plt.cm.Set2.colors)
    unique_trains = sorted(set(o["train_id"] for o in occupancies))
    train_color_map = {t: palette[i % len(palette)] for i, t in enumerate(unique_trains)}

    fig, ax = plt.subplots(figsize=(14, max(4, total_rendered_height * 1.3)))

    for idx, layout in enumerate(route_layouts):
        band_color = "#f4f4f4" if idx % 2 else "#ffffff"
        ax.axhspan(
            layout["base_y"], layout["base_y"] + layout["height"],
            facecolor=band_color, edgecolor="none", zorder=0,
        )
        if idx > 0:
            ax.axhline(
                layout["base_y"] - route_gap / 2,
                color="#d0d0d0", linewidth=0.8, zorder=1,
            )

    for layout in route_layouts:
        for occ in layout["occs"]:
            color = train_color_map[occ["train_id"]]
            y_bottom = layout["base_y"] + occ["lane"] * lane_height
            y_center = y_bottom + lane_height / 2
            ax.broken_barh(
                [(occ["start"], occ["duration"])],
                (y_bottom, lane_height * 0.92),
                facecolors=color, alpha=0.88,
                edgecolor="#333333", linewidth=0.6, zorder=2,
            )
            short_id = occ["train_id"].split("-")[-1] if "-" in occ["train_id"] else occ["train_id"]
            text_x = occ["start"] + occ["duration"] / 2
            rotation = 90 if occ["duration"] < 45 else 0
            ax.text(
                text_x, y_center, f"№{short_id}",
                ha="center", va="center", rotation=rotation,
                fontsize=7.5, fontweight="bold", zorder=3,
                color="#1a1a1a",
            )

    ax.set_ylim(-route_gap / 2, total_rendered_height + route_gap / 2)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([layout["center_y"] for layout in route_layouts])
    # Используем _get_route_display_name для читаемых подписей оси Y
    ax.set_yticklabels(
        [_get_route_display_name(layout["route"]) for layout in route_layouts],
        fontsize=9,
    )
    ax.set_xticks(np.arange(x_min, x_max + major_step * 0.5, major_step))
    ax.set_xticks(np.arange(x_min, x_max + minor_step * 0.5, minor_step), minor=True)
    ax.set_xlabel("Время, с", fontsize=10)
    ax.set_ylabel("Маршрут горловины", fontsize=10)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", which="major", length=5)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", which="major", linestyle="--", alpha=0.5)
    ax.grid(True, axis="x", which="minor", linestyle=":", alpha=0.22)

    # Вторая ось X — в минутах
    ax_min = ax.twiny()
    ax_min.set_xlim(x_min / 60.0, x_max / 60.0)
    ax_min.set_xlabel("Время, мин", fontsize=9, labelpad=4)
    ax_min.tick_params(axis="x", labelsize=8)

    n_trains = len(unique_trains)
    duration_min = time_span / 60.0
    scenario_display = _get_scenario_display_name(scenario_name)
    ax.set_title(
        f"Занятость горловины: диаграмма использования секций\n"
        f"Сценарий: {scenario_display}   |   Поездов: {n_trains}   |   "
        f"Продолжительность: {duration_min:.1f} мин",
        fontsize=11, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Диаграмма занятости сохранена: %s", path.absolute())


# ---------------------------------------------------------------------------
# Сравнение пропускной способности по сценариям
# ---------------------------------------------------------------------------

def plot_throughput_comparison(
    scenario_metrics: dict[str, dict[str, float]],
    out_path: Path | str,
) -> None:
    """Строит столбчатую диаграмму сравнения пропускной способности сценариев."""
    if not scenario_metrics:
        return

    path = Path(out_path)
    _ensure_dir(path)

    scenarios = list(scenario_metrics.keys())
    throughput = [scenario_metrics[s].get("throughput_trains_per_hour", 0.0) for s in scenarios]
    bar_colors = [SCENARIO_COLORS.get(s, _DEFAULT_COLOR) for s in scenarios]
    # Читаемые имена для оси X
    display_names = [_get_scenario_display_name(s) for s in scenarios]

    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 1.6), 6))
    bars = ax.bar(display_names, throughput, color=bar_colors, edgecolor="black", linewidth=0.8, zorder=3)

    # Горизонтальная линия базового АБ
    ab_val = None
    for key in ("Demo-AB", "Baseline"):
        if key in scenario_metrics:
            ab_val = scenario_metrics[key].get("throughput_trains_per_hour", None)
            break
    if ab_val:
        ax.axhline(
            ab_val, color="#555555", linestyle="--", linewidth=1.5,
            label=f"Базовый АБ ({ab_val:.1f} п/ч)", zorder=2,
        )
        ax.legend(fontsize=9, framealpha=0.85)

    ax.set_ylabel("Пропускная способность, поездов/ч", fontsize=10)
    ax.set_title("Сравнение пропускной способности по сценариям", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=20, ha="right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("График сравнения сценариев сохранён: %s", path.absolute())


# ---------------------------------------------------------------------------
# Сравнение методик А vs Б — два субплота (разные единицы)
# ---------------------------------------------------------------------------

# Группа 1: безразмерные / процентные метрики
_COMPARE_KEYS_PERC = [
    ("throughput_trains_per_hour", "Пропускная\nспособность, п/ч"),
    ("throat_utilization_pct",     "Использование\nгорловины, %"),
    ("packet_integrity_pct",       "Сохранность\nпакетов, %"),
]

# Группа 2: временные метрики (с)
# Термины скорректированы согласно нормативной терминологии:
# - «время занятия маршрута» вместо «ожидание маршрута»
# - «макс. интервал внутри пакета» вместо «задержка разделения»
_COMPARE_KEYS_TIME = [
    ("headway_avg_s",          "Ср. интервал\nотправл., с"),
    ("mean_wait_time_s",       "Ср. время занятия\nмаршрута, с"),
    ("max_intra_packet_gap_s", "Макс. интервал\nвнутри пакета, с"),
]


def _prepare_compare_values(metrics: dict[str, float]) -> dict[str, float]:
    """Преобразует ключи: процентные метрики → значения 0–100."""
    out = dict(metrics)
    out["throat_utilization_pct"] = metrics.get("throat_utilization", 0.0) * 100.0
    pi = metrics.get("packet_integrity_ratio", float("nan"))
    out["packet_integrity_pct"] = pi * 100.0 if pi == pi else 0.0
    return out


def _draw_bar_group(
    ax: plt.Axes,
    keys_labels: list[tuple[str, str]],
    va: dict[str, float],
    vb: dict[str, float],
    vab: dict[str, float] | None,
    ylabel: str,
) -> None:
    """Вспомогательная функция: рисует группу столбцов на переданной оси."""
    labels = [lbl for _, lbl in keys_labels]
    vals_a  = [va.get(k, 0.0)  for k, _ in keys_labels]
    vals_b  = [vb.get(k, 0.0)  for k, _ in keys_labels]
    vals_ab = [vab.get(k, 0.0) for k, _ in keys_labels] if vab else None

    x = np.arange(len(labels))

    if vals_ab is not None:
        width = 0.24
        bars_ab = ax.bar(x - width,   vals_ab, width, label="АБ (базовый)",    color="#a5a5a5", edgecolor="black", linewidth=0.7)
        bars_a  = ax.bar(x,           vals_a,  width, label="ВС — Методика А", color="#5b9bd5", edgecolor="black", linewidth=0.7)
        bars_b  = ax.bar(x + width,   vals_b,  width, label="ВС — Методика Б", color="#ed7d31", edgecolor="black", linewidth=0.7)
        bar_groups = (bars_ab, bars_a, bars_b)
    else:
        width = 0.35
        bars_a = ax.bar(x - width / 2, vals_a, width, label="ВС — Методика А", color="#5b9bd5", edgecolor="black", linewidth=0.7)
        bars_b = ax.bar(x + width / 2, vals_b, width, label="ВС — Методика Б", color="#ed7d31", edgecolor="black", linewidth=0.7)
        bar_groups = (bars_a, bars_b)

    for bar_group in bar_groups:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, framealpha=0.88)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)


def plot_methodology_comparison(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    metrics_ab: dict[str, float] | None = None,
    out_path: Path | str = "",
) -> None:
    """Строит двухпанельную диаграмму сравнения Методик А, Б и базового АБ.

    Верхняя панель — пропускная способность и эффективностные показатели.
    Нижняя панель  — временные метрики (интервалы, время занятия, разрывы).
    Разделение панелей необходимо: значения в п/ч и % несоизмеримы с секундами.
    """
    path = Path(out_path)
    _ensure_dir(path)

    va  = _prepare_compare_values(metrics_a)
    vb  = _prepare_compare_values(metrics_b)
    vab = _prepare_compare_values(metrics_ab) if metrics_ab else None

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 10),
        gridspec_kw={"hspace": 0.45},
    )

    fig.suptitle(
        "Сравнение показателей эффективности управления движением\n"
        "АБ (автоблокировка) vs ВС Методика А vs ВС Методика Б",
        fontsize=13, fontweight="bold", y=1.01,
    )

    _draw_bar_group(
        ax_top,
        _COMPARE_KEYS_PERC,
        va, vb, vab,
        ylabel="Значение (п/ч или %)",
    )
    ax_top.set_title(
        "Пропускная способность и эффективностные показатели",
        fontsize=11, fontweight="bold", pad=8,
    )

    _draw_bar_group(
        ax_bot,
        _COMPARE_KEYS_TIME,
        va, vb, vab,
        ylabel="Время, с",
    )
    ax_bot.set_title(
        "Временные показатели: интервалы, время занятия маршрута, разрывы в пакете",
        fontsize=11, fontweight="bold", pad=8,
    )

    fig.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("График сравнения методик сохранён: %s", path.absolute())
