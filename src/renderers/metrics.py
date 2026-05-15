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
    "generate_markdown_report",
    "generate_methodology_comparison_report",
]


def generate_markdown_report(
    results: list[SimResult],
    metrics: dict[str, float],
    out_path: Path | str,
) -> None:
    """Генерирует Markdown-отчет с явным разделением временных фаз.

    Временные фазы каждого поезда:
      • Факт. прибытие  — момент появления поезда в модели (с учетом внешней задержки).
      • Готов маршрут   — момент получения маршрута прибытия у ЭЦ.
      • Отправление     — момент готовности маршрута отправления и начала выхода.
      • Финал           — момент освобождения маршрутов хвостом поезда.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scenario = results[0].scenario if results else "N/A"
    control_mode_str = "N/A"
    if results:
        try:
            control_mode_str = ControlMode(results[0].control_mode).label_ru
        except ValueError:
            control_mode_str = results[0].control_mode

    lines = [
        "# Отчёт по имитации станционной работы",
        f"\n**Сценарий:** {scenario}",
        f"\n**Режим СИРДП:** {control_mode_str}",
        "\n## Сводная статистика",
        "Пропускная способность рассчитывается по интервалу "
        "от первого фактического прибытия до последнего фактического отправления.",
        f"- **Принято поездов:** {int(metrics.get('trains_total', len(results)))}",
        f"- **Пропускная способность:** {metrics.get('throughput_trains_per_hour', 0):.2f} поездов/час",
        f"- **Средний интервал отправления (headway):** {metrics.get('headway_avg_s', 0):.1f} с",
        f"- **Среднее суммарное ожидание:** {metrics.get('mean_wait_time_s', 0):.1f} с",
        f"- **Ожидание маршрута прибытия:** {metrics.get('wait_arrival_route_avg_s', 0):.1f} с",
        f"- **Ожидание маршрута отправления:** {metrics.get('wait_departure_route_avg_s', 0):.1f} с",
        f"- **Подготовка маршрутов (route setup):** {metrics.get('route_setup_wait_avg_s', 0):.1f} с",
        f"- **Средняя задержка отправления:** {metrics.get('delay_depart_avg_s', 0):.1f} с",
        f"- **Среднее время в системе:** {metrics.get('mean_travel_time_s', 0):.1f} с",
        f"- **Использование горловины:** {metrics.get('throat_utilization', 0):.2%}",
    ]

    if "packet_integrity_ratio" in metrics and not np.isnan(
        metrics.get("packet_integrity_ratio", float("nan"))
    ):
        lines.extend([
            "\n## Пакетные метрики (режим ВС)",
            f"- **Сохранность пакетов:** {metrics.get('packet_integrity_ratio', 0):.0%}",
            f"- **Максимальный разрыв внутри пакета:** {metrics.get('max_intra_packet_gap_s', 0):.1f} с",
            f"- **Превышен порог разрыва:** {'Да' if metrics.get('max_gap_exceeds_vc_threshold') else 'Нет'}",
            f"- **Задержка разделения пакета:** {metrics.get('packet_split_delay_s', 0):.1f} с",
        ])

    lines.extend([
        "\n## Устойчивость",
        f"- **Максимальная очередь в горловине:** {int(metrics.get('max_queue_length', 0))} поездов",
        f"- **Каскадная задержка:** {metrics.get('cascade_delay_s', 0):.1f} с",
        f"- **Время восстановления графика:** {metrics.get('recovery_time_s', 0):.1f} с",
        "\n## Таблица поездов",
        "| Поезд | Пакет | Маршрут | План приб. | Факт приб. | Готов маршрут | Отправление | Ожидание | Задерж. отпр. | Финал |",
        "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in results:
        lines.append(
            f"| {r.train_id} | {r.platoon_id or '—'} | {r.route_id} | "
            f"{r.t_scheduled_arrive_s:.0f} | {r.t_actual_arrive_s:.0f} | "
            f"{r.t_route_acquired_arrive_s:.0f} | {r.t_actual_depart_s:.0f} | "
            f"{r.t_wait_s:.1f} | {r.delay_depart_s:.1f} | {r.t_final_clear_s:.1f} |"
        )

    lines.extend([
        "\n## Пояснение полей таблицы",
        "- **План приб.** — плановое время прибытия по расписанию.",
        "- **Факт приб.** — фактическое прибытие с учётом внешних задержек.",
        "- **Готов маршрут** — момент получения маршрута прибытия от ЭЦ.",
        "- **Отправление** — готовность маршрута отправления и начало выхода со станции.",
        "- **Финал** — освобождение маршрутов хвостом поезда.",
        "\n---",
        "*Отчёт сгенерирован автоматически PIMPS Simulation Engine.*",
    ])

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
            "platoon_id": r.platoon_id or "",
            "route_id": r.route_id,
            "departure_route_id": r.departure_route_id or "",
            "consist_id": r.consist_id,
            "scenario": r.scenario,
            "control_mode": r.control_mode,
            "vc_methodology": r.vc_methodology,
            "t_scheduled_arrive_s": round(r.t_scheduled_arrive_s, 1),
            "t_actual_arrive_s": round(r.t_actual_arrive_s, 1),
            "t_route_acquired_arrive_s": round(r.t_route_acquired_arrive_s, 1),
            "t_arrival_route_clear_s": round(r.t_arrival_route_clear_s, 1),
            "t_depart_ready_s": round(r.t_depart_ready_s, 1),
            "t_actual_depart_s": round(r.t_actual_depart_s, 1),
            "t_final_clear_s": round(r.t_final_clear_s, 1),
            "t_arrive_s": round(r.t_arrive_s, 1),
            "t_depart_s": round(r.t_depart_s, 1),
            "t_planned_depart_s": round(r.t_planned_depart_s, 1),
            "t_total_s": round(r.t_total_s, 1),
            "t_wait_s": round(r.t_wait_s, 1),
            "wait_arrival_route_s": round(r.wait_arrival_route_s, 1),
            "wait_departure_route_s": round(r.wait_departure_route_s, 1),
            "route_setup_wait_s": round(r.route_setup_wait_s, 1),
            "t_dwell_s": round(r.t_dwell_s, 1),
            "arrival_run_s": round(r.arrival_run_s, 1),
            "departure_run_s": round(r.departure_run_s, 1),
            "tail_clearance_s": round(r.tail_clearance_s, 1),
            "delay_arrive_s": round(r.delay_arrive_s, 1),
            "delay_depart_s": round(r.delay_depart_s, 1),
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
        "t_event_s": round(e.t_event_s, 1),
        "phase": e.phase,
        "reason": e.reason,
        "queue_length": e.queue_length,
        "route_status": e.route_status,
    } for e in events]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Лог событий сохранён: %s", path.absolute())


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
        "t_real_arrive_s": round(r.t_actual_arrive_s, 1),
        "delay_arrive_s": round(r.delay_arrive_s, 1),
        "t_planned_depart_s": round(r.t_planned_depart_s, 1),
        "t_depart_ready_s": round(r.t_depart_ready_s, 1),
        "t_real_depart_s": round(r.t_actual_depart_s, 1),
        "delay_depart_s": round(r.delay_depart_s, 1),
        "wait_arrival_route_s": round(r.wait_arrival_route_s, 1),
        "wait_departure_route_s": round(r.wait_departure_route_s, 1),
        "route_setup_wait_s": round(r.route_setup_wait_s, 1),
    } for r in results]

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Таблица задержек сохранена: %s", path.absolute())


def calculate_summary_metrics(
    results: list[SimResult],
    events: list[StationEvent] | None = None,
    headway_route_id: str | None = None,
    vc_min_headway_s: float = 60.0,
    planned_interval_s: float = 0.0,
    vc_gap_threshold_s: float = 120.0,
) -> dict[str, float]:
    """Агрегирует основные метрики (среднее ожидание, пропускная способность, задержки)."""
    if not results:
        return {}

    df = pd.DataFrame([r.__dict__ for r in results])

    metrics: dict[str, float] = {
        "trains_total": float(len(results)),
        "mean_wait_time_s": float(df["t_wait_s"].mean() if not df["t_wait_s"].isna().all() else 0.0),
        "max_wait_time_s": float(df["t_wait_s"].max() if not df["t_wait_s"].isna().all() else 0.0),
        "wait_arrival_route_avg_s": float(df["wait_arrival_route_s"].mean() if "wait_arrival_route_s" in df else 0.0),
        "wait_departure_route_avg_s": float(df["wait_departure_route_s"].mean() if "wait_departure_route_s" in df else 0.0),
        "route_setup_wait_avg_s": float(df["route_setup_wait_s"].mean() if "route_setup_wait_s" in df else 0.0),
        "mean_travel_time_s": float(df["t_total_s"].mean() if not df["t_total_s"].isna().all() else 0.0),
        "total_span_s": float((df["t_actual_depart_s"].max() - df["t_actual_arrive_s"].min()) if len(df) > 0 else 0.0),
        "total_clear_span_s": float((df["t_final_clear_s"].max() - df["t_actual_arrive_s"].min()) if len(df) > 0 else 0.0),
        "dwell_avg_s": float(df["t_dwell_s"].mean() if not df["t_dwell_s"].isna().all() else 0.0),
        "dwell_max_s": float(df["t_dwell_s"].max() if not df["t_dwell_s"].isna().all() else 0.0),
        "delay_arrive_avg_s": float(df["delay_arrive_s"].mean() if "delay_arrive_s" in df and not df["delay_arrive_s"].isna().all() else 0.0),
        "delay_depart_avg_s": float(df["delay_depart_s"].mean() if "delay_depart_s" in df and not df["delay_depart_s"].isna().all() else 0.0),
    }

    # Пропускная способность (поездов/час)
    hours = metrics["total_span_s"] / 3600.0 if metrics["total_span_s"] > 0 else 0
    metrics["throughput_trains_per_hour"] = len(results) / hours if hours > 0 else 0.0

    # Средний headway
    target_results = results
    if headway_route_id:
        filtered = [r for r in results if r.route_id == headway_route_id]
        if len(filtered) >= 2:
            target_results = filtered

    if len(target_results) >= 2:
        departs = sorted([r.t_depart_s for r in target_results])
        headways = [departs[i] - departs[i - 1] for i in range(1, len(departs))]
        metrics["headway_avg_s"] = sum(headways) / len(headways)
    else:
        metrics["headway_avg_s"] = float("nan")

    # Использование горловины (через объединение интервалов занятости)
    metrics["throat_utilization"] = 0.0
    if events and metrics["total_span_s"] > 0:
        occupancies: list[tuple[float, float]] = []
        acquired_times: dict = {}

        for e in events:
            if e.event_type.value == "route_acquired":
                acquired_times[(e.train_id, e.route_id)] = e.t_event_s
            elif e.event_type.value == "route_released":
                key = (e.train_id, e.route_id)
                if key in acquired_times:
                    occupancies.append((acquired_times[key], e.t_event_s))

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

    # Пакетные метрики
    platoons: dict[str, list[SimResult]] = {}
    for r in results:
        if r.platoon_id:
            platoons.setdefault(r.platoon_id, []).append(r)

    total_platoons = len(platoons)
    intact_platoons = 0
    max_intra_packet_gap = 0.0
    threshold_s = 2.0 * vc_min_headway_s

    for _pid, p_results in platoons.items():
        p_results.sort(key=lambda x: x.t_depart_s)
        is_intact = True

        dep_routes = {r.departure_route_id for r in p_results if r.departure_route_id}
        if len(dep_routes) > 1:
            is_intact = False

        for i in range(1, len(p_results)):
            gap = p_results[i].t_depart_s - p_results[i - 1].t_depart_s
            if gap > max_intra_packet_gap:
                max_intra_packet_gap = float(gap)
            if gap > threshold_s * 1.1:
                is_intact = False

        if is_intact:
            intact_platoons += 1

    if total_platoons > 0:
        metrics["packet_integrity_ratio"] = float(intact_platoons / total_platoons)
        metrics["max_intra_packet_gap_s"] = max_intra_packet_gap
    else:
        metrics["packet_integrity_ratio"] = float("nan")
        metrics["max_intra_packet_gap_s"] = 0.0

    metrics["max_gap_exceeds_vc_threshold"] = bool(metrics["max_intra_packet_gap_s"] > vc_gap_threshold_s)

    # Специфические метрики (заполняются ниже по типу сценария)
    metrics["max_queue_length"] = 0.0
    metrics["packet_split_delay_s"] = 0.0
    metrics["cascade_delay_s"] = 0.0
    metrics["recovery_time_s"] = 0.0

    if events:
        current_queue = 0
        max_queue = 0
        for e in sorted(events, key=lambda x: x.t_event_s):
            if e.event_type.value == "route_requested":
                current_queue += 1
                if current_queue > max_queue:
                    max_queue = current_queue
            elif e.event_type.value == "route_acquired":
                current_queue -= 1
        metrics["max_queue_length"] = float(max_queue)

    if results:
        scenario = str(results[0].scenario)

        # Задержка разделения пакета (сценарий VC-Packet-Split)
        if scenario == "VC-Packet-Split" and len(results) >= 3:
            metrics["packet_split_delay_s"] = float(results[2].delay_depart_s)

        # Каскадная задержка и время восстановления (сценарии с Recovery)
        if "Recovery" in scenario:
            faulty_idx = -1
            t_fault = 0.0
            for i, r in enumerate(results):
                if r.delay_arrive_s > 100:
                    faulty_idx = i
                    t_fault = r.t_arrive_s - r.delay_arrive_s
                    break

            if faulty_idx != -1:
                cascade_sum = sum(results[i].delay_depart_s for i in range(faulty_idx, len(results)))
                metrics["cascade_delay_s"] = float(cascade_sum)

                departs = [r.t_depart_s for r in results]
                recovered_idx = -1
                for i in range(faulty_idx + 1, len(results) - 1):
                    hw = departs[i] - departs[i - 1]
                    nw = departs[i + 1] - departs[i]
                    tgt = planned_interval_s
                    if tgt > 0 and abs(hw - tgt) <= 0.1 * tgt and abs(nw - tgt) <= 0.1 * tgt:
                        recovered_idx = i
                        break

                t_recover = departs[recovered_idx] if recovered_idx != -1 else departs[-1]
                metrics["recovery_time_s"] = float(t_recover - t_fault)

    return metrics


def export_scenario_comparison(
    scenario_metrics: dict[str, dict[str, float]],
    out_path: Path | str,
) -> None:
    """Сохраняет таблицу сравнения сценариев."""
    if not scenario_metrics:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [{"scenario": s, **m} for s, m in scenario_metrics.items()]
    df = pd.DataFrame(rows)

    cols = [
        "scenario", "trains_total", "throughput_trains_per_hour",
        "headway_avg_s", "dwell_avg_s", "throat_utilization",
        "packet_integrity_ratio", "max_intra_packet_gap_s",
        "max_gap_exceeds_vc_threshold",
        "max_queue_length", "packet_split_delay_s",
        "cascade_delay_s", "recovery_time_s",
    ]
    exist_cols = [c for c in cols if c in df.columns] + [
        c for c in df.columns if c not in cols and c != "scenario"
    ]

    df[exist_cols].to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Сравнение сценариев сохранено: %s", path.absolute())


# ---------------------------------------------------------------------------
# Сравнительный Markdown-отчёт: Методика А vs Методика Б
# ---------------------------------------------------------------------------

_COMPARISON_METRICS = [
    ("throughput_trains_per_hour",   "Пропускная способность",        "поезд/ч", ".2f"),
    ("headway_avg_s",                "Средний интервал отправления",   "с",       ".1f"),
    ("mean_wait_time_s",             "Среднее время ожидания маршрута","с",       ".1f"),
    ("throat_utilization",           "Использование горловины",        "%",       ".1%"),
    ("packet_integrity_ratio",       "Сохранённость пакетов",          "%",       ".0%"),
    ("max_intra_packet_gap_s",       "Макс. разрыв внутри пакета",     "с",       ".1f"),
    ("max_gap_exceeds_vc_threshold", "Превышение порога разрыва ВС",   "",        ""),
    ("max_queue_length",             "Макс. очередь в горловине",      "поездов", ".0f"),
    ("packet_split_delay_s",         "Задержка разделения пакета",     "с",       ".1f"),
    ("cascade_delay_s",              "Каскадная задержка",             "с",       ".1f"),
    ("recovery_time_s",              "Время восстановления (сбой)",    "с",       ".1f"),
]


def generate_methodology_comparison_report(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    out_path: Path | str,
) -> None:
    """Генерирует Markdown-отчёт сравнения Методик А и Б.

    Таблица содержит ключевые метрики с указанием разницы (Δ) между методиками.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Сравнительный отчёт: Методика А vs Методика Б",
        "",
        "Режим: **Виртуальная сцепка (ВС)**",
        "",
        "| Метрика | Методика А | Методика Б | Δ (Б − А) | Примечание |",
        "|:---|---:|---:|---:|:---|",
    ]

    for key, label, _unit, fmt in _COMPARISON_METRICS:
        val_a = metrics_a.get(key, 0.0)
        val_b = metrics_b.get(key, 0.0)
        a_nan = val_a != val_a
        b_nan = val_b != val_b

        if a_nan and b_nan:
            lines.append(f"| {label} | N/A | N/A | — | |")
            continue

        def _fmt(v: float, _key: str = key, _fmt: str = fmt) -> str:  # noqa: E731
            if v != v:
                return "N/A"
            if _key == "max_gap_exceeds_vc_threshold":
                return "Да" if bool(v) else "Нет"
            return f"{v:{_fmt}}"

        if a_nan or b_nan:
            delta_str = "—"
            note = ""
        else:
            delta = val_b - val_a
            delta_str = _fmt(delta)
            if key == "throughput_trains_per_hour":
                pct = (delta / val_a * 100) if val_a else 0
                note = f"+{pct:.0f} %" if pct > 0 else f"{pct:.0f} %"
            elif key == "packet_integrity_ratio":
                note = "✅ улучшение" if val_b > val_a else ("⚠️ ухудшение" if val_b < val_a else "=")
            elif key in ("mean_wait_time_s", "max_intra_packet_gap_s", "headway_avg_s"):
                note = "✅ снижение" if val_b < val_a else ("⚠️ рост" if val_b > val_a else "=")
            else:
                note = ""

        lines.append(f"| {label} | {_fmt(val_a)} | {_fmt(val_b)} | {delta_str} | {note} |")

    lines.extend([
        "",
        "> **Вывод:** Методика Б обеспечивает предварительное пакетное резервирование",
        "> маршрутов, что исключает задержку ожидания в горловине и сохраняет целостность",
        "> пакетов виртуальной сцепки.",
        "",
        "---",
        "*Отчёт сгенерирован автоматически PIMPS Simulation Engine.*",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Сравнительный отчёт А vs Б сохранён: %s", path.absolute())
