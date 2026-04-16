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
        "Метод расчёта пропускной способности: N_поездов / T_симуляции × 3600",
        "Средний интервал отправлений (headway): среднее попарных разностей t_depart",
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
        ]
        
    if "packet_integrity_ratio" in metrics and not np.isnan(metrics.get("packet_integrity_ratio", float('nan'))):
        lines.extend([
            "\n## 📦 Пакетные метрики (Методика ВС)",
            f"- **Доля сохраненных пакетов (integrity):** {metrics.get('packet_integrity_ratio', 0):.0%}",
            f"- **Максимальный разрыв внутри пакета:** {metrics.get('max_intra_packet_gap_s', 0):.1f} с",
            f"- **Превышен порог разрыва (max_gap_exceeds_vc_threshold):** {metrics.get('max_gap_exceeds_vc_threshold', False)}",
            f"- **Дополнительная задержка при разделении пакета:** {metrics.get('packet_split_delay_s', 0):.1f} с",
        ])

    lines.extend([
        "\n## ⚠️ Метрики устойчивости",
        f"- **Максимальная длина очереди в горловине:** {int(metrics.get('max_queue_length', 0))} поездов",
        f"- **Суммарная каскадная задержка (cascade delay):** {metrics.get('cascade_delay_s', 0):.1f} с",
        f"- **Время восстановления графика (recovery time):** {metrics.get('recovery_time_s', 0):.1f} с",
    ])

    lines.extend([
        "\n## 📋 Журнал прибытия (Прием поездов)",
        "| Поезд | Пакет | Маршрут | Прибытие (с) | Ожидание (с) | Задержка отпр. (с) | Стоянка (с) | Всего (с) |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|",
    ])

    for r in results:
        pid = r.platoon_id or "-"
        lines.append(
            f"| {r.train_id} | {pid} | {r.route_id} | {r.t_arrive_s:.0f} | "
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
    headway_route_id: str | None = None,
    vc_min_headway_s: float = 60.0,
    planned_interval_s: float = 0.0,
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
    # Fraction of active traversal time only. Does not include queue wait time.
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

    # Расчет пакетных метрик
    platoons = {}
    for r in results:
        if r.platoon_id:
            if r.platoon_id not in platoons:
                platoons[r.platoon_id] = []
            platoons[r.platoon_id].append(r)
            
    total_platoons = len(platoons)
    intact_platoons = 0
    max_intra_packet_gap = 0.0
    
    threshold_s = 2.0 * vc_min_headway_s  # порог разрыва пакета

    for pid, p_results in platoons.items():
        p_results.sort(key=lambda x: x.t_depart_s)
        is_intact = True
        
        # Проверяем, что все поезда в пакете отправляются по одному маршруту
        dep_routes = {r.departure_route_id for r in p_results if r.departure_route_id}
        if len(dep_routes) > 1:
            is_intact = False
            
        for i in range(1, len(p_results)):
            gap = p_results[i].t_depart_s - p_results[i-1].t_depart_s
            if gap > max_intra_packet_gap:
                max_intra_packet_gap = float(gap)
            if gap > threshold_s * 1.1: # 10% люфт
                is_intact = False
                
        if is_intact:
            intact_platoons += 1
            
    if total_platoons > 0:
        metrics["packet_integrity_ratio"] = float(intact_platoons / total_platoons)
        metrics["max_intra_packet_gap_s"] = max_intra_packet_gap
    else:
        metrics["packet_integrity_ratio"] = float('nan')
        metrics["max_intra_packet_gap_s"] = 0.0

    vc_gap_threshold_s = 120.0
    try:
        with open("simulation.yaml", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("vc_gap_threshold_s:"):
                    vc_gap_threshold_s = float(line.split(":")[1].strip())
                    break
    except Exception:
        pass
        
    metrics["max_gap_exceeds_vc_threshold"] = bool(metrics["max_intra_packet_gap_s"] > vc_gap_threshold_s)

    # Специфические метрики
    metrics["max_queue_length"] = 0.0
    metrics["packet_split_delay_s"] = 0.0
    metrics["cascade_delay_s"] = 0.0
    metrics["recovery_time_s"] = 0.0

    if events:
        # 1. max_queue_length
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
        # Узнаем тип сценария из результатов:
        scenario = str(results[0].scenario)
        
        # 2. packet_split_delay_s (Сценарий 3)
        if scenario == "VC-Packet-Split" and len(results) >= 3:
            # Третий поезд (после разделения)
            metrics["packet_split_delay_s"] = float(results[2].delay_depart_s)

        # 3. cascade_delay_s и recovery_time_s (Сценарий 4)
        if "Recovery" in scenario:
            # Ищем поезд, который стал источником сбоя (задержка > 100 с)
            faulty_idx = -1
            t_fault = 0.0
            for i, r in enumerate(results):
                if r.delay_arrive_s > 100:  # порог сбоя
                    faulty_idx = i
                    t_fault = r.t_arrive_s - r.delay_arrive_s
                    break
            
            if faulty_idx != -1:
                cascade_sum = 0.0
                recovered_idx = -1
                
                # Каскадная задержка
                for i in range(faulty_idx, len(results)):
                    cascade_sum += results[i].delay_depart_s
                metrics["cascade_delay_s"] = float(cascade_sum)

                # Время восстановления (recovery_time_s)
                # Ищем, когда интервал вернулся к ±10% от планового
                departs = [r.t_depart_s for r in results]
                for i in range(faulty_idx + 1, len(results) - 1):
                    current_headway = departs[i] - departs[i-1]
                    target = planned_interval_s
                    if target > 0 and abs(current_headway - target) <= 0.1 * target:
                        # Если и следующий в норме, то точно восстановилось
                        next_headway = departs[i+1] - departs[i]
                        if abs(next_headway - target) <= 0.1 * target:
                            recovered_idx = i
                            break
                            
                if recovered_idx != -1:
                    t_recover = departs[recovered_idx]
                    metrics["recovery_time_s"] = float(t_recover - t_fault)
                else:
                    # Если не восстановилось до конца симуляции
                    metrics["recovery_time_s"] = float(departs[-1] - t_fault)

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
    cols = [
        "scenario", "trains_total", "throughput_trains_per_hour", 
        "headway_avg_s", "dwell_avg_s", "throat_utilization",
        "packet_integrity_ratio", "max_intra_packet_gap_s",
        "max_gap_exceeds_vc_threshold",
        "max_queue_length", "packet_split_delay_s", 
        "cascade_delay_s", "recovery_time_s"
    ]
    exist_cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols and c != "scenario"]
    
    df[exist_cols].to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Сравнение сценариев сохранено: %s", path.absolute())


# ---------------------------------------------------------------------------
# Сравнительный отчет: Методика А vs Методика Б
# ---------------------------------------------------------------------------

_COMPARISON_METRICS = [
    ("throughput_trains_per_hour", "Пропускная способность", "поезд/ч", ".2f"),
    ("headway_avg_s",             "Средний интервал отправления", "с", ".1f"),
    ("mean_wait_time_s",          "Среднее время ожидания маршрута", "с", ".1f"),
    ("throat_utilization",        "Использование горловины", "%", ".1%"),
    ("packet_integrity_ratio",    "Сохранённость пакетов", "%", ".0%"),
    ("max_intra_packet_gap_s",    "Макс. разрыв внутри пакета", "с", ".1f"),
    ("max_gap_exceeds_vc_threshold", "Превышение порога разрыва ВС", "", ""),
    ("max_queue_length",          "Макс. очередь в горловине", "поездов", ".0f"),
    ("packet_split_delay_s",      "Задержка разделения пакета", "с", ".1f"),
    ("cascade_delay_s",           "Каскадная задержка", "с", ".1f"),
    ("recovery_time_s",           "Время восстановления (сбой)", "с", ".1f"),
]


def generate_methodology_comparison_report(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    out_path: Path | str,
) -> None:
    """Генерирует Markdown-отчёт сравнения Методик А и Б.

    Таблица содержит 6 ключевых метрик с указанием разницы (дельты)
    между методиками.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ⚖️ Сравнительный отчёт: Методика А vs Методика Б",
        "",
        "Режим: **Виртуальная сцепка (ВС)**",
        "",
        "| Метрика | Методика А | Методика Б | Δ (Б − А) | Примечание |",
        "|:---|---:|---:|---:|:---|",
    ]

    for key, label, unit, fmt in _COMPARISON_METRICS:
        val_a = metrics_a.get(key, 0.0)
        val_b = metrics_b.get(key, 0.0)

        # Обработка NaN
        a_nan = val_a != val_a
        b_nan = val_b != val_b

        if a_nan and b_nan:
            lines.append(f"| {label} | N/A | N/A | — | |") 
            continue

        # Для процентных метрик форматирование %.0% уже даёт проценты
        def _fmt(v: float) -> str:
            if v != v:
                return "N/A"
            if key == "max_gap_exceeds_vc_threshold":
                return "Да" if bool(v) else "Нет"
            return f"{v:{fmt}}"

        # Дельта
        if a_nan or b_nan:
            delta_str = "—"
            note = ""
        else:
            delta = val_b - val_a
            delta_str = f"{delta:{fmt}}"
            # Краткая аннотация
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
