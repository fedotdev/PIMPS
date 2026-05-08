"""
Пакет traction — тяговые расчёты по ПТР РЖД 2016.

Публичный API
-------------
Для использования в SimPy-процессах и тестах достаточно импортировать
из этого пакета:

    from src.traction import solve_route, TractionCache, head_to_tail_profile
    from src.traction import load_train, DriveMode

Внутренние модули (loader, dynamics) импортировать напрямую не нужно —
весь интерфейс сосредоточен здесь.
"""


import sys
from pathlib import Path

# Позволяет напрямую запускать этот файл как скрипт из IDE (без флага -m)
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


from src.traction.dynamics import (
    DriveMode,
    TractionCache,
    apply_speed_limit,
    head_to_tail_profile,
    solve_route,
)
from src.traction.loader import load_train

__all__ = [
    "DriveMode",
    "TractionCache",
    "apply_speed_limit",
    "head_to_tail_profile",
    "load_train",
    "solve_route",
]
