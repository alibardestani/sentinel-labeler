# services/progress.py
from __future__ import annotations
from threading import Lock
from dataclasses import dataclass, asdict
import time

_lock = Lock()

@dataclass
class _State:
    phase: str = "idle"
    percent: float = 0.0
    detail: str = ""
    ts: float = 0.0   # برای دیباگ / تازه‌سازی

_state = _State()

def reset():
    with _lock:
        _state.phase = "idle"
        _state.percent = 0.0
        _state.detail = ""
        _state.ts = time.time()

def set_progress(phase: str, percent: float, detail: str = ""):
    if percent < 0: percent = 0
    if percent > 100: percent = 100
    with _lock:
        _state.phase = phase
        _state.percent = float(percent)
        if detail:
            _state.detail = detail
        _state.ts = time.time()

def get_progress() -> dict:
    with _lock:
        return asdict(_state)