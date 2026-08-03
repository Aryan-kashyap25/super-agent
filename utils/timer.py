from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class ExecutionTimer:
    label: str
    started_at: float = field(default_factory=perf_counter)
    ended_at: float | None = None

    def stop(self) -> float:
        if self.ended_at is None:
            self.ended_at = perf_counter()
        return self.elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        end_time = self.ended_at if self.ended_at is not None else perf_counter()
        return end_time - self.started_at


class TimerCollection:
    def __init__(self) -> None:
        self._timers: dict[str, ExecutionTimer] = {}

    def start(self, label: str) -> ExecutionTimer:
        timer = ExecutionTimer(label=label)
        self._timers[label] = timer
        return timer

    def stop(self, label: str) -> float:
        timer = self._timers[label]
        return timer.stop()
