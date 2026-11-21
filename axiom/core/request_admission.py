"""Bounded FIFO request admission for local inference workers."""
from __future__ import annotations

import threading
import time
from collections import deque
from threading import Event
from typing import Any


class RequestAdmissionError(RuntimeError):
    """Base class for bounded request admission failures."""


class RequestAdmissionQueueFull(RequestAdmissionError):
    """The configured request queue has no capacity."""


class RequestAdmissionTimeout(RequestAdmissionError):
    """The request expired while waiting for a native execution slot."""


class RequestAdmissionCancelled(RequestAdmissionError):
    """The client disconnected or cancelled while waiting in the queue."""


class BoundedRequestAdmission:
    """FIFO, cancellation-aware admission around a bounded number of native workers."""

    def __init__(self, *, max_active: int = 1, max_queue: int = 32) -> None:
        if type(max_active) is not int or not 1 <= max_active <= 8:
            raise ValueError("Maximum active requests must be an integer from 1 through 8.")
        if type(max_queue) is not int or not 0 <= max_queue <= 128:
            raise ValueError("Maximum queue depth must be an integer from 0 through 128.")
        self._max_active = max_active
        self._max_queue = max_queue
        self._active = 0
        self._waiters: deque[object] = deque()
        self._condition = threading.Condition(threading.Lock())

    @property
    def waiting_count(self) -> int:
        with self._condition:
            return len(self._waiters)

    @property
    def active_count(self) -> int:
        with self._condition:
            return self._active

    def acquire(
        self,
        *,
        queue_limit: int,
        timeout_seconds: float,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        if type(queue_limit) is not int or not 0 <= queue_limit <= self._max_queue:
            raise ValueError(f"Queue limit must be an integer from 0 through {self._max_queue}.")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 900:
            raise ValueError("Admission timeout must be a finite number above zero and at most 900 seconds.")
        started = time.monotonic()
        deadline = started + float(timeout_seconds)
        ticket = object()
        with self._condition:
            if cancel_event is not None and cancel_event.is_set():
                raise RequestAdmissionCancelled("Request was cancelled before admission.")
            if self._active < self._max_active and not self._waiters:
                self._active += 1
                return {"queue_position": 0, "waited_ms": 0}
            if queue_limit == 0 or len(self._waiters) >= queue_limit or len(self._waiters) >= self._max_queue:
                raise RequestAdmissionQueueFull("The native request queue is full; retry after capacity becomes available.")
            self._waiters.append(ticket)
            queue_position = len(self._waiters)
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RequestAdmissionCancelled("Request was cancelled while waiting in the native queue.")
                    if self._active < self._max_active and self._waiters and self._waiters[0] is ticket:
                        self._waiters.popleft()
                        self._active += 1
                        self._condition.notify_all()
                        return {
                            "queue_position": queue_position,
                            "waited_ms": max(0, int((time.monotonic() - started) * 1000)),
                        }
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RequestAdmissionTimeout("The native request queue wait exceeded the configured timeout.")
                    self._condition.wait(min(0.05, remaining))
            finally:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                    self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("Native request admission was released without an active slot.")
            self._active -= 1
            self._condition.notify_all()
