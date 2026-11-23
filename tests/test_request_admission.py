from __future__ import annotations

import threading
import time

import pytest

from axiom.core.request_admission import (
    BoundedRequestAdmission,
    RequestAdmissionCancelled,
    RequestAdmissionQueueFull,
    RequestAdmissionTimeout,
)


def test_fifo_queue_admits_in_order_and_enforces_bound():
    admission = BoundedRequestAdmission(max_active=1, max_queue=2)
    first = admission.acquire(queue_limit=2, timeout_seconds=1)
    assert first == {"queue_position": 0, "waited_ms": 0}
    order = []

    def wait_for(label):
        admission.acquire(queue_limit=2, timeout_seconds=2)
        order.append(label)
        admission.release()

    second = threading.Thread(target=wait_for, args=("second",))
    third = threading.Thread(target=wait_for, args=("third",))
    second.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    third.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert admission.waiting_count == 2
    with pytest.raises(RequestAdmissionQueueFull):
        admission.acquire(queue_limit=2, timeout_seconds=1)
    admission.release()
    second.join(1)
    third.join(1)
    assert not second.is_alive() and not third.is_alive()
    assert order == ["second", "third"]
    assert admission.active_count == 0


def test_queue_wait_times_out_and_cancelled_ticket_is_removed():
    admission = BoundedRequestAdmission(max_active=1, max_queue=2)
    admission.acquire(queue_limit=2, timeout_seconds=1)
    with pytest.raises(RequestAdmissionTimeout):
        admission.acquire(queue_limit=1, timeout_seconds=0.02)
    cancel_event = threading.Event()
    result = {}

    def wait_for_cancel():
        try:
            admission.acquire(queue_limit=1, timeout_seconds=1, cancel_event=cancel_event)
        except RequestAdmissionCancelled:
            result["cancelled"] = True

    waiter = threading.Thread(target=wait_for_cancel)
    waiter.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert admission.waiting_count == 1
    cancel_event.set()
    waiter.join(1)
    assert not waiter.is_alive()
    assert result["cancelled"] is True
    assert admission.waiting_count == 0
    admission.release()


def test_queue_limits_are_scoped_per_profile_with_a_global_safety_ceiling():
    admission = BoundedRequestAdmission(max_active=1, max_queue=2)
    admission.acquire(queue_limit=1, timeout_seconds=1, queue_key="profile-a")
    order = []
    results = {}

    def wait_for(key):
        result = admission.acquire(queue_limit=1, timeout_seconds=2, queue_key=key)
        results[key] = result
        order.append(key)
        admission.release()

    first = threading.Thread(target=wait_for, args=("profile-a",))
    second = threading.Thread(target=wait_for, args=("profile-b",))
    first.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    second.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    with pytest.raises(RequestAdmissionQueueFull):
        admission.acquire(queue_limit=1, timeout_seconds=1, queue_key="profile-a")
    with pytest.raises(RequestAdmissionQueueFull):
        admission.acquire(queue_limit=1, timeout_seconds=1, queue_key="profile-c")

    admission.release()
    first.join(1)
    second.join(1)
    assert not first.is_alive() and not second.is_alive()
    assert order == ["profile-a", "profile-b"]
    assert results["profile-a"]["queue_position"] == 1
    assert results["profile-b"]["queue_position"] == 2
    assert admission.active_count == 0


def test_queue_position_is_recomputed_after_an_earlier_ticket_is_cancelled():
    admission = BoundedRequestAdmission(max_active=1, max_queue=2)
    admission.acquire(queue_limit=2, timeout_seconds=1)
    cancelled = threading.Event()
    accepted = {}

    def wait_for_cancel():
        with pytest.raises(RequestAdmissionCancelled):
            admission.acquire(queue_limit=2, timeout_seconds=1, cancel_event=cancelled)

    def wait_for_slot():
        accepted.update(admission.acquire(queue_limit=2, timeout_seconds=2))
        admission.release()

    first = threading.Thread(target=wait_for_cancel)
    second = threading.Thread(target=wait_for_slot)
    first.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    second.start()
    deadline = time.monotonic() + 1
    while admission.waiting_count < 2 and time.monotonic() < deadline:
        time.sleep(0.005)

    cancelled.set()
    first.join(1)
    assert not first.is_alive()
    admission.release()
    second.join(1)
    assert not second.is_alive()
    assert accepted["queue_position"] == 1
    assert admission.active_count == 0
