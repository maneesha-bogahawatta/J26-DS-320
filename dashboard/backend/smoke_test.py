"""
End-to-end smoke test for the mock alerting API.

Run the server first:
    uvicorn main:app --reload

Then, in another terminal:
    python smoke_test.py

It walks the full contract the way a real pipeline and a real operator would:
post a valid event, reject a duplicate, reject pixel coordinates, reject a
name in track_id, then record an operator decision and read the audit trail
back. If this passes, your pipeline can integrate.
"""

from __future__ import annotations

import datetime
import sys
import uuid

import requests

BASE = "http://127.0.0.1:8000/api/v1"

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}{(' — ' + detail) if detail else ''}")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_event(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "component_id": "component_4",
        "incident_type": "fall",
        "camera_id": "clip_01",
        "timestamp_detected": utc_now(),
        "priority": "high",
        "status": "new",
        "summary": "Potential fall — review required.",
        "evidence": [
            {"name": "downward_motion", "value": 0.74, "unit": None},
            {"name": "low_posture_duration", "value": 3.8, "unit": "seconds"},
        ],
        "bounding_box": {"x_min": 0.30, "y_min": 0.40, "x_max": 0.56, "y_max": 0.86},
        "track_id": "person-99",
        "media_ref": "clips/clip_01.mp4",
        "media_offset_seconds": 3.0,
        "confidence": 0.77,
        "score_calibrated": False,
        "zone_id": "lobby_floor",
        "model_name": "smoke_test_detector",
        "model_version": "0.0",
    }


def main() -> int:
    print("J26-DS-320 — mock API smoke test\n")

    try:
        health = requests.get(f"{BASE}/health", timeout=5)
    except requests.exceptions.RequestException:
        print("  Could not reach the API. Start it first:")
        print("    cd backend_mock && uvicorn main:app --reload\n")
        return 1

    check("health endpoint responds", health.status_code == 200, str(health.status_code))
    check("health reports ok", health.json().get("ok") is True)

    print("\nReading the seeded queue")
    events = requests.get(f"{BASE}/events", timeout=5)
    check("events endpoint responds", events.status_code == 200)
    check("queue is not empty", len(events.json()) > 0, "seed_events.json may be missing")

    cameras = requests.get(f"{BASE}/cameras", timeout=5)
    check("cameras endpoint responds", cameras.status_code == 200)

    print("\nPosting a valid event (this is what your pipeline does)")
    event_id = f"smoke-{uuid.uuid4().hex[:8]}"
    created = requests.post(f"{BASE}/events", json=sample_event(event_id), timeout=5)
    check("valid event accepted (201)", created.status_code == 201, created.text[:200])

    fetched = requests.get(f"{BASE}/events/{event_id}", timeout=5)
    check("event can be read back", fetched.status_code == 200)
    check(
        "pipeline creation is recorded in the audit trail",
        any(a["action"] == "created" for a in fetched.json().get("audit", [])),
    )

    print("\nRejections your pipeline should expect")
    duplicate = requests.post(f"{BASE}/events", json=sample_event(event_id), timeout=5)
    check("duplicate event_id rejected (409)", duplicate.status_code == 409, duplicate.text[:200])

    pixel_box = sample_event(f"smoke-{uuid.uuid4().hex[:8]}")
    pixel_box["bounding_box"] = {"x_min": 120, "y_min": 80, "x_max": 310, "y_max": 460}
    rejected = requests.post(f"{BASE}/events", json=pixel_box, timeout=5)
    check("pixel bounding box rejected (422)", rejected.status_code == 422, rejected.text[:200])

    named = sample_event(f"smoke-{uuid.uuid4().hex[:8]}")
    named["track_id"] = "Nimal Perera"
    rejected_name = requests.post(f"{BASE}/events", json=named, timeout=5)
    check("personal name in track_id rejected (422)", rejected_name.status_code == 422)

    bad_time = sample_event(f"smoke-{uuid.uuid4().hex[:8]}")
    bad_time["timestamp_detected"] = "18/10/2026 14:05"
    rejected_time = requests.post(f"{BASE}/events", json=bad_time, timeout=5)
    check("non-ISO timestamp rejected (422)", rejected_time.status_code == 422)

    print("\nOperator decision flow")
    ack = requests.patch(
        f"{BASE}/events/{event_id}",
        json={
            "status": "acknowledged",
            "action": "acknowledge",
            "operator_id": "OP-01",
            "note": "Smoke test acknowledgement.",
        },
        timeout=5,
    )
    check("acknowledge accepted", ack.status_code == 200, ack.text[:200])
    check("status updated", ack.json().get("status") == "acknowledged")

    repeat = requests.patch(
        f"{BASE}/events/{event_id}",
        json={"status": "acknowledged", "action": "acknowledge", "operator_id": "OP-01", "note": None},
        timeout=5,
    )
    check("repeating the same action rejected (409)", repeat.status_code == 409)

    escalate = requests.patch(
        f"{BASE}/events/{event_id}",
        json={"status": "escalated", "action": "escalate", "operator_id": "OP-01", "note": "Sent a guard."},
        timeout=5,
    )
    check("escalate accepted", escalate.status_code == 200)
    check("audit trail keeps both decisions", len(escalate.json().get("audit", [])) >= 3)

    missing = requests.get(f"{BASE}/events/does-not-exist", timeout=5)
    check("unknown event id returns 404", missing.status_code == 404)

    print(f"\n{passed} passed, {failed} failed")
    if failed == 0:
        print("The API honours the contract. Your pipeline can integrate against it.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
