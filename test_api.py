"""
test_api.py – Integration tests for FastAPI endpoints
======================================================

Run with:  pytest test_api.py -v

Tests the four endpoints (/health, /reset, /step, /state) using FastAPI's
TestClient (synchronous wrapper around httpx).  No external LLM or
network calls are needed — these hit the in-process ASGI app directly.

Coverage
--------
  • GET  /health  → 200, fields present
  • POST /reset   → 200 with body, 200 without body, 422 on bad level
  • POST /step    → 200, reward in (0,1), done flag, episode-over guard
  • GET  /state   → 200, expected keys present
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from env import app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Yield a fresh TestClient; environment state is shared (singleton)."""
    with TestClient(app) as c:
        yield c


# ═════════════════════════════════════════════════════════════════════════════
# GET /health
# ═════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ═════════════════════════════════════════════════════════════════════════════
# POST /reset
# ═════════════════════════════════════════════════════════════════════════════

class TestReset:
    def test_reset_with_body(self, client):
        resp = client.post("/reset", json={"level": "easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "task_id" in data
        assert data["level"] == "easy"
        assert data["total_tasks"] == 3

    def test_reset_without_body(self, client):
        """The OpenEnv validator sends POST /reset with no body."""
        resp = client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["level"] == "easy"
        assert "observation" in data

    def test_reset_medium(self, client):
        resp = client.post("/reset", json={"level": "medium"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["level"] == "medium"
        assert data["total_tasks"] == 4

    def test_reset_hard(self, client):
        resp = client.post("/reset", json={"level": "hard"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["level"] == "hard"
        assert data["total_tasks"] == 3

    def test_reset_invalid_level(self, client):
        resp = client.post("/reset", json={"level": "nightmare"})
        assert resp.status_code == 422

    def test_reset_returns_observation_fields(self, client):
        data = client.post("/reset", json={"level": "easy"}).json()
        obs = data["observation"]
        assert "sender" in obs
        assert "subject" in obs
        assert "body" in obs
        assert "spf_record" in obs

    def test_reset_returns_task_id(self, client):
        data = client.post("/reset", json={"level": "easy"}).json()
        assert data["task_id"].startswith("lv")

    def test_reset_returns_task_group(self, client):
        data = client.post("/reset", json={"level": "easy"}).json()
        assert data["task_group"] == "easy"


# ═════════════════════════════════════════════════════════════════════════════
# POST /step
# ═════════════════════════════════════════════════════════════════════════════

class TestStep:
    def test_step_returns_200(self, client):
        client.post("/reset", json={"level": "easy"})
        resp = client.post("/step", json={"action": "MARK_SAFE"})
        assert resp.status_code == 200

    def test_step_has_required_fields(self, client):
        client.post("/reset", json={"level": "easy"})
        data = client.post("/step", json={"action": "QUARANTINE"}).json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "task_id" in data
        assert "is_correct" in data
        assert "info" in data

    def test_step_reward_in_open_interval(self, client):
        client.post("/reset", json={"level": "easy"})
        data = client.post("/step", json={"action": "MARK_SAFE"}).json()
        assert 0.0 < data["reward"] < 1.0

    def test_step_with_reasoning(self, client):
        client.post("/reset", json={"level": "easy"})
        resp = client.post("/step", json={
            "action": "QUARANTINE",
            "reasoning": "Suspicious sender domain",
        })
        assert resp.status_code == 200

    def test_full_easy_episode(self, client):
        """Run all 3 easy tasks and verify done=True at the end."""
        client.post("/reset", json={"level": "easy"})
        done = False
        steps = 0
        while not done and steps < 10:
            data = client.post("/step", json={"action": "QUARANTINE"}).json()
            done = data["done"]
            steps += 1
        assert done is True
        assert steps <= 5  # easy has 3 tasks; should never exceed that

    def test_step_after_episode_done(self, client):
        """Steps after episode ends should return done=True gracefully."""
        client.post("/reset", json={"level": "easy"})
        # Exhaust all tasks
        for _ in range(5):
            resp = client.post("/step", json={"action": "QUARANTINE"})
        # Extra step after episode is over
        data = client.post("/step", json={"action": "MARK_SAFE"}).json()
        assert data["done"] is True

    def test_step_invalid_action_still_200(self, client):
        """Invalid actions are graded as WRONG_PROCEDURE, not rejected with 4xx."""
        client.post("/reset", json={"level": "easy"})
        resp = client.post("/step", json={"action": "DELETE_EVERYTHING"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reward"] == 0.10  # R_WRONG_PROCEDURE


# ═════════════════════════════════════════════════════════════════════════════
# GET /state
# ═════════════════════════════════════════════════════════════════════════════

class TestState:
    def test_state_returns_200(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_has_expected_keys(self, client):
        client.post("/reset", json={"level": "easy"})
        data = client.get("/state").json()
        assert "active_level" in data
        assert "current_task_idx" in data
        assert "health" in data
        assert "score" in data
        assert "task_scores" in data
        assert "scenarios_total" in data
        assert "overall_score" in data

    def test_state_after_reset(self, client):
        client.post("/reset", json={"level": "medium"})
        data = client.get("/state").json()
        assert data["active_level"] == "medium"
        assert data["health"] == 3
        assert data["score"] == 0.0
        assert data["task_scores"] == []
        assert data["current_task_idx"] == 0

    def test_state_after_step(self, client):
        client.post("/reset", json={"level": "easy"})
        client.post("/step", json={"action": "QUARANTINE"})
        data = client.get("/state").json()
        assert data["current_task_idx"] == 1
        assert len(data["task_scores"]) == 1
