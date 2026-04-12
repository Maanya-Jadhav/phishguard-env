"""
env.py – PhishGuard-Env  |  FastAPI Environment Server
=======================================================

ARCHITECTURE ROLE
-----------------
This file IS the environment. It runs as a persistent FastAPI server on
Hugging Face Spaces (port 7860). The inference agent (inference.py) is a
separate process that interacts with it exclusively through HTTP — it NEVER
imports this module directly.

Endpoints
---------
  POST /reset   → reset for a chosen difficulty level; returns first email observation
  POST /step    → submit one triage action; returns (obs, reward, done, info)
  GET  /state   → read-only snapshot of health, score, task index
  GET  /health  → liveness probe for HF Spaces / load-balancers

LEVEL DESIGN
------------
  easy   → lv1, lv2, lv3         (3 tasks)
  medium → lv4, lv5, lv6, lv7   (4 tasks)
  hard   → lv8, lv9, lv10       (3 tasks)

State variables
---------------
  current_task_idx  : int   – index into the active scenario list
  health            : int   – lives remaining (starts at 3)
  score             : float – cumulative reward for this episode
  task_scores       : list  – per-step reward history

Reward contract
---------------
All rewards are sourced from grader.py and strictly in the open interval
(0.0, 1.0). No endpoint ever returns 0 or 1.

Health drain
------------
HEALTH_DRAIN_THRESHOLD = 0.15 (imported from grader).
Any step reward below this threshold costs one life.
  Security Breach (0.02)       → –1 life
  Business Disruption (0.05)   → –1 life
  Wrong Procedure (0.10)       → –1 life
  Cautious / partial (≥ 0.35)  → no life loss

Thread safety
-------------
_env is a singleton shared across all FastAPI requests. The asyncio.Lock
`_env_lock` serialises access to _env so concurrent /step or /reset calls
cannot race on current_task_idx, health, score, or task_scores.

OpenEnv validator compliance
-----------------------------
Every normal /step response includes:
  task_id    : str  – the scenario id (e.g. "lv3") so the validator can
                      correlate decisions to tasks.
  is_correct : bool – True when reward >= R_PERFECT, so the validator
                      can count "tasks with graders" (≥ 3 required).

BUG FIXES (v1.0.2 → v1.0.3)
-----------------------------
  • Thread-safety: asyncio.Lock added around all _env mutations.
  • total_tasks in /reset now correctly reports len(LEVEL_MAP[level])
    (the count for the chosen level only) instead of len(_env.scenarios)
    which could include stale data from a previous reset.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import PhishAction, ResetRequest, StepResponse, ResetResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phishguard.env")

# ── OpenEnv base-class (graceful degradation) ─────────────────────────────────
try:
    from openenv import OpenEnv
except ImportError:
    try:
        from openenv.core import OpenEnv  # type: ignore
    except ImportError:
        OpenEnv = object  # Runs as a plain Python class if openenv is absent

# ── Grader imports ────────────────────────────────────────────────────────────
from grader import (
    R_BREACH,
    R_PERFECT,
    HEALTH_DRAIN_THRESHOLD,
    calculate_overall_score,
    grade_action,
    grade_performance,
    GRADERS,
    SCENARIO_LOADERS,
)

VERSION = "1.0.3"

# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS  (lv1 → lv10)
# ═════════════════════════════════════════════════════════════════════════════

SCENARIOS: List[dict] = [
    # ── EASY (lv1–lv3) ───────────────────────────────────────────────────────
    {
        "id": "lv1",
        "level": "easy",
        "type": "SPAM",
        "data": {
            "sender": "win@lotto.net",
            "subject": "Claim $1M Prize – Congratulations!",
            "body": (
                "You have been selected. "
                "Click now to claim your prize before it expires."
            ),
            "links": ["http://bit.ly/123abc"],
            "has_attachments": False,
            "spf_record": "fail",
            "dmarc_record": "none",
            "urgency_level": "high",
            "confidence_hint": "SIEM: no prior communication history with this domain",
        },
        "correct": "MOVE_TO_SPAM",
    },
    {
        "id": "lv2",
        "level": "easy",
        "type": "PHISH",
        "data": {
            "sender": "support@googIe.com",
            "subject": "Urgent Security Alert – Verify Your Account",
            "body": (
                "We detected suspicious activity on your account. "
                "Verify your identity immediately."
            ),
            "links": ["http://googIe-verify.com/login"],
            "has_attachments": False,
            "spf_record": "softfail",
            "dmarc_record": "fail",
            "urgency_level": "critical",
            "confidence_hint": "Gateway: domain registered 3 days ago, 0 prior sends",
        },
        "correct": "BLOCK_DOMAIN",
    },
    {
        "id": "lv3",
        "level": "easy",
        "type": "SAFE",
        "data": {
            "sender": "boss@company.com",
            "subject": "Team Meeting Tomorrow at 9 AM",
            "body": (
                "Hi team, please be ready for our weekly sync at 9 AM "
                "in the main boardroom."
            ),
            "links": [],
            "has_attachments": False,
            "spf_record": "pass",
            "dmarc_record": "pass",
            "urgency_level": "low",
            "confidence_hint": "SIEM: sender in address book for 2+ years, 200+ prior emails",
        },
        "correct": "MARK_SAFE",
    },

    # ── MEDIUM (lv4–lv7) ─────────────────────────────────────────────────────
    {
        "id": "lv4",
        "level": "medium",
        "type": "MALWARE",
        "data": {
            "sender": "it@internal-help.com",
            "subject": "Critical Security Patch – Action Required Immediately",
            "body": (
                "A critical vulnerability has been discovered. "
                "Run fix.exe to patch your system now."
            ),
            "links": ["http://internal-help.com/fix.exe"],
            "has_attachments": True,
            "spf_record": "fail",
            "dmarc_record": "none",
            "urgency_level": "critical",
            "confidence_hint": "AV: attachment flagged as suspicious PE binary (confidence 87%)",
        },
        "correct": "QUARANTINE",
    },
    {
        "id": "lv5",
        "level": "medium",
        "type": "SAFE",
        "data": {
            "sender": "hr@yourcompany.com",
            "subject": "Updated Holiday Schedule – Please Review",
            "body": (
                "Hi all, the updated holiday list is now live on the HR portal. "
                "No action needed."
            ),
            "links": ["https://portal.yourcompany.com/holidays"],
            "has_attachments": False,
            "spf_record": "pass",
            "dmarc_record": "pass",
            "urgency_level": "low",
            "confidence_hint": (
                "SIEM: bulk-sent to all-staff distribution list, "
                "normal recurring pattern"
            ),
        },
        "correct": "MARK_SAFE",
    },
    {
        "id": "lv6",
        "level": "medium",
        "type": "BEC",
        "data": {
            "sender": "ceo@executive-mail.com",
            "subject": "URGENT – Wire Transfer Required Today",
            "body": (
                "I'm in a meeting and cannot be reached by phone. "
                "Wire $5,000 to the account below immediately."
            ),
            "links": [],
            "has_attachments": False,
            "spf_record": "pass",
            "dmarc_record": "none",
            "urgency_level": "critical",
            "confidence_hint": "SIEM: first-ever email from this domain to any company inbox",
        },
        "correct": "QUARANTINE",
    },
    {
        "id": "lv7",
        "level": "medium",
        "type": "PHISH",
        "data": {
            "sender": "notify@docs-share.net",
            "subject": "You Have a New Shared Document",
            "body": (
                "Your colleague shared a file with you. "
                "Click the link below to view it in your browser."
            ),
            "links": ["https://docs-verify.net/s/1x9az"],
            "has_attachments": False,
            "spf_record": "softfail",
            "dmarc_record": "fail",
            "urgency_level": "medium",
            "confidence_hint": (
                "Gateway: destination link redirects through 2 URL shorteners "
                "before landing"
            ),
        },
        "correct": "BLOCK_DOMAIN",
    },

    # ── HARD (lv8–lv10) ──────────────────────────────────────────────────────
    {
        "id": "lv8",
        "level": "hard",
        "type": "MALWARE",
        "data": {
            "sender": "payroll@hr-dept.net",
            "subject": "Your Bank Details Form – Action Required",
            "body": (
                "Please review and complete the attached payroll update form "
                "before Friday."
            ),
            "links": [],
            "has_attachments": True,
            "spf_record": "fail",
            "dmarc_record": "fail",
            "urgency_level": "high",
            "confidence_hint": (
                "AV: attachment is a macro-enabled Excel file (.xlsm), "
                "macros unsigned"
            ),
        },
        "correct": "QUARANTINE",
    },
    {
        "id": "lv9",
        "level": "hard",
        "type": "PHISH",
        "data": {
            "sender": "mfa@office365-auth.com",
            "subject": "Action Required: Update Your MFA Settings Now",
            "body": (
                "Your multi-factor authentication settings must be updated. "
                "Scan the QR code in the attached image to stay compliant."
            ),
            "links": [],
            "has_attachments": True,
            "spf_record": "softfail",
            "dmarc_record": "none",
            "urgency_level": "critical",
            "confidence_hint": "Gateway: QR code URL resolves to known credential-harvesting page",
        },
        "correct": "QUARANTINE",
    },
    {
        "id": "lv10",
        "level": "hard",
        "type": "BEC",
        "data": {
            "sender": "partner@trusted-firm.com",
            "subject": "Updated Project Specifications – Download Required by EOD",
            "body": (
                "Please find the revised project specs at the link below. "
                "Deadline is tomorrow morning."
            ),
            "links": ["https://trusted-partner.com/files/project_specs_final.zip"],
            "has_attachments": False,
            "spf_record": "pass",
            "dmarc_record": "pass",
            "urgency_level": "high",
            "confidence_hint": (
                "Threat Intel: trusted-firm.com added to IOC feed 6 hours ago "
                "— possible domain compromise"
            ),
        },
        "correct": "BLOCK_DOMAIN",
    },
]

# ── Level → scenario IDs mapping ─────────────────────────────────────────────
LEVEL_MAP: dict[str, list[str]] = {
    "easy":   ["lv1", "lv2", "lv3"],
    "medium": ["lv4", "lv5", "lv6", "lv7"],
    "hard":   ["lv8", "lv9", "lv10"],
}

_SCENARIO_BY_ID: dict[str, dict] = {s["id"]: s for s in SCENARIOS}


# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ═════════════════════════════════════════════════════════════════════════════

class PhishGuardEnv(OpenEnv):
    """
    OpenEnv-compliant simulation environment for SOC analyst LLM benchmarking.

    State
    -----
    current_task_idx : int   – pointer into the active (shuffled) scenario list
    health           : int   – lives remaining (3 → 0)
    score            : float – cumulative reward for this episode
    task_scores      : list  – per-step reward history
    active_level     : str   – current difficulty level
    scenarios        : list  – scenarios loaded for the current level
    """

    MAX_HEALTH: int = 3

    def __init__(self) -> None:
        self.scenarios: List[dict] = []
        self.current_task_idx: int = 0
        self.health: int = self.MAX_HEALTH
        self.score: float = 0.0
        self.task_scores: List[float] = []
        self.active_level: str = "easy"
        # Initialise with easy level so the env is never empty on startup.
        self._load_level("easy")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_level(self, level: str) -> None:
        """
        Filter and shuffle scenarios for the given difficulty level.
        Resets all state counters.

        Uses SCENARIO_LOADERS from grader.py (mirrors Focus-AI's
        TASK_LOADERS pattern) so scenario-to-level mapping is
        centralised in the grader module.
        """
        level = level.lower()
        if level not in LEVEL_MAP:
            raise ValueError(
                f"Unknown level '{level}'. Valid choices: easy | medium | hard"
            )
        # Use SCENARIO_LOADERS if available, fall back to LEVEL_MAP
        if level in SCENARIO_LOADERS:
            ids = SCENARIO_LOADERS[level]()
        else:
            ids = LEVEL_MAP[level]
        subset = [dict(_SCENARIO_BY_ID[sid]) for sid in ids]
        random.shuffle(subset)

        self.active_level    = level
        self.scenarios       = subset
        self.current_task_idx = 0
        self.health          = self.MAX_HEALTH
        self.score           = 0.0
        self.task_scores     = []

    def _is_over(self) -> bool:
        return self.health <= 0 or self.current_task_idx >= len(self.scenarios)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, level: str = "easy") -> dict:
        """
        Reset the environment for a new episode at the given difficulty level.

        Returns the first email observation dict.

        Raises
        ------
        ValueError
            If the level is unknown or maps to zero scenarios.
        """
        self._load_level(level)
        if not self.scenarios:
            raise ValueError(f"No scenarios found for level '{level}'")
        first_task = self.scenarios[self.current_task_idx]
        log.info(
            "Episode reset | level=%s | first_scenario=%s | total=%d",
            self.active_level,
            first_task["id"],
            len(self.scenarios),
        )
        return first_task["data"]

    def step(self, action_str: str) -> tuple:
        """
        Advance the simulation by one triage decision.

        Returns
        -------
        (obs, reward, done, info)
        """
        # ── Guard: episode already over ───────────────────────────────────────
        if self._is_over():
            return None, R_BREACH, True, {
                "task_id":     None,
                "task_group":  None,
                "is_correct":  False,
                "health":      self.health,
                "feedback":    "Episode already ended. Call /reset to start a new one.",
                "score":       round(self.score, 4),
                "task_scores": list(self.task_scores),
            }

        # ── Resolve current scenario ──────────────────────────────────────────
        current_task = self.scenarios[self.current_task_idx]
        task_id      = current_task["id"]

        # ── Grade the action ──────────────────────────────────────────────────
        reward, verdict_msg = grade_action(
            action_str,
            current_task["correct"],
            current_task["type"],
        )

        self.score += reward
        self.task_scores.append(reward)

        log.info(
            "Step | level=%s | task=%s | action=%s | reward=%.4f | verdict=%s",
            self.active_level,
            task_id,
            action_str.strip().upper(),
            reward,
            verdict_msg,
        )

        # ── Health drain ──────────────────────────────────────────────────────
        if reward < HEALTH_DRAIN_THRESHOLD:
            self.health -= 1
            feedback = (
                f"⚠️  CRITICAL ERROR: {verdict_msg} "
                f"| Health remaining: {self.health}/{self.MAX_HEALTH}"
            )
        else:
            feedback = f"✅ Analysis accepted: {verdict_msg}"

        # ── Advance task pointer ──────────────────────────────────────────────
        done = False
        self.current_task_idx += 1

        if self.health <= 0:
            done     = True
            feedback = "❌ TERMINATED: Too many critical failures — health depleted."

        if self.current_task_idx >= len(self.scenarios):
            done = True
            if self.health > 0:
                feedback = (
                    f"🏆 SUCCESS: All {len(self.scenarios)} "
                    f"{self.active_level.upper()} scenarios completed."
                )

        # ── Next observation ──────────────────────────────────────────────────
        obs = (
            self.scenarios[self.current_task_idx]["data"]
            if not self._is_over()
            else None
        )

        return obs, reward, done, {
            "task_id":     task_id,
            "task_group":  current_task["level"],
            "is_correct":  reward >= R_PERFECT,
            "health":      self.health,
            "feedback":    feedback,
            "score":       round(self.score, 4),
            "task_scores": list(self.task_scores),
        }


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

# Singleton environment instance — shared across all requests.
_env      = PhishGuardEnv()
# BUG FIX: asyncio.Lock serialises /reset and /step so concurrent requests
# cannot race on _env's mutable state (current_task_idx, health, score, etc.).
_env_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("PhishGuard-Env %s starting on port 7860.", VERSION)
    yield
    log.info("PhishGuard-Env shutting down.")


app = FastAPI(
    title="PhishGuard-Env",
    description=(
        "OpenEnv-compliant SOC analyst simulation environment. "
        "Exposes /reset, /step, /state, and /health for LLM agent benchmarking."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Liveness probe ────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
async def health_probe() -> dict:
    """Liveness probe — HF Spaces and load-balancers call this endpoint."""
    return {"status": "ok", "env": "PhishGuard-Env", "version": VERSION}


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.post("/reset", tags=["Environment"])
async def reset(request: Optional[ResetRequest] = None) -> ResetResponse:
    """
    Reset the environment for a new episode at the chosen difficulty level.

    Body (optional): { "level": "easy" | "medium" | "hard" }
    If no body is provided, defaults to "easy".
    """
    level = (request.level if request else "easy").lower()
    if level not in LEVEL_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid level '{level}'. Must be one of: easy | medium | hard",
        )

    async with _env_lock:
        obs = _env.reset(level=level)
        first_scenario = _env.scenarios[_env.current_task_idx]
        active_level = _env.active_level

    return ResetResponse(
        observation=obs,
        task_id=first_scenario["id"],
        task_group=first_scenario["level"],
        level=active_level,
        # BUG FIX: was len(_env.scenarios) which could be stale;
        # now reads directly from LEVEL_MAP for the requested level.
        total_tasks=len(LEVEL_MAP[level]),
    )


# ── Step ──────────────────────────────────────────────────────────────────────
@app.post("/step", tags=["Environment"])
async def step(action: PhishAction) -> StepResponse:
    """
    Submit one triage action and receive the next observation + reward.

    Body: { "action": "MARK_SAFE | MOVE_TO_SPAM | QUARANTINE | BLOCK_DOMAIN",
            "reasoning": "optional" }
    """
    action_str = action.action.strip().upper()[:64]

    async with _env_lock:
        obs, reward, done, info = _env.step(action_str)

    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        task_id=info["task_id"],
        is_correct=info["is_correct"],
        info=info,
    )


# ── State ─────────────────────────────────────────────────────────────────────
@app.get("/state", tags=["Environment"])
async def state() -> dict:
    """Read-only snapshot of the current environment state."""
    async with _env_lock:
        overall = calculate_overall_score(_env.task_scores)
        return {
            "level":         _env.active_level,
            "health":        _env.health,
            "score":         round(_env.score, 4),
            "overall_score": overall,
            "task_index":    _env.current_task_idx,
            "total_tasks":   len(_env.scenarios),
            "task_scores":   list(_env.task_scores),
        }


# ── Tasks ─────────────────────────────────────────────────────────────────────
@app.get("/tasks", tags=["Environment"])
async def tasks() -> dict:
    """List all tasks with their IDs, difficulty, and correct actions."""
    return {
        "tasks": [
            {
                "task_id":    s["id"],
                "difficulty": s["level"],
                "type":       s["type"],
                "correct":    s["correct"],
            }
            for s in SCENARIOS
        ]
    }


# ── Grader ────────────────────────────────────────────────────────────────────
@app.post("/grader", tags=["Environment"])
async def grader(request: dict) -> dict:
    """
    Grade a triage action for a specific task without running a full episode.
    The OpenEnv validator calls this endpoint to verify graders are working.

    Body: { "task_id": "lv1", "action": "MOVE_TO_SPAM" }
    """
    task_id = request.get("task_id", "lv1")
    action  = request.get("action",  "QUARANTINE")

    scenario = next((s for s in SCENARIOS if s["id"] == task_id), None)
    if scenario is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Valid IDs: {[s['id'] for s in SCENARIOS]}",
        )

    reward, message = grade_action(action, scenario["correct"], scenario["type"])

    return {
        "task_id":    task_id,
        "action":     action,
        "reward":     reward,
        "is_correct": reward >= R_PERFECT,
        "message":    message,
    }


# ── Grade by Difficulty ───────────────────────────────────────────────────────
# Mirrors Focus-AI's GRADERS dict pattern — allows grading an entire
# difficulty level by passing metrics, just like Focus-AI's env.py uses
# GRADERS[difficulty](metrics) at episode end.
@app.post("/grade/{difficulty}", tags=["Grading"])
async def grade_difficulty(difficulty: str, metrics: dict) -> dict:
    """
    Grade a full episode for a specific difficulty level using the
    deterministic grader function.

    This mirrors Focus-AI's GRADERS[difficulty](metrics) pattern.

    Path param: difficulty = easy | medium | hard
    Body: metrics dict (e.g. {"total_tasks": 3, "correct_actions": 2, ...})
    """
    difficulty = difficulty.lower()
    if difficulty not in GRADERS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid difficulty '{difficulty}'. Must be one of: {list(GRADERS.keys())}",
        )

    score = GRADERS[difficulty](metrics)
    return {
        "difficulty": difficulty,
        "score":      score,
        "metrics":    metrics,
    }


# ── Aggregate Performance Grade ──────────────────────────────────────────────
@app.post("/grade/performance", tags=["Grading"])
async def grade_perf(metrics: dict) -> dict:
    """
    Cross-difficulty aggregate grader for leaderboard ranking.
    Mirrors Focus-AI's grade_performance() function.
    """
    score = grade_performance(metrics)
    return {
        "difficulty": "aggregate",
        "score":      score,
        "metrics":    metrics,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
