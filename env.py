"""
env.py – PhishGuard-Env  |  FastAPI Environment Server
=======================================================

ARCHITECTURE ROLE
-----------------
This file IS the environment. It runs as a persistent FastAPI server on
Hugging Face Spaces (port 7860). The inference agent (inference.py) is a
separate process that interacts with it exclusively through HTTP.

Endpoints
---------
  POST /reset               → reset for a chosen difficulty level
  POST /step                → submit one triage action
  GET  /state               → read-only snapshot of health, score, metrics
  GET  /health              → liveness probe
  GET  /tasks               → list all task IDs and correct actions
  POST /grader              → grade a single action without a full episode
  POST /grade/{difficulty}  → grade a full metrics dict for a difficulty
  POST /grade/performance   → aggregate cross-level grader

DIFFICULTY → TASK MAPPING
--------------------------
  easy   → lv1 (SPAM), lv2 (PHISH), lv3 (SAFE)              — 3 scenarios
  medium → lv4 (MALWARE), lv5 (SAFE), lv6 (BEC), lv7 (PHISH) — 4 scenarios
  hard   → lv8 (MALWARE), lv9 (PHISH), lv10 (BEC)            — 3 scenarios

Scenario order is fixed via seed=42 for reproducibility across runs.

Metrics tracked per episode (keys required by GRADERS)
-------------------------------------------------------
  total_tasks      : number of scenarios in this difficulty
  completed_tasks  : steps where a graded action was taken
  perfect_tasks    : steps where reward >= R_PERFECT
  on_time          : steps where reward >= HEALTH_DRAIN_THRESHOLD
  breach_count     : steps where reward == R_BREACH
  disruption_count : steps where reward == R_DISRUPTION
  total_steps      : total /step calls

Target score ranges (calibrated grader weights)
-----------------------------------------------
  easy   → ~0.86   (max raw 0.87)
  medium → ~0.75   (max raw 0.76)
  hard   → ~0.56   (max raw 0.56)

Episode score
-------------
When done=True, info["score"] = GRADERS[active_difficulty](metrics).
This is the same function the OpenEnv validator checks — guaranteeing
consistency between what the validator sees and what we report.
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
        OpenEnv = object

# ── Grader imports ────────────────────────────────────────────────────────────
from grader import (
    R_BREACH,
    R_DISRUPTION,
    R_PERFECT,
    HEALTH_DRAIN_THRESHOLD,
    GRADERS,
    TASK_GRADERS,
    calculate_overall_score,
    grade_action,
    grade_performance,
)

VERSION = "1.1.0"

# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS  (lv1 → lv10)
# ═════════════════════════════════════════════════════════════════════════════

SCENARIOS: List[dict] = [
    # ── EASY (lv1–lv3) ───────────────────────────────────────────────────────
    {
        "id": "lv1", "level": "easy", "type": "SPAM",
        "data": {
            "sender": "win@lotto.net",
            "subject": "Claim $1M Prize – Congratulations!",
            "body": "You have been selected. Click now to claim your prize before it expires.",
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
        "id": "lv2", "level": "easy", "type": "PHISH",
        "data": {
            "sender": "support@googIe.com",
            "subject": "Urgent Security Alert – Verify Your Account",
            "body": "We detected suspicious activity on your account. Verify your identity immediately.",
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
        "id": "lv3", "level": "easy", "type": "SAFE",
        "data": {
            "sender": "boss@company.com",
            "subject": "Team Meeting Tomorrow at 9 AM",
            "body": "Hi team, please be ready for our weekly sync at 9 AM in the main boardroom.",
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
        "id": "lv4", "level": "medium", "type": "MALWARE",
        "data": {
            "sender": "it@internal-help.com",
            "subject": "Critical Security Patch – Action Required Immediately",
            "body": "A critical vulnerability has been discovered. Run fix.exe to patch your system now.",
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
        "id": "lv5", "level": "medium", "type": "SAFE",
        "data": {
            "sender": "hr@yourcompany.com",
            "subject": "Updated Holiday Schedule – Please Review",
            "body": "Hi all, the updated holiday list is now live on the HR portal. No action needed.",
            "links": ["https://portal.yourcompany.com/holidays"],
            "has_attachments": False,
            "spf_record": "pass",
            "dmarc_record": "pass",
            "urgency_level": "low",
            "confidence_hint": "SIEM: bulk-sent to all-staff distribution list, normal recurring pattern",
        },
        "correct": "MARK_SAFE",
    },
    {
        "id": "lv6", "level": "medium", "type": "BEC",
        "data": {
            "sender": "ceo@executive-mail.com",
            "subject": "URGENT – Wire Transfer Required Today",
            "body": "I'm in a meeting and cannot be reached by phone. Wire $5,000 to the account below immediately.",
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
        "id": "lv7", "level": "medium", "type": "PHISH",
        "data": {
            "sender": "notify@docs-share.net",
            "subject": "You Have a New Shared Document",
            "body": "Your colleague shared a file with you. Click the link below to view it in your browser.",
            "links": ["https://docs-verify.net/s/1x9az"],
            "has_attachments": False,
            "spf_record": "softfail",
            "dmarc_record": "fail",
            "urgency_level": "medium",
            "confidence_hint": "Gateway: destination link redirects through 2 URL shorteners before landing",
        },
        "correct": "BLOCK_DOMAIN",
    },

    # ── HARD (lv8–lv10) ──────────────────────────────────────────────────────
    {
        "id": "lv8", "level": "hard", "type": "MALWARE",
        "data": {
            "sender": "payroll@hr-dept.net",
            "subject": "Your Bank Details Form – Action Required",
            "body": "Please review and complete the attached payroll update form before Friday.",
            "links": [],
            "has_attachments": True,
            "spf_record": "fail",
            "dmarc_record": "fail",
            "urgency_level": "high",
            "confidence_hint": "AV: attachment is a macro-enabled Excel file (.xlsm), macros unsigned",
        },
        "correct": "QUARANTINE",
    },
    {
        "id": "lv9", "level": "hard", "type": "PHISH",
        "data": {
            "sender": "mfa@office365-auth.com",
            "subject": "Action Required: Update Your MFA Settings Now",
            "body": "Your multi-factor authentication settings must be updated. Scan the QR code in the attached image to stay compliant.",
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
        "id": "lv10", "level": "hard", "type": "BEC",
        "data": {
            "sender": "partner@trusted-firm.com",
            "subject": "Updated Project Specifications – Download Required by EOD",
            "body": "Please find the revised project specs at the link below. Deadline is tomorrow morning.",
            "links": ["https://trusted-partner.com/files/project_specs_final.zip"],
            "has_attachments": False,
            "spf_record": "pass",
            "dmarc_record": "pass",
            "urgency_level": "high",
            "confidence_hint": "Threat Intel: trusted-firm.com added to IOC feed 6 hours ago — possible domain compromise",
        },
        "correct": "BLOCK_DOMAIN",
    },
]

DIFFICULTY_MAP: dict[str, list[str]] = {
    "easy":   ["lv1", "lv2", "lv3"],
    "medium": ["lv4", "lv5", "lv6", "lv7"],
    "hard":   ["lv8", "lv9", "lv10"],
}

_SCENARIO_BY_ID: dict[str, dict] = {s["id"]: s for s in SCENARIOS}

# Fixed RNG seed — same scenario order every episode for reproducibility
_SHUFFLE_SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _empty_metrics(total_tasks: int = 0) -> dict:
    """Zeroed metrics dict with all keys expected by GRADERS."""
    return {
        "total_tasks":      total_tasks,
        "completed_tasks":  0,
        "perfect_tasks":    0,
        "on_time":          0,
        "breach_count":     0,
        "disruption_count": 0,
        "total_steps":      0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ═════════════════════════════════════════════════════════════════════════════

class PhishGuardEnv(OpenEnv):
    MAX_HEALTH: int = 3

    def __init__(self) -> None:
        self.scenarios: List[dict] = []
        self.current_task_idx: int = 0
        self.health: int = self.MAX_HEALTH
        self.score: float = 0.0
        self.task_scores: List[float] = []
        self.metrics: dict = _empty_metrics()
        self.active_difficulty: str = "easy"
        self._load_difficulty("easy")

    def _load_difficulty(self, difficulty: str) -> None:
        difficulty = difficulty.lower()
        if difficulty not in DIFFICULTY_MAP:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. Valid choices: easy | medium | hard"
            )
        ids    = DIFFICULTY_MAP[difficulty]
        subset = [dict(_SCENARIO_BY_ID[sid]) for sid in ids]
        # Fixed seed — same scenario order every run (reproducibility)
        random.Random(_SHUFFLE_SEED).shuffle(subset)

        self.active_difficulty    = difficulty
        self.scenarios            = subset
        self.current_task_idx     = 0
        self.health               = self.MAX_HEALTH
        self.score                = 0.0
        self.task_scores          = []
        self.metrics              = _empty_metrics(total_tasks=len(subset))

    def _is_over(self) -> bool:
        return self.health <= 0 or self.current_task_idx >= len(self.scenarios)

    def reset(self, difficulty: str = "easy") -> dict:
        self._load_difficulty(difficulty)
        if not self.scenarios:
            raise ValueError(f"No scenarios found for difficulty '{difficulty}'")
        first_task = self.scenarios[self.current_task_idx]
        log.info(
            "Episode reset | difficulty=%s | first=%s | total=%d",
            self.active_difficulty, first_task["id"], len(self.scenarios),
        )
        return first_task["data"]

    def step(self, action_str: str) -> tuple:
        """
        Advance by one triage decision.

        Returns (obs, reward, done, info).
        info["score"] is set (non-None) only when done=True, using
        GRADERS[active_difficulty](metrics) — the grader the validator checks.
        """
        # ── Guard ─────────────────────────────────────────────────────────────
        if self._is_over():
            self.metrics["total_steps"] += 1
            return None, R_BREACH, True, {
                "task_id":     None,
                "task_group":  None,
                "is_correct":  False,
                "health":      self.health,
                "feedback":    "Episode already ended. Call /reset to start a new one.",
                "score":       None,
                "metrics":     dict(self.metrics),
                "task_scores": list(self.task_scores),
            }

        current_task = self.scenarios[self.current_task_idx]
        task_id      = current_task["id"]

        # ── Grade ─────────────────────────────────────────────────────────────
        reward, verdict_msg = grade_action(
            action_str,
            current_task["correct"],
            current_task["type"],
        )

        self.score += reward
        self.task_scores.append(reward)

        # ── Update metrics ────────────────────────────────────────────────────
        self.metrics["total_steps"]     += 1
        self.metrics["completed_tasks"] += 1

        if reward >= R_PERFECT:
            self.metrics["perfect_tasks"] += 1

        if reward >= HEALTH_DRAIN_THRESHOLD:
            self.metrics["on_time"] += 1

        if reward == R_BREACH:
            self.metrics["breach_count"] += 1

        if reward == R_DISRUPTION:
            self.metrics["disruption_count"] += 1

        log.info(
            "Step | difficulty=%s | task=%s | action=%s | reward=%.4f | %s",
            self.active_difficulty, task_id,
            action_str.strip().upper(), reward, verdict_msg,
        )

        # ── Health drain ──────────────────────────────────────────────────────
        if reward < HEALTH_DRAIN_THRESHOLD:
            self.health -= 1
            feedback = (
                f"CRITICAL ERROR: {verdict_msg} "
                f"| Health remaining: {self.health}/{self.MAX_HEALTH}"
            )
        else:
            feedback = f"Analysis accepted: {verdict_msg}"

        # ── Advance pointer ───────────────────────────────────────────────────
        done = False
        self.current_task_idx += 1

        if self.health <= 0:
            done     = True
            feedback = "TERMINATED: Too many critical failures — health depleted."

        if self.current_task_idx >= len(self.scenarios):
            done = True
            if self.health > 0:
                feedback = (
                    f"SUCCESS: All {len(self.scenarios)} "
                    f"{self.active_difficulty.upper()} scenarios completed."
                )

        obs = (
            self.scenarios[self.current_task_idx]["data"]
            if not self._is_over()
            else None
        )

        # ── Episode score via GRADERS ─────────────────────────────────────────
        episode_score: Optional[float] = None
        if done:
            episode_score = GRADERS[self.active_difficulty](self.metrics)
            log.info(
                "Episode done | difficulty=%s | score=%.6f | metrics=%s",
                self.active_difficulty, episode_score, self.metrics,
            )

        return obs, reward, done, {
            "task_id":     task_id,
            "task_group":  current_task["level"],
            "is_correct":  reward >= R_PERFECT,
            "health":      self.health,
            "feedback":    feedback,
            "score":       episode_score,
            "metrics":     dict(self.metrics),
            "task_scores": list(self.task_scores),
        }


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

_env      = PhishGuardEnv()
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


@app.get("/health", tags=["Meta"])
async def health_probe() -> dict:
    return {"status": "ok", "env": "PhishGuard-Env", "version": VERSION}


@app.post("/reset", tags=["Environment"])
async def reset(request: Optional[ResetRequest] = None) -> ResetResponse:
    """Reset for a new episode. Body (optional): { \"difficulty\": \"easy\"|\"medium\"|\"hard\" }"""
    difficulty = (request.difficulty if request else "easy").lower()
    if difficulty not in DIFFICULTY_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid difficulty '{difficulty}'. Must be one of: easy | medium | hard",
        )
    async with _env_lock:
        obs               = _env.reset(difficulty=difficulty)
        first_scenario    = _env.scenarios[_env.current_task_idx]
        active_difficulty = _env.active_difficulty

    return ResetResponse(
        observation=obs,
        task_id=first_scenario["id"],
        task_group=first_scenario["level"],
        difficulty=active_difficulty,
        total_tasks=len(DIFFICULTY_MAP[difficulty]),
    )


@app.post("/step", tags=["Environment"])
async def step(action: PhishAction) -> StepResponse:
    """Submit one triage action."""
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


@app.get("/state", tags=["Environment"])
async def state() -> dict:
    """Read-only snapshot. Does not advance the simulation."""
    async with _env_lock:
        episode_score  = GRADERS[_env.active_difficulty](_env.metrics)
        rolling_score  = calculate_overall_score(_env.task_scores)
        return {
            "difficulty":    _env.active_difficulty,
            "health":        _env.health,
            "score":         round(_env.score, 4),
            "overall_score": episode_score,   # GRADERS-based — what validator checks
            "rolling_score": rolling_score,   # calculate_overall_score per-step avg
            "task_index":    _env.current_task_idx,
            "total_tasks":   len(_env.scenarios),
            "task_scores":   list(_env.task_scores),
            "metrics":       dict(_env.metrics),
        }


@app.get("/tasks", tags=["Environment"])
async def tasks() -> dict:
    return {
        "tasks": [
            {"task_id": s["id"], "difficulty": s["level"],
             "type": s["type"], "correct": s["correct"]}
            for s in SCENARIOS
        ]
    }


@app.post("/grader", tags=["Grading"])
async def grader_endpoint(request: dict) -> dict:
    """
    Grade a single action for a task without a full episode.
    Body: { "task_id": "lv1", "action": "MOVE_TO_SPAM" }
    """
    task_id  = request.get("task_id", "lv1")
    action   = request.get("action",  "QUARANTINE")
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


@app.post("/grade/{difficulty}", tags=["Grading"])
async def grade_difficulty_endpoint(difficulty: str, metrics: dict) -> dict:
    """
    Grade a full episode metrics dict for a difficulty level.
    Mirrors FocusAI's GRADERS[difficulty](metrics) pattern.
    """
    difficulty = difficulty.lower()
    if difficulty not in GRADERS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid difficulty '{difficulty}'. Must be one of: {list(GRADERS.keys())}",
        )
    return {"difficulty": difficulty, "score": GRADERS[difficulty](metrics), "metrics": metrics}


@app.post("/grade/performance", tags=["Grading"])
async def grade_perf(metrics: dict) -> dict:
    """Cross-difficulty aggregate grader. Mirrors FocusAI's grade_performance()."""
    return {"difficulty": "aggregate", "score": grade_performance(metrics), "metrics": metrics}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False, log_level="info")
