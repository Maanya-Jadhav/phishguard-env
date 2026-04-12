"""
models.py – PhishGuard-Env Pydantic Models
==========================================

Typed request / response schemas used by env.py (FastAPI).

PhishAction   : Body schema for POST /step
StepResponse  : Response schema for POST /step  (OpenEnv grader compliance)
ResetResponse : Response schema for POST /reset

BUG FIX (v1.0.2 → v1.0.3)
────────────────────────────────────────────────────────────────────────────
  StepResponse.task_id was typed as `str` but the episode-already-over guard
  branch in env.py returns task_id=None.  Pydantic would raise a validation
  error on every post-episode /step call.
  Fix: task_id is now Optional[str] with a default of None.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PhishAction(BaseModel):
    """
    Action submitted by the agent to POST /step.

    Fields
    ------
    action    : One of MARK_SAFE | MOVE_TO_SPAM | QUARANTINE | BLOCK_DOMAIN
    reasoning : Optional one-sentence technical justification (for logging).
    """
    action: str = Field(
        max_length=64,
        description="Triage decision. Must be exactly one of: "
                    "MARK_SAFE | MOVE_TO_SPAM | QUARANTINE | BLOCK_DOMAIN"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="One-sentence technical justification for the triage decision",
    )


class ResetRequest(BaseModel):
    """Body schema for POST /reset."""
    difficulty: str = Field(
        default="easy",
        description="Difficulty level: easy | medium | hard",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS  (OpenEnv spec — all fields required by validator)
# ─────────────────────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    """
    Full response for POST /step.

    The OpenEnv validator inspects `task_id` and `is_correct` on every step
    to count how many distinct tasks have been graded.

    task_id is Optional[str] (not str) because the episode-already-over guard
    branch returns None — a non-optional field would cause a Pydantic
    ValidationError on every post-episode call.
    """
    observation: Optional[Dict[str, Any]] = Field(
        description="Next email dict, or null when the episode is done"
    )
    reward: float = Field(
        description="Step reward strictly in (0.0, 1.0)"
    )
    done: bool = Field(
        description="True when all scenarios are complete or health reaches 0"
    )
    task_id: Optional[str] = Field(          # BUG FIX: was `str`, must be Optional
        default=None,
        description="Scenario ID e.g. 'lv3' — required by OpenEnv validator"
    )
    is_correct: bool = Field(
        description="True when reward >= R_PERFECT (0.95)"
    )
    info: Dict[str, Any] = Field(
        description="Full grader info payload"
    )


class ResetResponse(BaseModel):
    """Response for POST /reset."""
    observation: Dict[str, Any] = Field(
        description="First email observation for this episode"
    )
    task_id: str = Field(
        description="ID of the first scenario in this episode"
    )
    task_group: str = Field(
        description="Difficulty level of the first scenario: easy | medium | hard"
    )
    difficulty: str = Field(
        description="Active difficulty level for this episode"
    )
    total_tasks: int = Field(
        description="Total number of scenarios in this level"
    )
