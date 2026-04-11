"""
grader.py – PhishGuard-Env SOC Triage Scoring Logic
====================================================

REWARD CONTRACT  →  OPEN INTERVAL (0.0, 1.0)
---------------------------------------------
All rewards are STRICTLY greater than 0 and STRICTLY less than 1.
The endpoints 0 and 1 are NEVER returned.  This is a hard invariant
enforced by the constant table below and by calculate_overall_score().

Why open-interval?
  • 1.0 saturates the leaderboard and implies a theoretically perfect agent.
  • 0.0 is indistinguishable from a missing data-point in an RL pipeline.
  • Every decision carries a non-zero gradient signal so training never dies.

REWARD TABLE
────────────────────────────────────────────────────────────────────────────
  Constant                 Value   Outcome / Rationale
  ─────────────────────────────────────────────────────────────────────────
  R_PERFECT                0.95    Exact match — near-ideal; headroom for 1.0
  R_MALWARE_QUARANTINE     0.75    MALWARE → QUARANTINE (textbook isolation)
  R_PHISH_BEC_QUARANTINE   0.60    PHISH/BEC → QUARANTINE (domain still live)
  R_SPAM_BLOCK             0.40    SPAM → BLOCK_DOMAIN (over-escalation)
  R_SPAM_QUARANTINE        0.35    SPAM → QUARANTINE (lighter over-escalation)
  R_WRONG_PROCEDURE        0.10    Wrong action, no direct security/ops harm
  R_DISRUPTION             0.05    SAFE email blocked — operational cost
  R_BREACH                 0.02    Threat allowed into inbox — catastrophic

HEALTH-DRAIN THRESHOLD
────────────────────────────────────────────────────────────────────────────
  HEALTH_DRAIN_THRESHOLD = 0.15
  reward < 0.15  →  agent loses one life.

PASS_THRESHOLD
────────────────────────────────────────────────────────────────────────────
  PASS_THRESHOLD = 0.50

LEVEL CONTEXT (from env.py)
────────────────────────────────────────────────────────────────────────────
  easy   → lv1 (SPAM),    lv2 (PHISH),   lv3 (SAFE)
  medium → lv4 (MALWARE), lv5 (SAFE),    lv6 (BEC),   lv7 (PHISH)
  hard   → lv8 (MALWARE), lv9 (PHISH),   lv10 (BEC)

VALID AGENT ACTIONS
────────────────────────────────────────────────────────────────────────────
  MARK_SAFE      – deliver to inbox
  MOVE_TO_SPAM   – bulk / unsolicited mail
  QUARANTINE     – hold for analyst review
  BLOCK_DOMAIN   – perimeter block

BUG FIX (v1.0.2 → v1.0.3)
────────────────────────────────────────────────────────────────────────────
  SPAM added to _THREAT_TYPES so MARK_SAFE on any threat drains health.
"""

from __future__ import annotations

from typing import Tuple

__all__ = [
    "R_PERFECT",
    "R_MALWARE_QUARANTINE",
    "R_PHISH_BEC_QUARANTINE",
    "R_SPAM_BLOCK",
    "R_SPAM_QUARANTINE",
    "R_WRONG_PROCEDURE",
    "R_DISRUPTION",
    "R_BREACH",
    "R_PARTIAL",
    "PASS_THRESHOLD",
    "HEALTH_DRAIN_THRESHOLD",
    "grade_action",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "calculate_overall_score",
]


# ══════════════════════════════════════════════════════════════════════════════
# REWARD CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

R_PERFECT              = 0.95
R_MALWARE_QUARANTINE   = 0.75
R_PHISH_BEC_QUARANTINE = 0.60
R_SPAM_BLOCK           = 0.40
R_SPAM_QUARANTINE      = 0.35
R_WRONG_PROCEDURE      = 0.10
R_DISRUPTION           = 0.05
R_BREACH               = 0.02

R_PARTIAL              = R_MALWARE_QUARANTINE
PASS_THRESHOLD         = 0.50
HEALTH_DRAIN_THRESHOLD = 0.15

_THREAT_TYPES  = frozenset({"PHISH", "BEC", "MALWARE", "SPAM"})
_BLOCKED_MOVES = frozenset({"BLOCK_DOMAIN", "QUARANTINE", "MOVE_TO_SPAM"})
_VALID_ACTIONS = frozenset({"MARK_SAFE", "MOVE_TO_SPAM", "QUARANTINE", "BLOCK_DOMAIN"})


# ══════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION
# ══════════════════════════════════════════════════════════════════════════════

def grade_action(
    agent_output: str,
    expected_output: str,
    email_type: str,
) -> Tuple[float, str]:
    """
    Grade one SOC triage decision and return a reward in (0.0, 1.0).

    Parameters
    ----------
    agent_output    : Raw action string from the LLM agent (case-insensitive).
    expected_output : Ground-truth correct action for this scenario.
    email_type      : Threat category — PHISH | BEC | MALWARE | SPAM | SAFE.

    Returns
    -------
    (reward, message)
        reward  : float strictly in (0.0, 1.0)
        message : short verdict string for logs and agent feedback.
    """
    agent_action    = agent_output.strip().upper()
    expected_action = expected_output.strip().upper()
    etype           = email_type.strip().upper()

    # Step 1: Reject unrecognised tokens
    if agent_action not in _VALID_ACTIONS:
        return (
            R_WRONG_PROCEDURE,
            f"INVALID_ACTION: '{agent_action}' is not a recognised triage action — "
            f"must be one of: {', '.join(sorted(_VALID_ACTIONS))}",
        )

    # Step 2: Perfect match
    if agent_action == expected_action:
        return R_PERFECT, "PERFECT_TRIAGE: Correct action taken"

    # Step 3: Security Breach
    if etype in _THREAT_TYPES and agent_action == "MARK_SAFE":
        return (
            R_BREACH,
            f"SECURITY_BREACH: {etype} threat delivered to inbox unimpeded — "
            "catastrophic failure; source remains active",
        )

    # Step 4: Business Disruption
    if etype == "SAFE" and agent_action in _BLOCKED_MOVES:
        return (
            R_DISRUPTION,
            "BUSINESS_DISRUPTION: Legitimate communication was incorrectly blocked — "
            "operational impact; sender trust degraded",
        )

    # Step 5: Partial credit

    # MALWARE → QUARANTINE
    if etype == "MALWARE" and agent_action == "QUARANTINE":
        return (
            R_MALWARE_QUARANTINE,
            "CAUTIOUS: Malware isolated via QUARANTINE — strong containment; "
            "no further propagation risk detected",
        )

    # PHISH/BEC → QUARANTINE
    if etype in {"PHISH", "BEC"} and agent_action == "QUARANTINE":
        return (
            R_PHISH_BEC_QUARANTINE,
            f"UNDER_RESPONSE: {etype} quarantined but source domain still active — "
            "prefer BLOCK_DOMAIN to prevent further phishing delivery attempts",
        )

    # SPAM → BLOCK_DOMAIN
    if etype == "SPAM" and agent_action == "BLOCK_DOMAIN":
        return (
            R_SPAM_BLOCK,
            "OVER_ESCALATION: BLOCK_DOMAIN is disproportionate for SPAM — "
            "prefer MOVE_TO_SPAM to preserve block-list resources",
        )

    # SPAM → QUARANTINE
    if etype == "SPAM" and agent_action == "QUARANTINE":
        return (
            R_SPAM_QUARANTINE,
            "OVER_ESCALATION: QUARANTINE wastes analyst capacity on SPAM — "
            "prefer MOVE_TO_SPAM for bulk unsolicited mail",
        )

    # Step 6: Catch-all wrong procedure
    return (
        R_WRONG_PROCEDURE,
        f"INCORRECT_PROCEDURE: '{agent_action}' does not match security policy "
        f"for {etype} email (expected: {expected_action}) — "
        "review triage guidelines",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE_OVERALL_SCORE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_overall_score(task_scores: list) -> float:
    """
    Compute the final benchmark score from a list of per-step rewards.
    Result is clamped to [R_BREACH, R_PERFECT].
    Empty list returns R_BREACH.
    """
    if not task_scores:
        return R_BREACH

    raw_avg = sum(task_scores) / len(task_scores)
    clamped = max(R_BREACH, min(R_PERFECT, raw_avg))
    return round(clamped, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SCORE SAFETY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_score(raw: float) -> float:
    """Map any float to the open interval (R_BREACH, R_PERFECT)."""
    raw = float(raw)
    raw = max(0.0, min(1.0, raw))
    result = R_BREACH + (R_PERFECT - R_BREACH) * raw
    return round(result, 6)


def _safe_ratio(num: float, den: float) -> float:
    """Safe division clamped to [0, 1]."""
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


# ══════════════════════════════════════════════════════════════════════════════
# OPENENV GRADERS
# Called by the OpenEnv validator — one function per difficulty level.
# Signature: grade_X(metrics: dict) -> float strictly in (0, 1)
# ══════════════════════════════════════════════════════════════════════════════

def grade_easy(metrics: dict) -> float:
    """
    Grader for easy tasks (lv1-lv3): SPAM, PHISH, SAFE.
    Scoring: 60% correct action + 40% threat identification accuracy.
    """
    total   = max(1, metrics.get("total_tasks", metrics.get("total", 3)))
    correct = metrics.get("correct_actions", metrics.get("is_correct", 0))
    on_time = metrics.get("on_time", metrics.get("completed", correct))

    if isinstance(correct, bool):
        correct = int(correct)
    if isinstance(on_time, bool):
        on_time = int(on_time)

    raw = (
        0.60 * _safe_ratio(correct, total)
        + 0.40 * _safe_ratio(on_time, total)
    )
    return _safe_score(raw)


def grade_medium(metrics: dict) -> float:
    """
    Grader for medium tasks (lv4-lv7): MALWARE, SAFE HR, BEC, PHISH.
    Scoring: 40% correct + 35% on-time detection + 25% escalation quality.
    """
    total    = max(1, metrics.get("total_tasks", metrics.get("total", 4)))
    correct  = metrics.get("correct_actions", metrics.get("is_correct", 0))
    on_time  = metrics.get("on_time", metrics.get("completed", correct))
    steps    = max(1, metrics.get("total_steps", metrics.get("steps", 4)))
    good_esc = metrics.get("good_escalation", metrics.get("reward", correct))

    if isinstance(correct,  bool): correct  = int(correct)
    if isinstance(on_time,  bool): on_time  = int(on_time)
    if isinstance(good_esc, bool): good_esc = int(good_esc)

    raw = (
        0.40 * _safe_ratio(correct,  total)
        + 0.35 * _safe_ratio(on_time,  total)
        + 0.25 * _safe_ratio(good_esc, steps)
    )
    return _safe_score(raw)


def grade_hard(metrics: dict) -> float:
    """
    Grader for hard tasks (lv8-lv10): MALWARE macro, QR phishing, BEC domain.
    Scoring: 35% correct + 30% threat accuracy + 20% escalation + 15% priority.
    """
    total    = max(1, metrics.get("total_tasks", metrics.get("total", 3)))
    correct  = metrics.get("correct_actions", metrics.get("is_correct", 0))
    on_time  = metrics.get("on_time", metrics.get("completed", correct))
    steps    = max(1, metrics.get("total_steps", metrics.get("steps", 3)))
    good_esc = metrics.get("good_escalation", metrics.get("reward", correct))
    hi_pri   = metrics.get("high_priority_correct", correct)

    if isinstance(correct,  bool): correct  = int(correct)
    if isinstance(on_time,  bool): on_time  = int(on_time)
    if isinstance(good_esc, bool): good_esc = int(good_esc)
    if isinstance(hi_pri,   bool): hi_pri   = int(hi_pri)

    raw = (
        0.35 * _safe_ratio(correct,  total)
        + 0.30 * _safe_ratio(on_time,  total)
        + 0.20 * _safe_ratio(good_esc, steps)
        + 0.15 * _safe_ratio(hi_pri,   max(1, correct))
    )
    return _safe_score(raw)