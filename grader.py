"""
grader.py – PhishGuard-Env SOC Triage Scoring Logic
====================================================

SCORE CONTRACT  (HIGHEST PRIORITY — mirrors FocusAI reward_and_tasks.py)
-------------------------------------------------------------------------
Every public grader returns a float STRICTLY inside the open interval (0, 1).

    safe_score(raw) = LOWER + (UPPER - LOWER) * clamp(raw, 0, 1)

    where LOWER = 0.01, UPPER = 0.99

This guarantees:
    raw = 0.0  →  0.01   (> 0, never equals 0)
    raw = 1.0  →  0.99   (< 1, never equals 1)
    raw = 0.5  →  0.50

VALIDATOR COMPLIANCE — "not enough tasks with graders"
-------------------------------------------------------
The OpenEnv validator requires at least 3 task IDs that each have a
registered grader function.  This is satisfied by the GRADERS dict:

    GRADERS["easy"]   = grade_easy
    GRADERS["medium"] = grade_medium
    GRADERS["hard"]   = grade_hard

TASK_GRADERS additionally maps every lv1–lv10 scenario ID to its level
grader for per-task lookups from env.py.

PER-STEP REWARD TABLE  (grade_action — used by /step endpoint)
──────────────────────────────────────────────────────────────────────────
  Constant                 Value   Outcome
  ─────────────────────────────────────────────────────────────────────
  R_PERFECT                0.95    Exact triage match
  R_MALWARE_QUARANTINE     0.75    MALWARE → QUARANTINE (good containment)
  R_PHISH_BEC_QUARANTINE   0.60    PHISH/BEC → QUARANTINE (domain still live)
  R_SPAM_BLOCK             0.40    SPAM → BLOCK_DOMAIN (over-escalation)
  R_SPAM_QUARANTINE        0.35    SPAM → QUARANTINE (lighter over-escalation)
  R_WRONG_PROCEDURE        0.10    Wrong; no direct breach or disruption
  R_DISRUPTION             0.05    SAFE email blocked — operational cost
  R_BREACH                 0.02    Threat allowed into inbox — catastrophic

HEALTH-DRAIN THRESHOLD
──────────────────────────────────────────────────────────────────────────
  reward < 0.15  →  agent loses one life.
  Cautious / partial-credit scores (≥ 0.35) NEVER drain health.

LEVEL → TASK MAPPING
──────────────────────────────────────────────────────────────────────────
  easy   → lv1 (SPAM), lv2 (PHISH), lv3 (SAFE)
  medium → lv4 (MALWARE), lv5 (SAFE), lv6 (BEC), lv7 (PHISH)
  hard   → lv8 (MALWARE), lv9 (PHISH), lv10 (BEC)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# SCORE SAFETY  (mirrors FocusAI safe_score exactly)
# ══════════════════════════════════════════════════════════════════════════════

_SCORE_LOWER = 0.01
_SCORE_UPPER = 0.99


def safe_score(raw: float) -> float:
    """
    Map any raw float to the open interval (0.01, 0.99).

    Never returns 0 or 1 — satisfies the open-interval contract required
    by the OpenEnv validator and the RL pipeline.

        safe_score(0.0) = 0.01
        safe_score(1.0) = 0.99
        safe_score(0.5) = 0.50
    """
    raw = float(raw)
    raw = max(0.0, min(1.0, raw))
    result = _SCORE_LOWER + (_SCORE_UPPER - _SCORE_LOWER) * raw
    result = round(result, 6)
    assert 0.0 < result < 1.0, (
        f"safe_score VIOLATION: raw={raw!r} produced result={result!r} "
        f"which is not strictly inside (0, 1)"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PER-STEP REWARD CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

R_PERFECT              = 0.95
R_MALWARE_QUARANTINE   = 0.75
R_PHISH_BEC_QUARANTINE = 0.60
R_SPAM_BLOCK           = 0.40
R_SPAM_QUARANTINE      = 0.35
R_WRONG_PROCEDURE      = 0.10
R_DISRUPTION           = 0.05
R_BREACH               = 0.02

# Convenience alias (mid-range cautious signal)
R_PARTIAL = R_MALWARE_QUARANTINE

# Minimum weighted average for a run to be considered passing
PASS_THRESHOLD = 0.50

# env.py: `reward < HEALTH_DRAIN_THRESHOLD` → lose one life
HEALTH_DRAIN_THRESHOLD = 0.15

# Internal lookup sets
_THREAT_TYPES  = frozenset({"PHISH", "BEC", "MALWARE", "SPAM"})
_BLOCKED_MOVES = frozenset({"BLOCK_DOMAIN", "QUARANTINE", "MOVE_TO_SPAM"})
_VALID_ACTIONS = frozenset({"MARK_SAFE", "MOVE_TO_SPAM", "QUARANTINE", "BLOCK_DOMAIN"})


# ══════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION  (per-step reward, called on every /step)
# ══════════════════════════════════════════════════════════════════════════════

def grade_action(
    agent_output: str,
    expected_output: str,
    email_type: str,
) -> Tuple[float, str]:
    """
    Grade one SOC triage decision and return (reward, verdict_message).

    reward is a raw float in (0.02, 0.95).
    Callers may wrap with safe_score() for the hard (0.01, 0.99) contract.

    Decision tree
    -------------
    1. Unrecognised action          → R_WRONG_PROCEDURE
    2. action == correct            → R_PERFECT
    3. Any threat + MARK_SAFE       → R_BREACH
    4. SAFE + blocking action       → R_DISRUPTION
    5. MALWARE → QUARANTINE         → R_MALWARE_QUARANTINE
    6. PHISH/BEC → QUARANTINE       → R_PHISH_BEC_QUARANTINE
    7. SPAM → BLOCK_DOMAIN          → R_SPAM_BLOCK
    8. SPAM → QUARANTINE            → R_SPAM_QUARANTINE
    9. catch-all                    → R_WRONG_PROCEDURE
    """
    agent_action    = agent_output.strip().upper()
    expected_action = expected_output.strip().upper()
    etype           = email_type.strip().upper()

    if agent_action not in _VALID_ACTIONS:
        return (
            R_WRONG_PROCEDURE,
            f"INVALID_ACTION: '{agent_action}' is not a recognised triage action — "
            f"must be one of: {', '.join(sorted(_VALID_ACTIONS))}",
        )

    if agent_action == expected_action:
        return R_PERFECT, "PERFECT_TRIAGE: Correct action taken"

    if etype in _THREAT_TYPES and agent_action == "MARK_SAFE":
        return (
            R_BREACH,
            f"SECURITY_BREACH: {etype} threat delivered to inbox unimpeded",
        )

    if etype == "SAFE" and agent_action in _BLOCKED_MOVES:
        return (
            R_DISRUPTION,
            "BUSINESS_DISRUPTION: Legitimate communication was incorrectly blocked",
        )

    if etype == "MALWARE" and agent_action == "QUARANTINE":
        return (
            R_MALWARE_QUARANTINE,
            "CAUTIOUS: Malware isolated via QUARANTINE — strong containment",
        )

    if etype in {"PHISH", "BEC"} and agent_action == "QUARANTINE":
        return (
            R_PHISH_BEC_QUARANTINE,
            f"UNDER_RESPONSE: {etype} quarantined but source domain still active",
        )

    if etype == "SPAM" and agent_action == "BLOCK_DOMAIN":
        return (
            R_SPAM_BLOCK,
            "OVER_ESCALATION: BLOCK_DOMAIN is disproportionate for SPAM",
        )

    if etype == "SPAM" and agent_action == "QUARANTINE":
        return (
            R_SPAM_QUARANTINE,
            "OVER_ESCALATION: QUARANTINE wastes analyst capacity on SPAM",
        )

    return (
        R_WRONG_PROCEDURE,
        f"INCORRECT_PROCEDURE: '{agent_action}' does not match policy "
        f"for {etype} (expected: {expected_action})",
    )


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE GRADERS  (end-of-episode — required by OpenEnv validator)
#
# Each grader accepts a `metrics` dict built by env.py and returns safe_score.
#
# metrics keys
# ────────────
#   total_tasks      : int  — scenarios in this episode
#   completed_tasks  : int  — steps where any action was graded
#   perfect_tasks    : int  — steps where reward >= R_PERFECT
#   on_time          : int  — steps completed without health drain
#   breach_count     : int  — SECURITY_BREACH outcomes (threat + MARK_SAFE)
#   disruption_count : int  — BUSINESS_DISRUPTION outcomes (SAFE + blocked)
#   total_steps      : int  — total /step calls
# ══════════════════════════════════════════════════════════════════════════════

def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator clamped to [0, 1]. 0 if denominator ≤ 0."""
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def grade_easy(metrics: dict) -> float:
    """
    Easy episode grader  (lv1–lv3: SPAM, PHISH, SAFE).

    Weights
    -------
      60 % — perfect triage rate  (exact action matches / total tasks)
      40 % — completion rate      (any graded step / total tasks)

    Penalty: −0.15 × breach_rate (THREAT + MARK_SAFE outcomes)
    """
    total     = max(1, metrics.get("total_tasks",    1))
    perfect   = metrics.get("perfect_tasks",   0)
    completed = metrics.get("completed_tasks", 0)
    breaches  = metrics.get("breach_count",    0)

    raw = (
        0.60 * _safe_ratio(perfect,   total)
        + 0.40 * _safe_ratio(completed, total)
        - 0.15 * min(1.0, breaches / max(1, total))
    )
    return safe_score(max(0.0, raw))


def grade_medium(metrics: dict) -> float:
    """
    Medium episode grader  (lv4–lv7: MALWARE, SAFE, BEC, PHISH).

    Weights
    -------
      45 % — perfect triage rate
      35 % — on-time rate  (health not drained by step)
      20 % — completion rate

    Penalty: −0.10 × breach_rate, −0.05 × disruption_rate
    """
    total       = max(1, metrics.get("total_tasks",      1))
    perfect     = metrics.get("perfect_tasks",     0)
    on_time     = metrics.get("on_time",            0)
    completed   = metrics.get("completed_tasks",    0)
    breaches    = metrics.get("breach_count",       0)
    disruptions = metrics.get("disruption_count",   0)

    raw = (
        0.45 * _safe_ratio(perfect,   total)
        + 0.35 * _safe_ratio(on_time,   total)
        + 0.20 * _safe_ratio(completed, total)
        - 0.10 * min(1.0, breaches    / max(1, total))
        - 0.05 * min(1.0, disruptions / max(1, total))
    )
    return safe_score(max(0.0, raw))


def grade_hard(metrics: dict) -> float:
    """
    Hard episode grader  (lv8–lv10: adversarial MALWARE, PHISH, BEC).

    Weights
    -------
      40 % — perfect triage rate
      30 % — on-time rate
      20 % — completion rate
      10 % — zero-breach bonus  (1.0 if no breaches; else 0.0)

    Penalty: −0.12 × breach_rate, −0.06 × disruption_rate
    """
    total       = max(1, metrics.get("total_tasks",      1))
    perfect     = metrics.get("perfect_tasks",     0)
    on_time     = metrics.get("on_time",            0)
    completed   = metrics.get("completed_tasks",    0)
    breaches    = metrics.get("breach_count",       0)
    disruptions = metrics.get("disruption_count",   0)

    zero_breach_bonus = 1.0 if breaches == 0 else 0.0

    raw = (
        0.40 * _safe_ratio(perfect,   total)
        + 0.30 * _safe_ratio(on_time,   total)
        + 0.20 * _safe_ratio(completed, total)
        + 0.10 * zero_breach_bonus
        - 0.12 * min(1.0, breaches    / max(1, total))
        - 0.06 * min(1.0, disruptions / max(1, total))
    )
    return safe_score(max(0.0, raw))


def grade_performance(metrics: dict) -> float:
    """
    Aggregate grader for cross-level scoring in inference.py.

    Produces one float for the full run (all levels combined).
    Mirrors FocusAI's grade_performance signature exactly.

    Weights
    -------
      40 % — perfect triage rate
      30 % — on-time rate
      20 % — completion rate
      10 % — zero-breach bonus
    """
    total     = max(1, metrics.get("total_tasks",    1))
    perfect   = metrics.get("perfect_tasks",   0)
    on_time   = metrics.get("on_time",          0)
    completed = metrics.get("completed_tasks",  0)
    breaches  = metrics.get("breach_count",     0)

    zero_breach_bonus = 1.0 if breaches == 0 else 0.0

    raw = (
        0.40 * _safe_ratio(perfect,   total)
        + 0.30 * _safe_ratio(on_time,   total)
        + 0.20 * _safe_ratio(completed, total)
        + 0.10 * zero_breach_bonus
    )
    return safe_score(max(0.0, raw))


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY MAPS  (required by OpenEnv validator — ≥ 3 entries needed)
# ══════════════════════════════════════════════════════════════════════════════

# Primary registry — level name → episode grader.
# The validator scans GRADERS to confirm ≥ 3 tasks have graders.
GRADERS: Dict[str, Callable[[dict], float]] = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

# Per-scenario registry — each lv1–lv10 ID mapped to its level grader.
# env.py uses this for per-task score lookups in the /step response.
TASK_GRADERS: Dict[str, Callable[[dict], float]] = {
    "lv1":  grade_easy,
    "lv2":  grade_easy,
    "lv3":  grade_easy,
    "lv4":  grade_medium,
    "lv5":  grade_medium,
    "lv6":  grade_medium,
    "lv7":  grade_medium,
    "lv8":  grade_hard,
    "lv9":  grade_hard,
    "lv10": grade_hard,
}


# ══════════════════════════════════════════════════════════════════════════════
# CALCULATE_OVERALL_SCORE  (backward-compat helper for /state endpoint)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_overall_score(task_scores: list) -> float:
    """
    Average a list of per-step grade_action() rewards and return safe_score.

    Parameters
    ----------
    task_scores : list of raw floats from grade_action() calls.

    Returns
    -------
    float in (0.01, 0.99) — open-interval contract guaranteed.

    Edge cases
    ----------
    • Empty list  → safe_score(0) = 0.01
    • All perfect → safe_score(~0.968) ≈ 0.968
    """
    if not task_scores:
        return safe_score(0.0)

    raw_avg = sum(task_scores) / len(task_scores)
    # Normalise from the per-step range (R_BREACH … R_PERFECT) → (0, 1)
    normalised = (raw_avg - R_BREACH) / (R_PERFECT - R_BREACH)
    return safe_score(max(0.0, min(1.0, normalised)))
