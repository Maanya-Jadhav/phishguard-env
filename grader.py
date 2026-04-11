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

  R_PARTIAL is exported as an alias for R_MALWARE_QUARANTINE (0.75) so
  inference.py and env.py can import a single mid-range signal constant
  without hard-coding a numeric literal.

HEALTH-DRAIN THRESHOLD
────────────────────────────────────────────────────────────────────────────
  HEALTH_DRAIN_THRESHOLD = 0.15

  reward < 0.15  →  agent loses one life.  Covers:
    R_BREACH (0.02), R_DISRUPTION (0.05), R_WRONG_PROCEDURE (0.10)

  Cautious / partial-credit scores (≥ 0.35) NEVER drain health.

PASS_THRESHOLD
────────────────────────────────────────────────────────────────────────────
  PASS_THRESHOLD = 0.50

  The minimum overall_score an agent must achieve across a full run to be
  considered a passing benchmark result.  Imported by inference.py so the
  bar is defined in exactly one place.

LEVEL CONTEXT (from env.py)
────────────────────────────────────────────────────────────────────────────
  easy   → lv1 (SPAM),    lv2 (PHISH),   lv3 (SAFE)
  medium → lv4 (MALWARE), lv5 (SAFE),    lv6 (BEC),   lv7 (PHISH)
  hard   → lv8 (MALWARE), lv9 (PHISH),   lv10 (BEC)

  email_type values in play: SPAM | PHISH | BEC | MALWARE | SAFE

VALID AGENT ACTIONS
────────────────────────────────────────────────────────────────────────────
  MARK_SAFE      – deliver to inbox          (use ONLY for confirmed-safe)
  MOVE_TO_SPAM   – bulk / unsolicited mail   (no active threat)
  QUARANTINE     – hold for analyst review
  BLOCK_DOMAIN   – perimeter block           (confirmed phishing / BEC source)

DECISION TREE  (grade_action)
────────────────────────────────────────────────────────────────────────────
  1. Unrecognised action token               → R_WRONG_PROCEDURE
  2. action == correct                       → R_PERFECT
  3. Threat type (incl. SPAM) + MARK_SAFE    → R_BREACH      (Security Breach)
  4. SAFE type + blocking move               → R_DISRUPTION  (Business Disruption)
  5. MALWARE → QUARANTINE                    → R_MALWARE_QUARANTINE
  6. PHISH/BEC → QUARANTINE                  → R_PHISH_BEC_QUARANTINE
  7. SPAM → BLOCK_DOMAIN                     → R_SPAM_BLOCK
  8. SPAM → QUARANTINE                       → R_SPAM_QUARANTINE
  9. Catch-all wrong procedure               → R_WRONG_PROCEDURE

BUG FIX (v1.0.2 → v1.0.3)
────────────────────────────────────────────────────────────────────────────
  _THREAT_TYPES previously excluded SPAM.  This meant lv1 SPAM + MARK_SAFE
  returned R_WRONG_PROCEDURE (0.10) instead of R_BREACH (0.02) — a
  security-critical email type was not penalised as a breach.
  Fix: SPAM added to _THREAT_TYPES so MARK_SAFE on any threat drains health.
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
    "calculate_overall_score",
]


# ══════════════════════════════════════════════════════════════════════════════
# REWARD CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# All numeric reward values are defined ONCE here.
# env.py and inference.py import these — neither file hard-codes numbers.

R_PERFECT              = 0.95   # Exact triage match — near-ideal signal
R_MALWARE_QUARANTINE   = 0.75   # MALWARE isolated via QUARANTINE
R_PHISH_BEC_QUARANTINE = 0.60   # PHISH/BEC held but source domain still live
R_SPAM_BLOCK           = 0.40   # SPAM → BLOCK_DOMAIN (over-escalation)
R_SPAM_QUARANTINE      = 0.35   # SPAM → QUARANTINE   (lighter over-escalation)
R_WRONG_PROCEDURE      = 0.10   # Wrong action; no security / operational harm
R_DISRUPTION           = 0.05   # Business Disruption — SAFE email blocked
R_BREACH               = 0.02   # Security Breach — threat allowed through

# ── Convenience alias ─────────────────────────────────────────────────────────
# R_PARTIAL = mid-range cautious signal used by inference.py / reporting layers.
# Maps to R_MALWARE_QUARANTINE (0.75) — the highest partial-credit reward.
R_PARTIAL = R_MALWARE_QUARANTINE

# ── Benchmark pass threshold ──────────────────────────────────────────────────
# Imported by inference.py so the pass/fail bar is defined in one place.
PASS_THRESHOLD = 0.50

# ── Health-drain threshold ────────────────────────────────────────────────────
# env.py compares `reward < HEALTH_DRAIN_THRESHOLD` to decide life-loss.
# Must sit above R_WRONG_PROCEDURE (0.10) and below R_SPAM_QUARANTINE (0.35)
# so cautious over-escalations never drain health.
HEALTH_DRAIN_THRESHOLD = 0.15

# ── Internal lookup sets ──────────────────────────────────────────────────────
# BUG FIX: SPAM is now included in _THREAT_TYPES.
# Previously SPAM was omitted, so SPAM + MARK_SAFE returned R_WRONG_PROCEDURE
# (0.10) instead of the correct R_BREACH (0.02).
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
        reward  : float strictly in (0.0, 1.0) — NEVER 0, NEVER 1.
        message : short verdict string for logs and agent feedback.

    Examples (aligned with env.py scenarios)
    -----------------------------------------
    lv1  SPAM  / MOVE_TO_SPAM  + MOVE_TO_SPAM  → 0.95  PERFECT_TRIAGE
    lv1  SPAM  / MOVE_TO_SPAM  + MARK_SAFE     → 0.02  SECURITY_BREACH  ← BUG FIX
    lv1  SPAM  / MOVE_TO_SPAM  + QUARANTINE    → 0.35  OVER_ESCALATION
    lv2  PHISH / BLOCK_DOMAIN  + BLOCK_DOMAIN  → 0.95  PERFECT_TRIAGE
    lv2  PHISH / BLOCK_DOMAIN  + QUARANTINE    → 0.60  UNDER_RESPONSE
    lv2  PHISH / BLOCK_DOMAIN  + MARK_SAFE     → 0.02  SECURITY_BREACH
    lv3  SAFE  / MARK_SAFE     + MARK_SAFE     → 0.95  PERFECT_TRIAGE
    lv3  SAFE  / MARK_SAFE     + QUARANTINE    → 0.05  BUSINESS_DISRUPTION
    lv4  MALWARE/ QUARANTINE   + QUARANTINE    → 0.95  PERFECT_TRIAGE
    lv4  MALWARE/ QUARANTINE   + BLOCK_DOMAIN  → 0.10  INCORRECT_PROCEDURE
    lv6  BEC   / QUARANTINE    + MARK_SAFE     → 0.02  SECURITY_BREACH
    lv10 BEC   / BLOCK_DOMAIN  + QUARANTINE    → 0.60  UNDER_RESPONSE
    """
    agent_action    = agent_output.strip().upper()
    expected_action = expected_output.strip().upper()
    etype           = email_type.strip().upper()

    # ── Step 1: Reject unrecognised tokens ───────────────────────────────────
    # Treat unknown output as wrong procedure rather than raising so a single
    # malformed response does not crash the entire episode.
    if agent_action not in _VALID_ACTIONS:
        return (
            R_WRONG_PROCEDURE,
            f"INVALID_ACTION: '{agent_action}' is not a recognised triage action — "
            f"must be one of: {', '.join(sorted(_VALID_ACTIONS))}",
        )

    # ── Step 2: Perfect match ─────────────────────────────────────────────────
    if agent_action == expected_action:
        return R_PERFECT, "PERFECT_TRIAGE: Correct action taken"

    # ── Step 3: Security Breach — most severe outcome ────────────────────────
    # Any threat type (PHISH, BEC, MALWARE, SPAM) rubber-stamped as safe.
    # Covers: lv1 SPAM→MARK_SAFE, lv2 PHISH→MARK_SAFE, lv6 BEC→MARK_SAFE, etc.
    if etype in _THREAT_TYPES and agent_action == "MARK_SAFE":
        return (
            R_BREACH,
            f"SECURITY_BREACH: {etype} threat delivered to inbox unimpeded — "
            "catastrophic failure; source remains active",
        )

    # ── Step 4: Business Disruption — severe false-positive ──────────────────
    # A clean, legitimate email (SAFE) was blocked, quarantined, or spammed.
    # Covers: lv3 SAFE→QUARANTINE, lv5 SAFE→BLOCK_DOMAIN, etc.
    if etype == "SAFE" and agent_action in _BLOCKED_MOVES:
        return (
            R_DISRUPTION,
            "BUSINESS_DISRUPTION: Legitimate communication was incorrectly blocked — "
            "operational impact; sender trust degraded",
        )

    # ── Step 5: Partial credit — cautious but sub-optimal ────────────────────

    # MALWARE → QUARANTINE
    # Isolation is textbook containment; BLOCK_DOMAIN is preferred for confirmed
    # malware sources but QUARANTINE still prevents propagation.
    if etype == "MALWARE" and agent_action == "QUARANTINE":
        return (
            R_MALWARE_QUARANTINE,
            "CAUTIOUS: Malware isolated via QUARANTINE — strong containment; "
            "no further propagation risk detected",
        )

    # PHISH/BEC → QUARANTINE
    # Threat is held but the sending domain remains live and can re-deliver.
    # Preferred action is BLOCK_DOMAIN to sever the attack vector entirely.
    if etype in {"PHISH", "BEC"} and agent_action == "QUARANTINE":
        return (
            R_PHISH_BEC_QUARANTINE,
            f"UNDER_RESPONSE: {etype} quarantined but source domain still active — "
            "prefer BLOCK_DOMAIN to prevent further phishing delivery attempts",
        )

    # SPAM → BLOCK_DOMAIN
    # Disproportionate escalation; burns perimeter block-list capacity on a
    # low-severity bulk sender.  Prefer MOVE_TO_SPAM.
    if etype == "SPAM" and agent_action == "BLOCK_DOMAIN":
        return (
            R_SPAM_BLOCK,
            "OVER_ESCALATION: BLOCK_DOMAIN is disproportionate for SPAM — "
            "prefer MOVE_TO_SPAM to preserve block-list resources",
        )

    # SPAM → QUARANTINE
    # Lighter over-escalation; clogs the analyst review queue with bulk mail.
    if etype == "SPAM" and agent_action == "QUARANTINE":
        return (
            R_SPAM_QUARANTINE,
            "OVER_ESCALATION: QUARANTINE wastes analyst capacity on SPAM — "
            "prefer MOVE_TO_SPAM for bulk unsolicited mail",
        )

    # ── Step 6: General incorrect procedure (catch-all) ──────────────────────
    # Wrong action with no direct security breach or operational disruption.
    # e.g. MALWARE → BLOCK_DOMAIN when QUARANTINE is expected (lv4, lv8).
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

    The result is clamped to [R_BREACH, R_PERFECT] — maintaining the
    open-interval contract — so downstream consumers always receive a float
    strictly greater than 0 and strictly less than 1.

    Parameters
    ----------
    task_scores : list of floats, each in (0.0, 1.0).

    Returns
    -------
    float in [R_BREACH, R_PERFECT] — always a valid open-interval value.

    Edge cases
    ----------
    • Empty list    → R_BREACH (minimum non-zero signal; not zero)
    • Single step   → that step's reward, clamped to [R_BREACH, R_PERFECT]
    • All perfect   → R_PERFECT (0.95), never 1.0

    Level score examples (env.py alignment)
    ----------------------------------------
    easy   (3 tasks): [0.95, 0.95, 0.95]       → 0.95
    medium (4 tasks): [0.95, 0.02, 0.95, 0.60] → 0.63
    hard   (3 tasks): [0.02, 0.05, 0.95]        → 0.34
    """
    if not task_scores:
        return R_BREACH  # No tasks completed: minimum signal, not zero

    raw_avg = sum(task_scores) / len(task_scores)

    # Clamp strictly within the open-interval boundary constants.
    clamped = max(R_BREACH, min(R_PERFECT, raw_avg))
    return round(clamped, 4)
