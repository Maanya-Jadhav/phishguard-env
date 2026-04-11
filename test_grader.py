"""
test_grader.py – Unit tests for grader.py
==========================================

Run with:  pytest test_grader.py -v

These tests cover every branch of grade_action() and the edge-cases of
calculate_overall_score().  They also act as a regression guard — any
change to a reward constant will immediately fail the assertion that was
relying on it, forcing an intentional review.
"""

from __future__ import annotations

import pytest

from grader import (
    HEALTH_DRAIN_THRESHOLD,
    PASS_THRESHOLD,
    R_BREACH,
    R_DISRUPTION,
    R_MALWARE_QUARANTINE,
    R_PARTIAL,
    R_PERFECT,
    R_PHISH_BEC_QUARANTINE,
    R_SPAM_BLOCK,
    R_SPAM_QUARANTINE,
    R_WRONG_PROCEDURE,
    calculate_overall_score,
    grade_action,
)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANT SANITY CHECKS
# ═════════════════════════════════════════════════════════════════════════════

class TestRewardConstants:
    def test_all_rewards_in_open_interval(self):
        for r in [R_PERFECT, R_MALWARE_QUARANTINE, R_PHISH_BEC_QUARANTINE,
                  R_SPAM_BLOCK, R_SPAM_QUARANTINE, R_WRONG_PROCEDURE,
                  R_DISRUPTION, R_BREACH]:
            assert 0.0 < r < 1.0, f"Reward {r} violates open-interval contract"

    def test_reward_ordering(self):
        assert R_BREACH < R_DISRUPTION < R_WRONG_PROCEDURE
        assert R_WRONG_PROCEDURE < HEALTH_DRAIN_THRESHOLD
        assert HEALTH_DRAIN_THRESHOLD < R_SPAM_QUARANTINE
        assert R_SPAM_QUARANTINE < R_SPAM_BLOCK
        assert R_SPAM_BLOCK < R_PHISH_BEC_QUARANTINE
        assert R_PHISH_BEC_QUARANTINE < R_MALWARE_QUARANTINE
        assert R_MALWARE_QUARANTINE < R_PERFECT

    def test_partial_alias(self):
        assert R_PARTIAL == R_MALWARE_QUARANTINE

    def test_health_drain_covers_breach_disruption_wrong(self):
        assert R_BREACH < HEALTH_DRAIN_THRESHOLD
        assert R_DISRUPTION < HEALTH_DRAIN_THRESHOLD
        assert R_WRONG_PROCEDURE < HEALTH_DRAIN_THRESHOLD

    def test_cautious_scores_never_drain_health(self):
        assert R_SPAM_QUARANTINE >= HEALTH_DRAIN_THRESHOLD
        assert R_SPAM_BLOCK >= HEALTH_DRAIN_THRESHOLD
        assert R_PHISH_BEC_QUARANTINE >= HEALTH_DRAIN_THRESHOLD
        assert R_MALWARE_QUARANTINE >= HEALTH_DRAIN_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION — PERFECT MATCH
# ═════════════════════════════════════════════════════════════════════════════

class TestPerfectTriage:
    @pytest.mark.parametrize("action,etype", [
        ("MOVE_TO_SPAM",  "SPAM"),
        ("BLOCK_DOMAIN",  "PHISH"),
        ("MARK_SAFE",     "SAFE"),
        ("QUARANTINE",    "MALWARE"),
        ("QUARANTINE",    "BEC"),
        ("BLOCK_DOMAIN",  "BEC"),
        ("QUARANTINE",    "PHISH"),
    ])
    def test_perfect_match(self, action, etype):
        reward, msg = grade_action(action, action, etype)
        assert reward == R_PERFECT
        assert "PERFECT_TRIAGE" in msg

    def test_case_insensitive_action(self):
        reward, _ = grade_action("move_to_spam", "MOVE_TO_SPAM", "SPAM")
        assert reward == R_PERFECT

    def test_case_insensitive_type(self):
        reward, _ = grade_action("MARK_SAFE", "MARK_SAFE", "safe")
        assert reward == R_PERFECT


# ═════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION — SECURITY BREACH
# ═════════════════════════════════════════════════════════════════════════════

class TestSecurityBreach:
    @pytest.mark.parametrize("etype", ["PHISH", "BEC", "MALWARE", "SPAM"])
    def test_mark_safe_on_any_threat_is_breach(self, etype):
        """BUG FIX coverage: SPAM + MARK_SAFE must return R_BREACH (not R_WRONG_PROCEDURE)."""
        reward, msg = grade_action("MARK_SAFE", "BLOCK_DOMAIN", etype)
        assert reward == R_BREACH
        assert "SECURITY_BREACH" in msg

    def test_spam_mark_safe_is_breach_not_wrong_procedure(self):
        """Regression: the original code excluded SPAM from _THREAT_TYPES."""
        reward, _ = grade_action("MARK_SAFE", "MOVE_TO_SPAM", "SPAM")
        assert reward == R_BREACH, (
            "SPAM + MARK_SAFE must be R_BREACH — SPAM was excluded from "
            "_THREAT_TYPES in the v1.0.2 bug"
        )
        assert reward < HEALTH_DRAIN_THRESHOLD  # must drain health


# ═════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION — BUSINESS DISRUPTION
# ═════════════════════════════════════════════════════════════════════════════

class TestBusinessDisruption:
    @pytest.mark.parametrize("blocking_action", ["BLOCK_DOMAIN", "QUARANTINE", "MOVE_TO_SPAM"])
    def test_blocking_safe_email_is_disruption(self, blocking_action):
        reward, msg = grade_action(blocking_action, "MARK_SAFE", "SAFE")
        assert reward == R_DISRUPTION
        assert "BUSINESS_DISRUPTION" in msg


# ═════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION — PARTIAL CREDIT
# ═════════════════════════════════════════════════════════════════════════════

class TestPartialCredit:
    def test_malware_quarantine(self):
        reward, msg = grade_action("QUARANTINE", "BLOCK_DOMAIN", "MALWARE")
        assert reward == R_MALWARE_QUARANTINE
        assert "CAUTIOUS" in msg

    def test_phish_quarantine(self):
        reward, msg = grade_action("QUARANTINE", "BLOCK_DOMAIN", "PHISH")
        assert reward == R_PHISH_BEC_QUARANTINE
        assert "UNDER_RESPONSE" in msg

    def test_bec_quarantine(self):
        reward, msg = grade_action("QUARANTINE", "BLOCK_DOMAIN", "BEC")
        assert reward == R_PHISH_BEC_QUARANTINE

    def test_spam_block_domain(self):
        reward, msg = grade_action("BLOCK_DOMAIN", "MOVE_TO_SPAM", "SPAM")
        assert reward == R_SPAM_BLOCK
        assert "OVER_ESCALATION" in msg

    def test_spam_quarantine(self):
        reward, msg = grade_action("QUARANTINE", "MOVE_TO_SPAM", "SPAM")
        assert reward == R_SPAM_QUARANTINE
        assert "OVER_ESCALATION" in msg

    @pytest.mark.parametrize("action", ["QUARANTINE", "BLOCK_DOMAIN", "MOVE_TO_SPAM"])
    def test_partial_credits_never_drain_health(self, action):
        reward, _ = grade_action(action, "MOVE_TO_SPAM", "SPAM")
        assert reward >= HEALTH_DRAIN_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# GRADE_ACTION — WRONG PROCEDURE / INVALID
# ═════════════════════════════════════════════════════════════════════════════

class TestWrongProcedure:
    def test_malware_block_domain_is_wrong_procedure(self):
        """lv4 / lv8: expected QUARANTINE, agent picks BLOCK_DOMAIN."""
        reward, msg = grade_action("BLOCK_DOMAIN", "QUARANTINE", "MALWARE")
        assert reward == R_WRONG_PROCEDURE
        assert "INCORRECT_PROCEDURE" in msg

    def test_invalid_action_token(self):
        reward, msg = grade_action("DELETE", "QUARANTINE", "MALWARE")
        assert reward == R_WRONG_PROCEDURE
        assert "INVALID_ACTION" in msg

    def test_invalid_action_drains_health(self):
        reward, _ = grade_action("NONSENSE", "QUARANTINE", "PHISH")
        assert reward < HEALTH_DRAIN_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# CALCULATE_OVERALL_SCORE
# ═════════════════════════════════════════════════════════════════════════════

class TestCalculateOverallScore:
    def test_empty_list_returns_r_breach(self):
        assert calculate_overall_score([]) == R_BREACH

    def test_all_perfect_returns_r_perfect(self):
        assert calculate_overall_score([R_PERFECT, R_PERFECT, R_PERFECT]) == R_PERFECT

    def test_result_never_exceeds_r_perfect(self):
        assert calculate_overall_score([1.0, 1.0]) == R_PERFECT

    def test_result_never_below_r_breach(self):
        assert calculate_overall_score([0.0, 0.0]) == R_BREACH

    def test_easy_all_perfect(self):
        scores = [R_PERFECT] * 3
        assert calculate_overall_score(scores) == R_PERFECT

    def test_medium_mixed(self):
        scores = [R_PERFECT, R_BREACH, R_PERFECT, R_PHISH_BEC_QUARANTINE]
        result = calculate_overall_score(scores)
        expected = round((R_PERFECT + R_BREACH + R_PERFECT + R_PHISH_BEC_QUARANTINE) / 4, 4)
        assert result == expected

    def test_hard_mostly_bad(self):
        scores = [R_BREACH, R_DISRUPTION, R_PERFECT]
        result = calculate_overall_score(scores)
        assert result < PASS_THRESHOLD

    def test_returns_four_decimal_places(self):
        result = calculate_overall_score([R_PERFECT, R_BREACH])
        assert result == round(result, 4)

    def test_single_perfect_step(self):
        assert calculate_overall_score([R_PERFECT]) == R_PERFECT

    def test_single_breach_step(self):
        assert calculate_overall_score([R_BREACH]) == R_BREACH


# ═════════════════════════════════════════════════════════════════════════════
# EDGE CASES — INPUT NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

class TestInputNormalization:
    def test_whitespace_padded_action(self):
        """Actions with leading/trailing spaces should still match."""
        reward, msg = grade_action("  MARK_SAFE  ", "MARK_SAFE", "SAFE")
        assert reward == R_PERFECT
        assert "PERFECT_TRIAGE" in msg

    def test_mixed_case_email_type_phish(self):
        reward, _ = grade_action("MARK_SAFE", "BLOCK_DOMAIN", "Phish")
        assert reward == R_BREACH

    def test_mixed_case_email_type_bec(self):
        reward, _ = grade_action("QUARANTINE", "BLOCK_DOMAIN", "bEc")
        assert reward == R_PHISH_BEC_QUARANTINE

    def test_tab_in_action(self):
        reward, _ = grade_action("\tQUARANTINE\t", "QUARANTINE", "MALWARE")
        assert reward == R_PERFECT

    def test_empty_action_string(self):
        reward, msg = grade_action("", "QUARANTINE", "MALWARE")
        assert reward == R_WRONG_PROCEDURE
        assert "INVALID_ACTION" in msg

    def test_calculate_overall_score_single_breach(self):
        assert calculate_overall_score([R_BREACH]) == R_BREACH

    def test_calculate_overall_score_all_disruption(self):
        result = calculate_overall_score([R_DISRUPTION] * 5)
        assert result == R_DISRUPTION

