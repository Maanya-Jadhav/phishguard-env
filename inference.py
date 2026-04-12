"""
inference.py – PhishGuard-Env Baseline Inference Script
========================================================

Structured stdout logs (required by OpenEnv validator):
  [START] task=<level>
  [STEP]  task=<level> step=N reward=R is_correct=true|false
  [END]   task=<level> score=S steps=N

The episode score in [END] comes directly from info["score"] returned by
/step when done=True — which is GRADERS[level](metrics) from grader.py.
This guarantees the validator sees the same grader-based score that env.py
computes internally.
"""

from __future__ import annotations

import sys
import io
# Force UTF-8 output so emoji in env.py feedback strings don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os

import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file first so HF_TOKEN / OPENAI_API_KEY are available via os.getenv
load_dotenv()

from grader import PASS_THRESHOLD, GRADERS, grade_performance

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

MAX_STEPS_PER_LEVEL = 15
HTTP_MAX_RETRIES    = 3
HTTP_BACKOFF_BASE   = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────

if not API_KEY:
    print("[ERROR] No API key found. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a SOC (Security Operations Centre) Analyst triaging incoming emails.

    Analyse the email data provided and respond ONLY with valid JSON in this exact format:
    {"action": "<ACTION>", "reasoning": "<one sentence technical justification>"}

    Valid actions:
    - MARK_SAFE       : Deliver to inbox (confirmed legitimate email)
    - MOVE_TO_SPAM    : Bulk/unsolicited mail with no active threat
    - QUARANTINE      : Hold for analyst review (suspicious but unconfirmed)
    - BLOCK_DOMAIN    : Block sender domain at perimeter (confirmed phishing/malware source)

    Signal interpretation:
    - SPF fail + DMARC fail + urgency + links → likely PHISH or MALWARE → BLOCK_DOMAIN or QUARANTINE
    - Known sender, SPF pass, DMARC pass, no suspicious links → likely SAFE → MARK_SAFE
    - Bulk unsolicited with no malicious payload → SPAM → MOVE_TO_SPAM
    - Wire transfer / CEO fraud / financial urgency from unknown domain → BEC → QUARANTINE
    - Malware attachment confirmed by AV → QUARANTINE (isolate, do not deliver)
    - Confirmed phishing domain → BLOCK_DOMAIN (sever attack vector)

    confidence_hint field:
    - This is a contextual signal from your SIEM, mail gateway, or threat-intel feed.
    - It is intentionally noisy — treat it as one data-point, not ground truth.
    - If it directly contradicts other signals (SPF, DMARC, links), weigh all evidence.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

_session = requests.Session()


def _post(endpoint: str, payload: dict) -> dict:
    url = f"{ENV_BASE_URL}{endpoint}"
    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            resp = _session.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            wait = HTTP_BACKOFF_BASE ** attempt
            print(f"  [WARN] POST {endpoint} failed (attempt {attempt+1}): {exc} — retrying in {wait:.1f}s", flush=True)
            time.sleep(wait)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
            wait = HTTP_BACKOFF_BASE ** attempt
            print(f"  [WARN] POST {endpoint} server error (attempt {attempt+1}): {exc} — retrying in {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"POST {endpoint} failed after {HTTP_MAX_RETRIES} attempts: {last_exc}")


def _get(endpoint: str) -> dict:
    url = f"{ENV_BASE_URL}{endpoint}"
    last_exc: Optional[Exception] = None
    for attempt in range(HTTP_MAX_RETRIES):
        try:
            resp = _session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            wait = HTTP_BACKOFF_BASE ** attempt
            print(f"  [WARN] GET {endpoint} failed (attempt {attempt+1}): {exc} — retrying in {wait:.1f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"GET {endpoint} failed after {HTTP_MAX_RETRIES} attempts: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback triage  (used when LLM is unavailable / errors out)
# Covers all 10 PhishGuard scenarios deterministically.
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_triage(obs: Dict[str, Any]) -> tuple[str, str]:
    """
    Deterministic SOC triage using email header signals and confidence hints.
    Returns (action, reasoning) — same signature as the LLM path.

    Decision priority
    -----------------
    1. Threat-intel IOC hit on domain/link   → BLOCK_DOMAIN
    2. QR-code / credential-harvesting hint  → QUARANTINE
    3. AV-flagged or macro attachment        → QUARANTINE
    4. Suspicious attachment + auth failure  → QUARANTINE
    5. Spam content keywords                 → MOVE_TO_SPAM
    6. BEC / financial urgency keywords      → QUARANTINE
    7. URL redirect chain + auth failure     → BLOCK_DOMAIN
    8. Auth failure + links                  → BLOCK_DOMAIN
    9. Auth-OK, no threats                   → MARK_SAFE
    10. Default (uncertain)                  → QUARANTINE
    """
    spf  = obs.get("spf_record",  "").lower()
    dmarc = obs.get("dmarc_record", "").lower()
    urgency = obs.get("urgency_level", "").lower()
    links = obs.get("links", [])
    has_attach = obs.get("has_attachments", False)
    subject = obs.get("subject", "").lower()
    body    = obs.get("body",    "").lower()
    hint    = obs.get("confidence_hint", "").lower()

    auth_ok   = (spf == "pass" and dmarc == "pass")
    auth_fail = spf in ("fail", "softfail") or dmarc in ("fail", "none")

    # 1. Threat-intel IOC hit → block the domain
    if "ioc feed" in hint or "ioc" in hint:
        if links:
            return "BLOCK_DOMAIN", "Domain appears on threat-intel IOC feed — block at perimeter"
        return "QUARANTINE", "IOC hit with no links — quarantine for analyst review"

    # 2. QR-code / credential harvesting phishing
    if "credential-harvest" in hint or "credential harvest" in hint:
        return "QUARANTINE", "QR-code credential-harvesting page detected — quarantine attachment"

    # 3. AV-flagged attachment (PE binary, macros, unsigned)
    if has_attach and any(kw in hint for kw in ("av:", "macro", "pe binary", "unsigned")):
        return "QUARANTINE", "AV/macro-flagged attachment — isolate from delivery"

    # 4. Attachment with authentication failure
    if has_attach and auth_fail:
        return "QUARANTINE", "Suspicious attachment combined with SPF/DMARC failure"

    # 5. Spam: prize / lottery / mass-marketing content
    spam_kw = ("prize", "congratulations", "claim", "won", "lottery", "$1m", "million")
    if any(kw in subject + " " + body for kw in spam_kw) and urgency != "critical":
        return "MOVE_TO_SPAM", "Bulk prize/lottery spam — no active threat payload"

    # 6. BEC: financial urgency keywords in body
    bec_kw = ("wire", "transfer", "account below", "fund", "bank details")
    if any(kw in body for kw in bec_kw) and urgency in ("critical", "high"):
        return "QUARANTINE", "BEC wire-transfer / financial-fraud pattern detected"

    # 7. URL redirect chain with auth failure → confirmed phishing source
    if ("redirect" in hint or "url shortener" in hint) and auth_fail:
        return "BLOCK_DOMAIN", "Multi-hop URL redirect chain with auth failure — block domain"

    # 8. Auth failure + suspicious links → block
    if auth_fail and links:
        return "BLOCK_DOMAIN", "Domain authentication failure with outbound links — block"

    # 9. Clean authentication, no threat signals → safe
    if auth_ok and not has_attach:
        safe_negative = ("ioc" not in hint and "malware" not in hint
                         and "phish" not in hint and "credential" not in hint)
        if safe_negative:
            return "MARK_SAFE", "SPF/DMARC pass, no threat indicators — deliver to inbox"

    # 10. Default: hold for analyst review
    return "QUARANTINE", "Uncertain signals — quarantine as precaution"


# ─────────────────────────────────────────────────────────────────────────────
# LLM action selection  (rule-based fallback when LLM errors)
# ─────────────────────────────────────────────────────────────────────────────

def _choose_action(observation: Dict[str, Any]) -> tuple[str, str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(observation, indent=2)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        parsed    = json.loads(completion.choices[0].message.content)
        action    = parsed.get("action", "QUARANTINE").strip().upper()
        reasoning = parsed.get("reasoning", "")
        return action, reasoning
    except Exception as exc:
        print(f"  [WARN] LLM unavailable ({type(exc).__name__}) — using rule-based fallback", flush=True)
        return _rule_based_triage(observation)


# ─────────────────────────────────────────────────────────────────────────────
# Run one level
# ─────────────────────────────────────────────────────────────────────────────

def run_level(level: str) -> Dict[str, Any]:
    """
    Run a complete episode for the given difficulty level.

    The episode score is taken from info["score"] on the terminal step
    (done=True) — this is GRADERS[level](metrics) computed by env.py,
    the same value the OpenEnv validator uses.

    Falls back to /state's overall_score only if no terminal step score
    was captured (e.g. MAX_STEPS_PER_LEVEL reached without done=True).
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  LEVEL: {level.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Emit [START] ─────────────────────────────────────────────────────────
    print(f"[START] task={level}", flush=True)

    reset_resp  = _post("/reset", {"difficulty": level})
    obs         = reset_resp["observation"]
    total_tasks = reset_resp["total_tasks"]
    print(f"  Tasks in this level: {total_tasks}", flush=True)

    steps: List[dict] = []
    step_num            = 0
    done                = False
    step_resp: Dict[str, Any] = {}
    # episode_score is populated from info["score"] when done=True.
    # It comes from GRADERS[level](metrics) inside env.py.
    episode_score: Optional[float] = None
    # episode_metrics is populated from info["metrics"] when done=True.
    episode_metrics: Optional[dict] = None

    # Track current scenario ID: starts from reset, then updated after each step.
    current_scenario_id = reset_resp.get("task_id", "?")

    while not done and step_num < MAX_STEPS_PER_LEVEL:
        step_num += 1
        print(f"\n  Step {step_num} | scenario={current_scenario_id}", flush=True)

        action, reasoning = _choose_action(obs)
        print(f"    -> Action   : {action}", flush=True)
        print(f"    -> Reasoning: {reasoning[:80]}", flush=True)

        step_resp  = _post("/step", {"action": action, "reasoning": reasoning})
        reward     = step_resp["reward"]
        done       = step_resp["done"]
        is_correct = step_resp["is_correct"]
        info       = step_resp.get("info", {})

        # task_id in response is the scenario JUST processed — update for display
        graded_scenario_id = step_resp.get("task_id") or current_scenario_id

        print(f"    <- Graded   : scenario={graded_scenario_id}  correct={is_correct}  reward={reward:.4f}  done={done}", flush=True)
        feedback_raw  = info.get('feedback', '')
        feedback_safe = feedback_raw.encode('ascii', errors='replace').decode('ascii')[:120]
        print(f"    <- Feedback : {feedback_safe}", flush=True)

        # ── Emit [STEP] ───────────────────────────────────────────────────────
        print(f"[STEP] task={level} step={step_num} reward={reward:.4f} is_correct={is_correct}", flush=True)

        steps.append({
            "step":       step_num,
            "task_id":    graded_scenario_id,
            "action":     action,
            "reward":     reward,
            "is_correct": is_correct,
            "reasoning":  reasoning,
        })

        # Advance scenario ID tracker: next obs comes from the following scenario
        # (we don't know its ID until after the next step, so use graded+1 label)
        current_scenario_id = step_resp.get("task_id", "?")  # refreshed next iteration

        # Capture episode score and metrics from the terminal step.
        # info["score"] is non-None only when done=True (set by env.py via GRADERS).
        if done:
            episode_score   = info.get("score")
            episode_metrics = info.get("metrics")

        obs = step_resp.get("observation")
        if obs is None and not done:
            print("  [WARN] obs is None but done=False — breaking.", flush=True)
            break

    # ── Fallback: fetch from /state if episode ended without done=True ────────
    # (happens when MAX_STEPS_PER_LEVEL is reached before all tasks complete)
    if episode_score is None:
        state_resp    = _get("/state")
        episode_score = state_resp.get("overall_score", 0.01)
        episode_metrics = state_resp.get("metrics")

    print(f"\n  {'-'*50}", flush=True)
    print(f"  Level {level.upper()} complete | steps={step_num} | score={episode_score:.4f}", flush=True)

    # ── Emit [END] ────────────────────────────────────────────────────────────
    print(f"[END] task={level} score={episode_score:.4f} steps={step_num}", flush=True)

    return {
        "level":           level,
        "total_tasks":     total_tasks,
        "steps":           steps,
        "overall_score":   episode_score,
        "episode_metrics": episode_metrics or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PhishGuard-Env Baseline Inference")
    parser.add_argument(
        "--level",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run a single difficulty level instead of all three.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results.",
    )
    args = parser.parse_args()

    levels_to_run = [args.level] if args.level else ["easy", "medium", "hard"]
    output_path   = args.output or f"results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"

    try:
        health = _get("/health")
        print(f"  server status: {health.get('status', 'unknown')}", flush=True)
    except Exception as exc:
        print(f"[ERROR] Cannot reach environment server at {ENV_BASE_URL}: {exc}", flush=True)
        print("        Make sure `python env.py` is running in another terminal.", flush=True)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for level in levels_to_run:
        result = run_level(level)
        results.append(result)
        time.sleep(1)

    # ── Aggregate scoring ─────────────────────────────────────────────────────
    # Weighted by task count so all 10 tasks contribute equally
    # (easy=3, medium=4, hard=3).
    total_steps   = sum(len(r["steps"]) for r in results)
    total_correct = sum(s["is_correct"] for r in results for s in r["steps"])
    weighted_sum  = sum(r["overall_score"] * r["total_tasks"] for r in results)
    total_tasks   = sum(r["total_tasks"] for r in results)
    avg_score     = weighted_sum / total_tasks if total_tasks else 0.0

    # Cross-level grade_performance over combined metrics (mirrors FocusAI)
    if len(results) > 1:
        combined_metrics: Dict[str, Any] = {
            "total_tasks":      sum(r["episode_metrics"].get("total_tasks",      0) for r in results),
            "completed_tasks":  sum(r["episode_metrics"].get("completed_tasks",  0) for r in results),
            "perfect_tasks":    sum(r["episode_metrics"].get("perfect_tasks",    0) for r in results),
            "on_time":          sum(r["episode_metrics"].get("on_time",          0) for r in results),
            "breach_count":     sum(r["episode_metrics"].get("breach_count",     0) for r in results),
            "disruption_count": sum(r["episode_metrics"].get("disruption_count", 0) for r in results),
            "total_steps":      sum(r["episode_metrics"].get("total_steps",      0) for r in results),
        }
        performance_score = float(grade_performance(combined_metrics))
    else:
        performance_score = avg_score

    print(f"\n{'='*60}", flush=True)
    print(f"  BASELINE SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total steps      : {total_steps}", flush=True)
    print(f"  Correct steps    : {total_correct}", flush=True)
    print(f"  Weighted score   : {avg_score:.4f}  (pass threshold: {PASS_THRESHOLD})", flush=True)
    print(f"  Performance score: {performance_score:.4f}  (grade_performance)", flush=True)
    for r in results:
        print(f"  {r['level']:8s} score: {r['overall_score']:.4f}  ({r['total_tasks']} tasks)", flush=True)

    success = avg_score >= PASS_THRESHOLD

    # Build per-task summary for the results file
    all_tasks: List[Dict[str, Any]] = []
    for r in results:
        level_correct = sum(1 for s in r["steps"] if s["is_correct"])
        all_tasks.append({
            "task_id":    r["level"],
            "is_correct": level_correct > 0,
            "reward":     r["overall_score"],
            "level":      r["level"],
            "steps":      r["steps"],
        })

    run_summary = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "model":             MODEL_NAME,
        "env":               ENV_BASE_URL,
        "levels":            levels_to_run,
        "total_steps":       total_steps,
        "total_correct":     total_correct,
        "avg_score":         round(avg_score, 4),
        "performance_score": round(performance_score, 4),
        "pass_threshold":    PASS_THRESHOLD,
        "success":           success,
        "tasks":             all_tasks,
        "level_results":     results,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=2)
        print(f"\n  Results saved -> {output_path}", flush=True)
    except OSError as exc:
        print(f"\n  [WARN] Could not save results: {exc}", flush=True)


if __name__ == "__main__":
    main()
