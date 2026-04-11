"""
inference.py – PhishGuard-Env Baseline Inference Script
========================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from grader import PASS_THRESHOLD

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
    - Example hints: "AV: attachment flagged", "Gateway: domain registered 3 days ago",
      "Threat Intel: domain added to IOC feed", "SIEM: sender in address book 2+ years".
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
# LLM action selection
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
        print(f"  [WARN] LLM error: {exc} — defaulting to QUARANTINE", flush=True)
        return "QUARANTINE", "Parse error — safe fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Run one level
# ─────────────────────────────────────────────────────────────────────────────

def run_level(level: str) -> Dict[str, Any]:
    print(f"\n{'═'*60}", flush=True)
    print(f"  LEVEL: {level.upper()}", flush=True)
    print(f"{'═'*60}", flush=True)

    reset_resp  = _post("/reset", {"level": level})
    obs         = reset_resp["observation"]
    total_tasks = reset_resp["total_tasks"]
    print(f"  Tasks in this level: {total_tasks}", flush=True)

    steps: List[dict] = []
    step_num = 0
    done     = False
    step_resp: Dict[str, Any] = {}

    while not done and step_num < MAX_STEPS_PER_LEVEL:
        step_num += 1

        task_id = reset_resp["task_id"] if step_num == 1 else step_resp.get("task_id", "?")
        print(f"\n  Step {step_num} | task={task_id}", flush=True)

        action, reasoning = _choose_action(obs)
        print(f"    → Action   : {action}", flush=True)
        print(f"    → Reasoning: {reasoning[:80]}", flush=True)

        step_resp  = _post("/step", {"action": action, "reasoning": reasoning})
        reward     = step_resp["reward"]
        done       = step_resp["done"]
        is_correct = step_resp["is_correct"]
        info       = step_resp.get("info", {})

        print(f"    ← Reward   : {reward:.4f}  |  correct={is_correct}  |  done={done}", flush=True)
        print(f"    ← Feedback : {info.get('feedback', '')[:100]}", flush=True)

        steps.append({
            "step":       step_num,
            "task_id":    step_resp.get("task_id", task_id),
            "action":     action,
            "reward":     reward,
            "is_correct": is_correct,
            "reasoning":  reasoning,
        })

        obs = step_resp.get("observation")
        if obs is None and not done:
            print("  [WARN] obs is None but done=False — breaking.", flush=True)
            break

    state   = _get("/state")
    overall = state.get("overall_score", 0.0)

    print(f"\n  {'─'*50}", flush=True)
    print(f"  Level {level.upper()} complete | steps={step_num} | overall_score={overall:.4f}", flush=True)

    return {
        "level":         level,
        "total_tasks":   total_tasks,
        "steps":         steps,
        "overall_score": overall,
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
        help="Path to write JSON results (default: results_<timestamp>.json).",
    )
    args = parser.parse_args()

    levels_to_run = [args.level] if args.level else ["easy", "medium", "hard"]
    output_path   = args.output or f"results_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"

    print(f"[START] PhishGuard-Env Baseline Inference", flush=True)
    print(f"        model  = {MODEL_NAME}", flush=True)
    print(f"        env    = {ENV_BASE_URL}", flush=True)
    print(f"        levels = {levels_to_run}", flush=True)
    print(f"        output = {output_path}", flush=True)

    try:
        health = _get("/health")
        print(f"        server status: {health.get('status', 'unknown')}", flush=True)
    except Exception as exc:
        print(f"[ERROR] Cannot reach environment server at {ENV_BASE_URL}: {exc}", flush=True)
        print("        Make sure `python env.py` is running in another terminal.", flush=True)
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    for level in levels_to_run:
        result = run_level(level)
        results.append(result)
        time.sleep(1)

    total_steps   = sum(len(r["steps"])                       for r in results)
    total_correct = sum(s["is_correct"] for r in results for s in r["steps"])
    weighted_sum  = sum(r["overall_score"] * r["total_tasks"] for r in results)
    total_tasks   = sum(r["total_tasks"]                      for r in results)
    avg_score     = weighted_sum / total_tasks if total_tasks else 0.0

    print(f"\n{'═'*60}", flush=True)
    print(f"  BASELINE SUMMARY", flush=True)
    print(f"{'═'*60}", flush=True)
    print(f"  Total steps   : {total_steps}", flush=True)
    print(f"  Correct steps : {total_correct}", flush=True)
    print(f"  Weighted score: {avg_score:.4f}  (pass threshold: {PASS_THRESHOLD})", flush=True)
    for r in results:
        print(f"  {r['level']:8s} score: {r['overall_score']:.4f}  ({r['total_tasks']} tasks)", flush=True)

    success = avg_score >= PASS_THRESHOLD

    # ── FIX: flatten all steps into a top-level tasks array ──────────────────
    # The OpenEnv validator counts tasks by reading this flat list.
    # Nesting steps inside level_results caused "Not enough tasks with graders".
    all_tasks: List[Dict[str, Any]] = []
    for r in results:
        for s in r["steps"]:
            all_tasks.append({
                "task_id":    s["task_id"],
                "is_correct": s["is_correct"],
                "reward":     s["reward"],
                "action":     s["action"],
                "reasoning":  s["reasoning"],
                "level":      r["level"],
            })

    run_summary = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "model":          MODEL_NAME,
        "env":            ENV_BASE_URL,
        "levels":         levels_to_run,
        "total_steps":    total_steps,
        "total_correct":  total_correct,
        "avg_score":      round(avg_score, 4),
        "pass_threshold": PASS_THRESHOLD,
        "success":        success,
        "tasks":          all_tasks,          # ← FIX: flat list for validator
        "level_results":  results,            # ← kept for human readability
    }

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=2)
        print(f"\n  Results saved → {output_path}", flush=True)
    except OSError as exc:
        print(f"\n  [WARN] Could not save results: {exc}", flush=True)

    print(
        f"\n[END] success={str(success).lower()} "
        f"steps={total_steps} "
        f"score={avg_score:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()