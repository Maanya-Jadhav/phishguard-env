# 🛡️ PhishGuard-Env

> **An OpenEnv-compliant reinforcement learning environment that benchmarks LLM agents on real-world SOC analyst email triage — built for the Meta × PyTorch × Scaler AI Hackathon 2026.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live-yellow)](https://huggingface.co/spaces/Scalar-hackathon/Phishing-env)

---

## 📌 What is PhishGuard-Env?

PhishGuard-Env is a **Gymnasium-style RL simulation environment** where an AI agent plays the role of a Security Operations Center (SOC) analyst. The agent receives structured email metadata and must correctly triage each email — deciding whether to mark it safe, move it to spam, quarantine it, or block the sender's domain entirely.

Unlike static classifiers, PhishGuard-Env models **sequential decision-making under uncertainty**. The agent manages a Health Bar (3 lives), faces adversarial email signals designed to deceive, and earns rewards strictly within the open interval `(0.0, 1.0)` — making it suitable for reinforcement learning and LLM fine-tuning workflows.

---

## 🎯 Why This Problem Matters

Phishing attacks are among the most costly and pervasive threats in cybersecurity. Modern attacks have grown far more sophisticated:

- **Business Email Compromise (BEC)** — impersonating executives to authorise wire transfers, with SPF/DMARC authentication passing on lookalike domains
- **Typosquatting** — domains like `googIe.com` (capital-I) that evade simple string matching
- **Quishing** — QR codes embedded in attachments linking to credential-harvesting pages
- **Supply-chain attacks** — legitimate partner domains compromised and added to threat intel feeds hours after the fact

Traditional static models fail to capture the **risk-aware, sequential reasoning** a real SOC analyst applies. PhishGuard-Env simulates exactly that decision loop.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py                         │
│         (LLM Agent — separate process)                  │
│                                                         │
│   1. POST /reset  →  receive first email observation    │
│   2. LLM decides action  (THINKING)                     │
│   3. POST /step   →  receive reward + next obs          │
│   4. Repeat until done=True                             │
└────────────────────┬────────────────────────────────────┘
                     │  HTTP
┌────────────────────▼────────────────────────────────────┐
│                    env.py (FastAPI Server)               │
│                    port 7860                            │
│                                                         │
│   PhishGuardEnv                                         │
│   ├── 10 email scenarios (lv1 → lv10)                  │
│   ├── 3-life Health Bar system                          │
│   └── Calls grader.py for every step                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    grader.py                            │
│   grade_spam / grade_phishing / grade_safe /            │
│   grade_malware / grade_bec                             │
│   → reward strictly in (0.0, 1.0)                      │
└─────────────────────────────────────────────────────────┘
```

The environment server and the agent are **completely decoupled** — the agent communicates exclusively over HTTP, never importing `env.py` directly. This matches the OpenEnv specification.

---

## 🧠 Environment Design

### Action Space

The agent must choose exactly one action per email:

| Action | When to use |
|---|---|
| `MARK_SAFE` | Confirmed legitimate email — deliver to inbox |
| `MOVE_TO_SPAM` | Unsolicited bulk mail with no active threat |
| `QUARANTINE` | Suspicious — hold for analyst investigation |
| `BLOCK_DOMAIN` | Confirmed phishing/BEC source — block at perimeter |

### Observation Space

Each step returns a fully structured email observation:

```json
{
  "sender":          "support@googIe.com",
  "subject":         "Urgent Security Alert – Verify Your Account",
  "body":            "We detected suspicious activity...",
  "links":           ["http://googIe-verify.com/login"],
  "has_attachments": false,
  "spf_record":      "softfail",
  "dmarc_record":    "fail",
  "urgency_level":   "critical",
  "confidence_hint": "Gateway: domain registered 3 days ago, 0 prior sends"
}
```

The `confidence_hint` field is **deliberately noisy** — it surfaces SIEM/gateway signals that may contradict the true threat level, forcing the agent to reason across all signals rather than pattern-match on a single field.

### Difficulty Levels

| Level | Scenarios | Threat Types |
|---|---|---|
| **Easy** | lv1, lv2, lv3 | SPAM, PHISH, SAFE |
| **Medium** | lv4, lv5, lv6, lv7 | MALWARE, SAFE, BEC, PHISH |
| **Hard** | lv8, lv9, lv10 | MALWARE (macro), PHISH (quishing), BEC (supply-chain) |

Hard scenarios are specifically designed to deceive: the supply-chain attack at lv10 has `spf_record: pass` and `dmarc_record: pass` — authentication signals that look clean, but threat intelligence says otherwise.

---

## 🏆 Reward System

### Open-Interval Contract

All rewards are **strictly in (0.0, 1.0)** — the endpoints 0 and 1 are never returned. This guarantees a non-zero gradient signal for RL training at every step.

| Outcome | Reward | Health Impact |
|---|---|---|
| ✅ Perfect triage | **0.95** | No drain |
| 🟡 MALWARE → QUARANTINE | **0.75** | No drain |
| 🟡 PHISH/BEC → QUARANTINE | **0.60** | No drain |
| 🟡 SPAM → BLOCK_DOMAIN | **0.40** | No drain |
| 🟡 SPAM → QUARANTINE | **0.35** | No drain |
| ❌ Wrong procedure | **0.10** | **−1 life** |
| 🔴 Business Disruption | **0.05** | **−1 life** |
| 💥 Security Breach | **0.02** | **−1 life** |

### Health Bar System

The agent starts each episode with **3 lives**. Any reward below `0.15` costs one life. When health reaches 0, the episode terminates early with a `TERMINATED` status. This punishes reckless decisions — especially marking threats as safe — while allowing partial-credit cautious actions (like quarantining instead of blocking) to pass without penalty.

### Task-Specific Graders

Five dedicated grader functions — `grade_spam`, `grade_phishing`, `grade_safe`, `grade_malware`, `grade_bec` — each handle the scoring logic for their respective task type. This satisfies the OpenEnv validator requirement of at least 3 distinct graders.

---

## 📁 Project Structure

```
phishguard-env/
├── env.py              # FastAPI environment server + PhishGuardEnv class
├── grader.py           # Reward logic, 5 per-task grader functions, TASK_REGISTRY
├── inference.py        # LLM agent driver (HF Router / OpenAI compatible)
├── models.py           # Pydantic schemas: PhishAction, StepResponse, ResetResponse
├── openenv.yaml        # OpenEnv manifest — 5 tasks with unique graders
├── test_grader.py      # pytest unit tests for all grader branches
├── server/
│   └── app.py          # Entry point: re-exports `app` and defines `main()`
├── Dockerfile          # Production container (python:3.10.14-slim, non-root user)
├── pyproject.toml      # Dependencies + [project.scripts] server = "server.app:main"
└── requirements.txt    # Flat dependency list for Docker pip install
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Maanya-Jadhav/phishguard-env
cd phishguard-env
pip install -r requirements.txt
```

### 2. Start the Environment Server

```bash
uvicorn env:app --host 0.0.0.0 --port 7860
```

The server starts on `http://localhost:7860`. Visit `http://localhost:7860/docs` for the interactive API explorer.

### 3. Set Your API Key

```bash
# Hugging Face
export HF_TOKEN=hf_...

# Or OpenAI
export OPENAI_API_KEY=sk-...
```

### 4. Run the Agent

```bash
python inference.py
```

Run a specific difficulty level:

```bash
python inference.py --level hard
```

Save results to a file:

```bash
python inference.py --output my_run.json
```

---

## 🐳 Docker

```bash
# Build
docker build -t phishguard-env .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  phishguard-env
```

---

## ⚙️ API Reference

All endpoints are served on port `7860`. Full interactive docs at `/docs`.

### `POST /reset`

Start a new episode.

**Request body** (optional — defaults to `easy` if omitted):
```json
{ "level": "easy" | "medium" | "hard" }
```

**Response:**
```json
{
  "observation": { ...email fields... },
  "task_id": "lv1",
  "level": "easy",
  "total_tasks": 10
}
```

### `POST /step`

Submit one triage action.

**Request body:**
```json
{
  "action": "MARK_SAFE | MOVE_TO_SPAM | QUARANTINE | BLOCK_DOMAIN",
  "reasoning": "optional one-sentence justification"
}
```

**Response:**
```json
{
  "observation": { ...next email... } | null,
  "reward": 0.95,
  "done": false,
  "task_id": "lv2",
  "is_correct": true,
  "info": {
    "health": 3,
    "feedback": "✅ Analysis accepted: PERFECT_TRIAGE: Correct action taken",
    "score": 0.95,
    "task_scores": [0.95]
  }
}
```

### `GET /state`

Read current episode state without advancing it.

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

---

## 🌍 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face API token (required if using HF Router) |
| `OPENAI_API_KEY` | — | OpenAI key (alternative to HF_TOKEN) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM inference router URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use for triage decisions |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |

---

## 📊 Expected Output

```
[START] PhishGuard-Env Baseline Inference
  model  = Qwen/Qwen2.5-72B-Instruct
  env    = http://localhost:7860
  levels = ['easy', 'medium', 'hard']

════════════════════════════════════════════════════════════
 LEVEL: EASY
════════════════════════════════════════════════════════════

  Step 1 | task=lv1
  → Action   : MOVE_TO_SPAM
  → Reasoning: SPF fail, no communication history, prize scam pattern
  ← Reward   : 0.9500 | correct=True | done=False
  ← Feedback : ✅ Analysis accepted: PERFECT_TRIAGE: Correct action taken

  ...

════════════════════════════════════════════════════════════
 BASELINE SUMMARY
════════════════════════════════════════════════════════════
  Total steps   : 10
  Correct steps : 8
  Weighted score: 0.8650 (pass threshold: 0.7)

[END] success=true steps=10 score=0.865
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest test_grader.py -v
```

Tests cover every branch of `grade_action()`, all reward constant ordering invariants, the Health Drain threshold guarantee, open-interval compliance, and the SPAM + MARK_SAFE security breach regression.

---

## 📋 OpenEnv Compliance

| Requirement | Status |
|---|---|
| `reset()` accepts empty body | ✅ |
| `step()` returns `(obs, reward, done, info)` | ✅ |
| Rewards strictly in `(0.0, 1.0)` | ✅ |
| At least 3 tasks with distinct graders | ✅ 5 tasks |
| `openenv.yaml` with task registry | ✅ |
| `[project.scripts]` server entry point | ✅ |
| `server/app.py` with `main()` + `app` object | ✅ |
| Dockerfile at repo root | ✅ |
| `inference.py` at repo root | ✅ |
| HF Space deployable | ✅ |

---

## 🔗 Links

- **Live HF Space:** https://huggingface.co/spaces/Scalar-hackathon/Phishing-env
- **OpenEnv Spec:** https://github.com/meta-pytorch/OpenEnv
- **Hackathon:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
