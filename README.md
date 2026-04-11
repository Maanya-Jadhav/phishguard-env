# PhishGuard-Env

An OpenEnv-compliant SOC analyst simulation environment for benchmarking LLM-based email triage agents.

## Overview

PhishGuard-Env presents an agent with phishing, malware, BEC, spam, and safe emails. The agent must classify each one using a four-action triage system. Rewards are graded continuously in `(0.0, 1.0)` — never 0, never 1 — and a health system penalises critical mistakes.

## Difficulty Levels

| Level  | Scenarios       | Tasks |
|--------|-----------------|-------|
| easy   | lv1–lv3         | 3     |
| medium | lv4–lv7         | 4     |
| hard   | lv8–lv10        | 3     |

## Action Space

| Action        | When to use                                      |
|---------------|--------------------------------------------------|
| `MARK_SAFE`   | Confirmed legitimate email — deliver to inbox    |
| `MOVE_TO_SPAM`| Bulk / unsolicited mail, no active threat        |
| `QUARANTINE`  | Suspicious but unconfirmed — hold for review     |
| `BLOCK_DOMAIN`| Confirmed phishing / BEC / malware source        |

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/your-username/phishguard-env
cd phishguard-env
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN or OPENAI_API_KEY
```

### 3. Start the environment server

```bash
python env.py
# Server starts on http://localhost:7860
```

### 4. Run the inference agent

```bash
# All three levels
python inference.py

# Single level
python inference.py --level hard

# Custom output path
python inference.py --output my_run.json
```

### 5. Run tests

```bash
pytest test_grader.py -v
```

## Docker

```bash
docker build -t phishguard-env .
docker run -p 7860:7860 -e HF_TOKEN=hf_... phishguard-env
```

## API Reference

### `POST /reset`
Start a new episode at the chosen difficulty level. The request body is **optional** — if omitted, defaults to `"easy"`.

```json
{ "level": "easy" }
```

### `POST /step`
Submit one triage action.

```json
{ "action": "BLOCK_DOMAIN", "reasoning": "Domain registered 3 days ago with SPF fail." }
```

### `GET /state`
Read-only snapshot of current environment state.

### `GET /health`
Liveness probe for HF Spaces / load-balancers.

## Environment Variables

| Variable          | Default                              | Description                    |
|-------------------|--------------------------------------|--------------------------------|
| `HF_TOKEN`        | —                                    | Hugging Face API key           |
| `OPENAI_API_KEY`  | —                                    | OpenAI-compatible API key      |
| `API_BASE_URL`    | `https://router.huggingface.co/v1`   | LLM API base URL               |
| `MODEL_NAME`      | `Qwen/Qwen2.5-72B-Instruct`          | Model to use for inference     |
| `ENV_BASE_URL`    | `http://localhost:7860`              | Environment server URL         |
