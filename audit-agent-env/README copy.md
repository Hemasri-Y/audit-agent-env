---
title: AuditAgentEnv
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# AuditAgentEnv 📊

An OpenEnv-compliant audit simulation environment where AI agents perform financial document auditing tasks.

## What It Does

An AI agent receives financial documents (invoice + ledger) and must:
- Extract and analyze document data
- Detect inconsistencies, missing fields, and financial risks
- Generate an audit report with findings

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `missing_field_detection` | 🟢 Easy | Find missing/empty fields in invoice | 8 |
| `mismatch_detection` | 🟡 Medium | Compare invoice vs ledger for mismatches | 10 |
| `risk_analysis` | 🔴 Hard | Detect calculation errors, tax issues, threshold breaches | 15 |

## Action Space

| Action | Parameters | Purpose |
|--------|-----------|---------|
| `extract_data` | `{source: "invoice"\|"ledger"}` | Extract a document |
| `check_missing_fields` | `{source, fields: [...]}` | Report missing fields |
| `compare_fields` | `{invoice_field, ledger_field, discrepancy}` | Report mismatches |
| `flag_risk` | `{risk_type, details, severity}` | Flag a financial risk |
| `generate_report` | `{findings: [...], risk_level}` | Submit final report (ends episode) |
| `noop` | `{}` | Skip turn (penalized) |

## Observation Space

```json
{
  "phase": "extraction | analysis | completed",
  "available_documents": ["invoice", "ledger"],
  "extracted": [],
  "documents": {},
  "identified_issues": [],
  "action_history": [],
  "steps_taken": 0,
  "max_steps": 8,
  "current_task": "missing_field_detection",
  "done": false,
  "error": null
}
```

## Workflow

```
EXTRACTION (mandatory) → ANALYSIS (find issues) → REPORT (submit findings)
```

Agent must extract both documents before analyzing. Episode ends on `generate_report` or max steps.

## Reward Design

- **Correct detection:** +0.3
- **Correct risk + correct severity:** +0.3
- **Correct risk + wrong severity:** +0.15 (partial credit)
- **False positive:** -0.2
- **Report with findings:** +0.5
- **Noop / invalid action:** -0.1 to -0.3

Grader score (0.0–1.0) is independent of rewards — checks findings against ground truth.

## Setup

```bash
# Build
docker build -t audit-agent-env .

# Run
docker run -p 7860:7860 audit-agent-env

# Test
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/reset` | POST | Start new episode. Body: `{"task": "task_name"}` |
| `/step` | POST | Take action. Body: `{"action": "...", "params": {...}}` |
| `/state` | GET | Get current state |

## Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

## Baseline Scores

| Task | Expected Score |
|------|---------------|
| missing_field_detection | 0.7 – 0.9 |
| mismatch_detection | 0.4 – 0.7 |
| risk_analysis | 0.2 – 0.5 |

## Project Structure

```
├── app.py              # FastAPI server
├── inference.py        # Agent inference script
├── models.py           # Pydantic models
├── core.py             # Environment logic
├── data_loader.py      # Excel → JSON converter
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile          # Container config
├── requirements.txt    # Dependencies
├── data/
│   ├── invoice.xlsx    # Source invoice
│   └── ledger.xlsx     # Source ledger
└── README.md           # This file
```
