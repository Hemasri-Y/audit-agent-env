"""
AuditAgentEnv — Inference Script iterated
====================================
Runs LLM agent against all 3 tasks.
Mandatory [START]/[STEP]/[END] logging format.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Config ──
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "audit_agent_env"
MAX_STEPS_MAP = {
    "missing_field_detection": 8,
    "mismatch_detection": 10,
    "risk_analysis": 15,
}
TASKS = ["missing_field_detection", "mismatch_detection", "risk_analysis"]


###############################################################################
# LOGGING — exact hackathon format
###############################################################################

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    d = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={d} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)


###############################################################################
# OBSERVATION SUMMARIZER — keeps LLM prompt small and readable
###############################################################################

def summarize_obs(observation: dict) -> str:
    """Turn raw observation dict into a concise string for the LLM."""
    lines = []
    lines.append(f"Phase: {observation.get('phase')}")
    lines.append(f"Task: {observation.get('current_task')}")
    lines.append(f"Step: {observation.get('steps_taken')}/{observation.get('max_steps')}")
    lines.append(f"Extracted docs: {observation.get('extracted', [])}")

    err = observation.get("error")
    if err:
        lines.append(f"LAST ERROR: {err}")

    docs = observation.get("documents", {})

    if "invoice" in docs:
        inv = docs["invoice"]
        lines.append("")
        lines.append("=== INVOICE ===")
        lines.append(f"invoice_id: {inv.get('invoice_id')}")
        lines.append(f"vendor_name: {inv.get('vendor_name')}")
        lines.append(f"vendor_gstin: {inv.get('vendor_gstin')}")
        lines.append(f"buyer_name: {inv.get('buyer_name')}")
        lines.append(f"buyer_gstin: {inv.get('buyer_gstin')}")
        lines.append(f"invoice_date: {inv.get('invoice_date')}")
        lines.append(f"due_date: {inv.get('due_date')}")
        lines.append(f"payment_terms: {inv.get('payment_terms')}")
        lines.append(f"approval_threshold: {inv.get('approval_threshold')}")
        lines.append(f"subtotal: {inv.get('subtotal')}")
        lines.append(f"total_tax: {inv.get('total_tax')}")
        lines.append(f"grand_total: {inv.get('grand_total')}")
        for it in inv.get("line_items", []):
            lines.append(
                f"  {it['item_id']}: desc={it['description']}, qty={it['quantity']}, "
                f"unit_price={it['unit_price']}, total={it['total']}, "
                f"tax_rate={it['tax_rate']}%, tax_amount={it['tax_amount']}"
            )

    if "ledger" in docs:
        led = docs["ledger"]
        lines.append("")
        lines.append("=== LEDGER ===")
        for e in led.get("entries", []):
            lines.append(
                f"  {e['entry_id']}: date={e['date']}, ref={e['ref']}, "
                f"vendor='{e['vendor']}', desc={e['desc']}, "
                f"debit={e['debit']}, credit={e['credit']}, "
                f"tax={e['tax']}, total={e['total']}"
            )

    issues = observation.get("identified_issues", [])
    if issues:
        lines.append("")
        lines.append("=== ISSUES FOUND ===")
        for iss in issues:
            lines.append(f"  - {iss.get('type')}: {iss.get('details')} [{iss.get('status')}]")

    history = observation.get("action_history", [])
    if history:
        lines.append("")
        lines.append("=== RECENT ACTIONS ===")
        for h in history[-5:]:
            lines.append(f"  {h}")

    return "\n".join(lines)


###############################################################################
# SYSTEM PROMPTS
###############################################################################

SYSTEM_PROMPTS = {
    "missing_field_detection": textwrap.dedent("""\
You are an AI auditor. Find missing or empty fields in the invoice.

RESPOND WITH ONLY A SINGLE JSON OBJECT. No explanation. No markdown.

Actions to use IN ORDER:
1. {"action":"extract_data","params":{"source":"invoice"}}
2. {"action":"extract_data","params":{"source":"ledger"}}
3. {"action":"check_missing_fields","params":{"source":"invoice","fields":["field_name"]}}
4. {"action":"generate_report","params":{"findings":["description"],"risk_level":"low"}}

Look for fields that are null or empty."""),

    "mismatch_detection": textwrap.dedent("""\
You are an AI auditor. Compare invoice and ledger to find mismatches.

RESPOND WITH ONLY A SINGLE JSON OBJECT. No explanation. No markdown.

Actions to use IN ORDER:
1. {"action":"extract_data","params":{"source":"invoice"}}
2. {"action":"extract_data","params":{"source":"ledger"}}
3. {"action":"compare_fields","params":{"invoice_field":"path","ledger_field":"path","discrepancy":0}}
4. {"action":"generate_report","params":{"findings":["description"],"risk_level":"medium"}}

Check: line item totals vs ledger debits, vendor name consistency."""),

    "risk_analysis": textwrap.dedent("""\
You are an AI auditor. Detect calculation errors, tax issues, threshold breaches.

RESPOND WITH ONLY A SINGLE JSON OBJECT. No explanation. No markdown.

Actions to use IN ORDER:
1. {"action":"extract_data","params":{"source":"invoice"}}
2. {"action":"extract_data","params":{"source":"ledger"}}
3. {"action":"flag_risk","params":{"risk_type":"type","details":"explanation","severity":"high|medium|low"}}
4. {"action":"generate_report","params":{"findings":["description"],"risk_level":"high"}}

Check: qty*unit_price vs total, tax calculations, grand_total vs approval_threshold, subtotal accuracy.
Valid risk_type: calculation_error, tax_mismatch, threshold_breach, subtotal_inconsistency"""),
}


###############################################################################
# LLM CALL + JSON PARSER
###############################################################################

def parse_llm_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM text."""
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    if "```" in text:
        for block in text.split("```"):
            block = block.strip().removeprefix("json").strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # 3. Extract first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 4. Give up
    print(f"[DEBUG] PARSE FAILED: {text[:200]}", flush=True)
    return {"action": "noop", "params": {}}


def call_llm(client: OpenAI, task: str, observation: dict) -> dict:
    """Send observation to LLM, return parsed action dict."""
    summary = summarize_obs(observation)
    user_msg = f"Current state:\n{summary}\n\nYour next action (JSON only):"

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "").strip()
        print(f"[DEBUG] LLM says: {raw[:200]}", flush=True)
        return parse_llm_json(raw)

    except Exception as exc:
        print(f"[DEBUG] LLM CALL FAILED: {type(exc).__name__}: {exc}", flush=True)
        return {"action": "noop", "params": {}}


###############################################################################
# TASK RUNNER (local mode — direct env)
###############################################################################

async def run_one_task(client: OpenAI, env, task: str):
    from models import AuditAction, TaskName

    task_enum = TaskName(task)
    max_steps = MAX_STEPS_MAP[task]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_enum)
        observation = obs.model_dump()

        for step in range(1, max_steps + 1):
            if observation.get("done", False):
                break

            action_dict = call_llm(client, task, observation)
            print(f"[DEBUG] Parsed action: {action_dict}", flush=True)

            action = AuditAction(**action_dict)
            result = env.step(action)

            reward = result.reward
            done = result.done
            observation = result.observation.model_dump()
            error = result.observation.error

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_dict.get("action", "unknown"),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                score = result.info.get("grader_score", 0.0)
                break

        score = max(0.0, min(1.0, score))
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] TASK ERROR: {type(exc).__name__}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


###############################################################################
# TASK RUNNER (docker / remote mode)
###############################################################################

async def run_one_task_remote(client: OpenAI, env, task: str):
    max_steps = MAX_STEPS_MAP[task]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs_result = await env.reset({"task": task})
        observation = obs_result if isinstance(obs_result, dict) else obs_result.model_dump()

        for step in range(1, max_steps + 1):
            if observation.get("done", False):
                break

            action_dict = call_llm(client, task, observation)
            print(f"[DEBUG] Parsed action: {action_dict}", flush=True)

            step_result = await env.step(action_dict)
            result = step_result if isinstance(step_result, dict) else step_result.model_dump()

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            observation = result.get("observation", {})
            error = observation.get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_dict.get("action", "unknown"),
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                score = result.get("info", {}).get("grader_score", 0.0)
                break

        score = max(0.0, min(1.0, score))
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] TASK ERROR: {type(exc).__name__}: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


###############################################################################
# MAIN
###############################################################################

async def main():
    # ── Validate config ──
    if not API_KEY:
        print("[ERROR] HF_TOKEN not set! Run: export HF_TOKEN=hf_your_token", flush=True)
        return

    print(f"[INFO] API_BASE_URL = {API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME}", flush=True)
    try:
        masked = API_KEY[:6] + "..." if API_KEY else "(missing)"
    except Exception:
        masked = "(invalid)"
    print(f"[INFO] API_KEY      = {masked}", flush=True)
    print(flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # ── Quick sanity check — can we reach the LLM? ──
    print("[INFO] Testing LLM connection...", flush=True)
    try:
        test = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "say ok"}],
            max_tokens=5,
        )
        print(f"[INFO] LLM reachable! Response: {test.choices[0].message.content}", flush=True)
    except Exception as exc:
        print(f"[ERROR] Cannot reach LLM: {type(exc).__name__}: {exc}", flush=True)
        print("[ERROR] Fix your HF_TOKEN / API_BASE_URL / MODEL_NAME and retry.", flush=True)
        return

    print(flush=True)

    # ── Run tasks ──
    if IMAGE_NAME:
        # remote/docker mode — robust import and error handling
        try:
            try:
                from audit_agent_env import AuditAgentEnvEnv
            except Exception:
                # ensure current dir is on path (submission environments vary)
                import sys, os

                cwd = os.getcwd()
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)
                from audit_agent_env import AuditAgentEnvEnv

            for task in TASKS:
                env = None
                try:
                    env = await AuditAgentEnvEnv.from_docker_image(IMAGE_NAME)
                    await run_one_task_remote(client, env, task)
                except Exception as exc:
                    print(f"[ERROR] Remote task failed: {type(exc).__name__}: {exc}", flush=True)
                finally:
                    if env:
                        try:
                            await env.close()
                        except Exception:
                            pass
        except Exception as exc:
            print(f"[ERROR] Cannot initialize remote env: {type(exc).__name__}: {exc}", flush=True)
            return
    else:
        from data_loader import load_invoice, load_ledger
        from core import AuditEnv

        invoice = load_invoice()
        ledger = load_ledger()
        env = AuditEnv(invoice=invoice, ledger=ledger)

        for task in TASKS:
            await run_one_task(client, env, task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:  # top-level catch so hackathon runner sees diagnostics
        import traceback, sys

        traceback.print_exc()
        print(f"[ERROR] inference.py failed: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)