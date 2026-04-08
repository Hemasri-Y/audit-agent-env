"""
AuditAgentEnv — FastAPI Server
===============================
Thin wrapper over AuditEnv. Three endpoints, nothing more.
"""

from fastapi import FastAPI, HTTPException
from models import AuditAction, AuditObservation, ResetRequest, StepResult, TaskName
from core import AuditEnv
from data_loader import load_invoice, load_ledger

# ── Load data once at startup ──
invoice = load_invoice()
ledger = load_ledger()

# ── Create environment ──
env = AuditEnv(invoice=invoice, ledger=ledger)

# ── FastAPI app ──
app = FastAPI(title="AuditAgentEnv", version="1.0.0")


@app.get("/")
def health():
    return {"status": "ok", "env": "AuditAgentEnv"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()) -> dict:
    obs = env.reset(request.task)
    return obs.model_dump()


@app.post("/step")
def step(action: AuditAction) -> dict:
    try:
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state() -> dict:
    try:
        obs = env.state()
        return obs.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
