"""
AuditAgentEnv — FastAPI Server
===============================
OpenEnv-compliant server: reset(), step(), state() over HTTP.
"""

import os
import sys

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from models import AuditAction, ResetRequest
from core import AuditEnv
from data_loader import load_invoice, load_ledger

invoice = load_invoice()
ledger = load_ledger()

env = AuditEnv(invoice=invoice, ledger=ledger)

app = FastAPI(title="AuditAgentEnv", version="1.0.0")


@app.get("/")
def health():
    return {"status": "ok", "env": "AuditAgentEnv"}


@app.post("/reset")
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict:
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


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
