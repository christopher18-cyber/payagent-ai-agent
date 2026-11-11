from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent import app_agent
from procure import run_procure

app = FastAPI(title="PayAgent AI - Python orchestrator")


class PayoutEvaluationRequest(BaseModel):
    request_id: str
    item_description: str
    amount: float
    vendor_email: str
    context: Dict[str, Any] = Field(default_factory=dict)


class ProcureRequest(BaseModel):
    invoices: List[str]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/agent/evaluate")
def evaluate_payout(payload: PayoutEvaluationRequest):
    try:
        initial_state = {
            "request_id": payload.request_id,
            "item_description": payload.item_description,
            "amount": payload.amount,
            "vendor_email": payload.vendor_email,
            "context": payload.context,
            "flags": [],
            "steps": [],
            "hard_block": False,
            "rule_risk": 0.0,
            "llm_verdict": "",
            "llm_risk": 0.0,
            "llm_reason": "",
            "risk_score": 0.0,
            "approval_status": "PENDING",
            "audit_reason": "",
        }

        final = app_agent.invoke(initial_state)

        return {
            "request_id": final["request_id"],
            "risk_score": final["risk_score"],
            "approval_status": final["approval_status"],
            "audit_reason": final["audit_reason"],
            "flags": final["flags"],
            "steps": final["steps"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/procure")
def procure(payload: ProcureRequest):
    invoices = [i[:2000] for i in payload.invoices if i.strip()][:8]
    if not invoices:
        raise HTTPException(status_code=400, detail="No invoices provided")
    try:
        return run_procure(invoices)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)