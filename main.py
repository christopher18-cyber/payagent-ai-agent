import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app_agent

app = FastAPI(title="PayAgent AI - Python orchestrator")

class PayoutEvaluationRequest(BaseModel):
    request_id:str
    item_description: str
    amount: float
    vendor_email: str

@app.post("/agent/evaluate")
def evaluate_payout(payload: PayoutEvaluationRequest):
    try:
        initial_state = {
            "request_id": payload.request_id,
            "item_description":payload.item_description,
            "amount": payload.amount,
            "vendor_email": payload.vendor_email,
            "risk_score":0.0,
            "approval_status":"PENDING",
            "audit_reason":""
        }

        final_state = app_agent.invoke(initial_state)

        return {
            "request_id": final_state["request_id"],
            "risk_score": final_state["risk_score"],
            "approval_status": final_state["approval_status"],
            "audit_reason": final_state["audit_reason"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

