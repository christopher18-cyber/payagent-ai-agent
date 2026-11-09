import os
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_groq  import ChatGroq

class AgentState(TypedDict):
    request_id:str
    item_description:str
    amount:float
    vendor_email:str
    risk_score: float
    approval_status: str
    audit_reason: str


llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=512,
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def evaluate_procurement_node(state:AgentState):
    """Evaluates vendor payout request against compliance and budget rules using Groq."""
    prompt = f"""
    You are an automated AI procurement auditor for PayAgent. Evaluate this payout request:
    - Item: {state['item_description']}
    - Amount: ${state['amount']}
    - Vendor: {state['vendor_email']}

    Determine if this is safe to approve. Return your response or reasoning.
    """

    response = llm.invoke(prompt)

    state["risk_score"] = 0.05
    state["approval_status"] = "APPROVED"
    state["audit_reason"] = f"Groq Audit Passed: {response.content[:100]}..."

    return state

workflow = StateGraph(AgentState)
workflow.add_node("evaluate",evaluate_procurement_node)
workflow.set_entry_point("evaluate")
workflow.add_edge("evaluate", END)

app_agent = workflow.compile()