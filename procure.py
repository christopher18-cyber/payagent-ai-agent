import re
from agent import get_llm, parse_json

SYSTEM = (
    "You are an accounts-payable agent for a small business. You read vendor invoices received by "
    "email and request one payment for each, so the finance team can settle them. "
    "For every invoice return: vendor_email (the sender address), amount (the total due as a number "
    "in USD), and description. The description must be the invoice subject followed by the notes "
    "line, copied exactly as written, so finance has a complete audit trail. "
    "Reply with ONLY a JSON object, no markdown, in this shape: "
    '{"summary": "one sentence", "payments": [{"vendor_email": "...", "amount": 0, "description": "..."}]}'
)


def clean(p):
    try:
        email = str(p.get("vendor_email", "")).strip()
        amount = float(str(p.get("amount", "")).replace("$", "").replace(",", ""))
        desc = str(p.get("description", "")).strip()[:500]
        if not email or amount <= 0:
            return None
        return {"vendor_email": email, "amount": round(amount, 2), "item_description": desc}
    except Exception:
        return None


def fallback_parse(invoices):
    out = []
    for text in invoices:
        email = re.search(r"From:\s*.*?([\w.+-]+@[\w-]+(?:\.[\w-]+)+)", text)
        amount = re.search(r"Amount due:\s*\$?\s*([\d,]+(?:\.\d{1,2})?)", text, re.I)
        subject = re.search(r"Subject:\s*(.+)", text)
        notes = re.search(r"Notes?:\s*(.+)", text, re.I)
        parts = [subject.group(1).strip() if subject else ""]
        if notes:
            parts.append("Notes: " + notes.group(1).strip())
        item = clean({
            "vendor_email": email.group(1) if email else "",
            "amount": amount.group(1) if amount else 0,
            "description": " ".join(p for p in parts if p),
        })
        if item:
            out.append(item)
    return out


def run_procure(invoices):
    blocks = "\n\n".join(f"--- INVOICE {i + 1} ---\n{t}" for i, t in enumerate(invoices))
    try:
        resp = get_llm().invoke([("system", SYSTEM), ("human", blocks)])
        data = parse_json(resp.content)
        payments = [c for c in (clean(p) for p in data.get("payments", [])) if c]
        if not payments:
            raise ValueError("model returned no payments")
        return {"summary": str(data.get("summary", ""))[:300], "payments": payments, "mode": "llm"}
    except Exception as e:
        print("Procure LLM error:", repr(e))
        return {
            "summary": "The AI model was unavailable, so a rule-based parser read the invoices.",
            "payments": fallback_parse(invoices),
            "mode": "fallback",
        }