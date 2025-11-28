# main.py
# Load environment variables first
from dotenv import load_dotenv
load_dotenv()   # loads GEMINI_API_KEY from .env

import os
import re
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Google GenAI SDK (Gemini)
from google import genai

# include anomaly router (make sure anomaly_router.py exists in project root)
from anomaly_router import router as anomaly_router

# -----------------------------------------------------------
# Lazy Gemini client initialization
# -----------------------------------------------------------
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
_client = None

def get_genai_client():
    """Return a genai.Client if API key present, else None."""
    global _client
    if _client is not None:
        return _client
    if not _GEMINI_KEY:
        return None
    # create and cache the client
    _client = genai.Client(api_key=_GEMINI_KEY)
    return _client

# -----------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------
app = FastAPI(title="Finance-Only Chat (Gemini + FastAPI)")

# register anomaly router
app.include_router(anomaly_router)

# -----------------------------------------------------------
# Finance Keyword Filter (fast)
# -----------------------------------------------------------
FINANCE_KEYWORDS = [
    "expense", "spend", "spending", "budget", "savings", "save", "loan", "emi",
    "interest", "credit card", "card", "transactions", "balance", "investment",
    "mutual fund", "sip", "tax", "insurance", "mortgage", "net worth",
    "salary", "paycheck", "retirement", "financial", "broker"
]

SYSTEM_PROMPT = """
You are SmartFinanceBot, an assistant inside a personal finance app.

Rules:
- Only respond to finance-related user queries:
  budgeting, savings, loans, EMI, credit cards, insurance, investments,
  taxes, spending analysis, financial literacy, and user's own finance data.
- If question is NOT finance-related, reply exactly:
  "I can only answer finance-related questions inside this app."

- Never reveal unmasked personal data (card numbers, account numbers).
- Provide helpful and simple explanations.
- For legal/tax advice, tell user to contact a real professional.
"""

# -----------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------
class ChatRequest(BaseModel):
    userId: str
    message: str
    locale: Optional[str] = "en-IN"

class ChatResponse(BaseModel):
    reply: str
    intent: str
    topic: str
    usedUserData: bool = False

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def looks_finance_by_keywords(text: str) -> bool:
    text_l = text.lower()
    for kw in FINANCE_KEYWORDS:
        if kw in text_l:
            return True
    return False

async def classify_with_gemini(text: str) -> str:
    """
    Ask Gemini to classify FINANCE vs NON_FINANCE.
    If Gemini client is not configured, return NON_FINANCE to be safe.
    """
    client = get_genai_client()
    if client is None:
        # fallback: assume non-finance (prevents accidentally answering OOD queries)
        return "NON_FINANCE"

    prompt = (
        "Classify this message as FINANCE or NON_FINANCE.\n"
        "Return only the one word.\n\n"
        f"Message: \"{text}\""
    )

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        content = getattr(resp, "text", None) or str(resp)
        token = content.strip().split()[0].upper()
        if token in ["FINANCE", "NON_FINANCE"]:
            return token
    except Exception as e:
        # Log error server-side (print for now)
        print("Classification error:", e)

    return "NON_FINANCE"

def get_user_finance_context(user_id: str) -> Dict[str, Any]:
    """
    Replace this stub with real DB or ML model calls.
    """
    return {
        "healthScore": 72,
        "recentTransactions": [
            {"date": "2025-11-10", "merchant": "Zomato", "amount": 420.50, "category": "Food & Dining"},
            {"date": "2025-11-09", "merchant": "SBI Card", "amount": 15000.0, "category": "EMI"},
        ],
        "budgets": {
            "food": 5000,
            "entertainment": 2000
        },
        "anomalies": []
    }

def mask_personal_data(text: str) -> str:
    """
    Mask long digit sequences that may represent account/card numbers.
    """
    return re.sub(r"\d{8,}", "[MASKED_NUMBER]", text)

# -----------------------------------------------------------
# Chat Endpoint
# -----------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_id = req.userId
    msg = req.message.strip()

    if not user_id or not msg:
        raise HTTPException(status_code=400, detail="userId and message required")

    # 1) Quick local keyword check
    if looks_finance_by_keywords(msg):
        topic = "FINANCE"
    else:
        # 2) Ask Gemini classifier (may fallback to NON_FINANCE)
        topic = await classify_with_gemini(msg)

    # If not finance - block immediately
    if topic != "FINANCE":
        return ChatResponse(
            reply="I can only answer finance-related questions inside this app.",
            intent="OUT_OF_DOMAIN",
            topic=topic,
            usedUserData=False
        )

    # Ensure LLM client exists before making generation call
    client = get_genai_client()
    if client is None:
        raise HTTPException(status_code=503, detail="LLM not configured. Contact admin.")

    # 3) Fetch finance context (replace stub)
    context = get_user_finance_context(user_id)
    context_json = json.dumps(context)

    # 4) Mask any sensitive digits from user input
    masked_msg = mask_personal_data(msg)

    # 5) Build full prompt for Gemini
    full_prompt = f"""
System instructions:
{SYSTEM_PROMPT}

User Finance Context (JSON, masked):
{context_json}

User question:
{masked_msg}

Respond concisely using the data above.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )

        reply_text = getattr(response, "text", None) or str(response)

    except Exception as e:
        # server-side log
        print("Gemini generation error:", e)
        raise HTTPException(status_code=502, detail="LLM service error")

    return ChatResponse(
        reply=reply_text.strip(),
        intent="FINANCE_QUERY",
        topic="FINANCE",
        usedUserData=True
    )

# -----------------------------------------------------------
# Health Check
# -----------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}
