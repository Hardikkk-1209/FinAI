# main.py - FinAI FastAPI Server for Render Deployment
import os
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="FinAI Assistant",
    description="AI-powered personal finance assistant using Google Gemini",
    version="1.0.0"
)

# Configure CORS for all origins (required for Flutter app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Flutter app domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Google Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Finance advisor system prompt
FINANCE_SYSTEM_PROMPT = """You are FinAI, an AI-powered personal finance assistant inside a mobile app.
Your ONLY job is to help users understand, analyze, and improve their personal finances.

CRITICAL BEHAVIOR RULES:
- Stay strictly within the domain of personal finance, budgeting, savings, spending analysis, debt management, credit cards, and basic investing concepts.
- If the user asks for anything unrelated to money/finance, politely refuse and redirect them back to financial topics.
- Never roleplay, tell stories, generate code, or answer questions about unrelated domains (e.g., medicine, politics, entertainment, relationships, exams, coding, agriculture, etc.).

APP CONTEXT:
- The app automatically reads and categorizes transaction SMS and bank data (using NLP).
- It computes a Financial Health Score based on income, expenses, savings, debts, and risk signals.
- It detects unusual/anomalous spending patterns and risky transactions.
- It provides personalized, actionable tips to help users budget better, save more, and avoid unnecessary risk.

TONE & STYLE:
- Be concise, clear, and friendly.
- Explain concepts in simple language so that a non-expert can understand.
- Prefer short paragraphs and bullet points.
- When giving suggestions, be practical and realistic. Avoid generic motivational quotes.

HOW TO ANSWER:
1. When the user asks general finance questions:
   - Explain the concept briefly.
   - Show simple, concrete examples with numbers.
   - If relevant, suggest simple actions they can take.

2. When the user asks about their own finances (e.g., “Am I spending too much on X?”):
   - Ask for any missing key details (income, typical monthly expenses, debts, timeframe).
   - Then respond with a structured analysis and 2–4 specific recommendations.

3. When the user mentions risky behavior (e.g., high-interest loans, heavy credit card usage):
   - Clearly highlight the risk.
   - Suggest safer alternatives or mitigation steps.
   - Emphasize long-term financial health.

4. When you are NOT sure or data is missing:
   - Be honest. Say “I don’t have enough information to be precise, but here is a safe general guideline…”
   - Never fabricate numbers, transactions, or user data.

SAFETY & LIMITS:
- Do NOT give legal, tax, or investment advice that sounds like a guarantee.
- Use language like “general information”, “this is not professional advice”, and “please consult a qualified financial advisor or tax professional for decisions involving large amounts or legal implications.”
- Do NOT recommend specific stocks, individual crypto coins, or high-risk speculative products.
- You may talk about broad asset classes (e.g., equity mutual funds, index funds, fixed deposits) in a generic, educational way.

IF THE USER'S MESSAGE IS OFF-TOPIC:
- Say something like: “I am designed only for personal finance guidance. Let us talk about your money, spending, saving, or financial goals.”
- Then offer a helpful finance-related follow-up question.

Your primary goal: help the user make better day-to-day financial decisions, understand their money patterns, and build healthier financial habits over time.
"""

# Pydantic models
class TextPrompt(BaseModel):
    # user question
    prompt: str
    # optional financial data context from frontend/backend
    # e.g. monthly summary, category spends, suspicious txns, etc.
    context: Optional[dict] = None


class APIResponse(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "FinAI Assistant API is running successfully!",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "text_chat": "/generate (POST)",
            "documentation": "/docs"
        },
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": "gemini-1.5-flash",
        "service": "FinAI Assistant API",
    }


# Text-based finance advice endpoint
@app.post("/generate", response_model=APIResponse)
async def generate_finance_advice(body: TextPrompt):
    """
    Generate personal finance advice from text prompt + optional data context.
    """
    try:
        user_prompt = (body.prompt or "").strip()
        if not user_prompt:
            return APIResponse(
                success=False,
                response="",
                error="Please provide a finance-related question.",
            )

        context_data = body.context or {}
        context_json = json.dumps(context_data, indent=2)

        # Combine system prompt with user question and structured financial data
        full_prompt = f"""
{FINANCE_SYSTEM_PROMPT}

Below is the user's current financial data in JSON format.
Use ONLY this data for any numbers, amounts, or statistics.
If something is missing, say so and give general guidance instead of guessing.

USER DATA (JSON):
{context_json}

USER QUESTION:
{user_prompt}

INSTRUCTIONS:
- Do not invent or assume any transactions or amounts.
- If you mention any number, it must come from the JSON above.
- If the question asks for a summary, first restate the key numbers, then give 2–4 practical suggestions.
- Keep the answer concise and friendly.
"""

        # Generate response using Gemini
        response = model.generate_content(full_prompt)

        if not getattr(response, "text", None):
            return APIResponse(
                success=False,
                response="",
                error="No response generated. Please try again.",
            )

        return APIResponse(
            success=True,
            response=response.text,
            error=None,
        )

    except Exception as e:
        print(f"Error in generate_finance_advice: {str(e)}")
        return APIResponse(
            success=False,
            response="",
            error="Sorry, I'm having trouble processing your request. Please try again later.",
        )


# Ping endpoint for monitoring
@app.get("/ping")
async def ping():
    return {"ping": "pong", "timestamp": "ok"}


# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
