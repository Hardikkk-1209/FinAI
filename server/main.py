"""
FinAI Backend - FastAPI Server
Fixed version with proper Gemini model name and error handling
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Optional, Dict, Any

app = FastAPI(title="FinAI Backend API")

# Configure CORS - allow Flutter app to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
# Make sure to set your API key in environment variables or here
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)


class GenerateRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FinAI Backend API",
        "version": "1.0.0"
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_finance_advice(request: GenerateRequest):
    """
    Generate AI-powered financial advice based on user prompt and context
    
    Args:
        request: Contains prompt (user question) and optional context (financial data)
    
    Returns:
        GenerateResponse with success status, AI response, or error message
    """
    try:
        # FIX 1: Use correct model name
        # Changed from 'models/gemini-1.5-flash' to 'gemini-1.5-flash-latest'
        # Alternative stable option: 'gemini-pro'
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Build enhanced prompt with financial context
        full_prompt = _build_prompt_with_context(request.prompt, request.context)
        
        # Generate AI response
        response = model.generate_content(full_prompt)
        
        # Extract text from response
        ai_response = response.text
        
        # FIX 2: Only return success=True if we actually got a response
        if not ai_response or len(ai_response.strip()) == 0:
            raise Exception("Empty response from AI model")
        
        return GenerateResponse(
            success=True,
            response=ai_response,
            error=None
        )
        
    except Exception as e:
        # FIX 3: Log error and return proper error response
        error_message = str(e)
        print(f"Error in generate_finance_advice: {error_message}")
        
        # Return success=False so Flutter app knows it's an error
        return GenerateResponse(
            success=False,
            response=None,
            error=error_message
        )


def _build_prompt_with_context(user_prompt: str, context: Optional[Dict[str, Any]]) -> str:
    """
    Build an enhanced prompt with financial context for better AI responses
    
    Args:
        user_prompt: The user's question
        context: Optional financial data (spending, savings, transactions, etc.)
    
    Returns:
        Enhanced prompt string with context
    """
    if not context:
        return f"""You are FinAI, a helpful personal finance assistant. 
Answer the following question in a friendly, concise manner:

{user_prompt}"""
    
    # Build context string
    context_parts = []
    
    if 'user_name' in context:
        context_parts.append(f"User: {context['user_name']}")
    
    if 'currency' in context:
        context_parts.append(f"Currency: {context['currency']}")
    
    if 'financial_health_score' in context:
        context_parts.append(f"Financial Health Score: {context['financial_health_score']}/100")
    
    if 'monthly_spending' in context:
        context_parts.append(f"Monthly Spending: {context['monthly_spending']}")
    
    if 'monthly_savings' in context:
        context_parts.append(f"Monthly Savings: {context['monthly_savings']}")
    
    if 'spending_by_category' in context:
        categories = context['spending_by_category']
        category_str = ", ".join([f"{k}: {v}" for k, v in categories.items()])
        context_parts.append(f"Spending by Category: {category_str}")
    
    if 'recent_transactions' in context:
        transactions = context['recent_transactions']
        trans_str = ", ".join([
            f"{t.get('merchant', 'Unknown')} (${t.get('amount', 0)}, {t.get('category', 'Other')})"
            for t in transactions[:5]  # Limit to 5 recent
        ])
        context_parts.append(f"Recent Transactions: {trans_str}")
    
    context_text = "\n".join(context_parts)
    
    return f"""You are FinAI, a helpful personal finance assistant.

FINANCIAL CONTEXT:
{context_text}

USER QUESTION:
{user_prompt}

Provide a helpful, personalized response based on the financial context above. 
Be concise, friendly, and actionable. If suggesting actions, be specific."""


@app.get("/models")
async def list_available_models():
    """
    List all available Gemini models that support content generation
    Useful for debugging model availability issues
    """
    try:
        models_list = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models_list.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description,
                    "supported_methods": model.supported_generation_methods
                })
        
        return {
            "success": True,
            "count": len(models_list),
            "models": models_list
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """Check if the API and Gemini connection are working"""
    try:
        # Test if we can list models (verifies API key and connection)
        list(genai.list_models())
        return {
            "status": "healthy",
            "api_configured": True,
            "gemini_connected": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "api_configured": GEMINI_API_KEY != "YOUR_API_KEY_HERE",
            "gemini_connected": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    # For development: uvicorn main:app --reload
    # For production on Render: Use Procfile or render.yaml
    uvicorn.run(app, host="0.0.0.0", port=8000)
