import google.generativeai as genai
import json
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def gemini_parse_sms(sms):
    prompt = f"""
    Extract the following as JSON:
    - type
    - amount
    - merchant
    - balance
    - date
    - time
    - bank
    SMS: "{sms}"
    """

    r = model.generate_content(prompt)

    try:
        return json.loads(r.text)
    except:
        return {"error": "Gemini returned invalid JSON", "raw": r.text}
