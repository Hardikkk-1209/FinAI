from sms_nlp import parse_sms
from gemini_nlp import gemini_parse_sms

def hybrid_parse(sms):
    result = parse_sms(sms)

    if not result["amount"] or result["type"] == "unknown":
        return gemini_parse_sms(sms)

    return result
