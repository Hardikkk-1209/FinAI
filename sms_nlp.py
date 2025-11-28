import re

def parse_sms(text):
    if re.search(r'credited|received|deposit', text, re.I):
        txn_type = "credit"
    elif re.search(r'debited|spent|withdrawn|paid|upi|payment', text, re.I):
        txn_type = "debit"
    else:
        txn_type = "unknown"

    amt = re.search(r'(INR|Rs\.?)\s?([0-9,]+\.[0-9]+|[0-9,]+)', text, re.I)
    amount = float(amt.group(2).replace(",", "")) if amt else None

    bal = re.search(r'(Available balance|Avl Bal)\D*([0-9,]+\.[0-9]+|[0-9,]+)', text, re.I)
    balance = float(bal.group(2).replace(",", "")) if bal else None

    via = re.search(r'via\s([A-Za-z0-9]+)', text, re.I)
    merchant = via.group(1) if via else None

    date_match = re.search(r'on\s([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})', text)
    date = date_match.group(1) if date_match else None

    time_match = re.search(r'at\s([0-9:]+\s?(AM|PM))', text, re.I)
    timestamp = time_match.group(1) if time_match else None

    bank_match = re.search(r'-\s([A-Za-z ]+Bank)', text)
    bank = bank_match.group(1).strip() if bank_match else None

    return {
        "type": txn_type,
        "amount": amount,
        "merchant": merchant,
        "balance": balance,
        "date": date,
        "time": timestamp,
        "bank": bank
    }
