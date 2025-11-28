from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import os
import random
from datetime import datetime
import re

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


class TxIn(BaseModel):
    userId: str
    amount: float
    timestamp: str  # ISO string
    merchant: str
    merchant_category: str = ""
    is_international: bool = False
    currency: str = "INR"
    meta: Dict[str, Any] = {}


class AnomalyResp(BaseModel):
    anomaly: bool
    score: float
    reasons: List[str]


ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "iso_model.joblib")
_ml_model = None


def load_ml_model():
    global _ml_model
    if _ml_model is None and os.path.exists(ML_MODEL_PATH):
        _ml_model = joblib.load(ML_MODEL_PATH)
    return _ml_model


def get_user_history_stub(user_id: str) -> Dict[str, Any]:
    return {
        "avg_amount": 600.0,
        "median_amount": 350.0,
        "std_amount": 400.0,
        "transactions_today": 2,
        "merchants": ["Zomato", "SBI Card", "Amazon"],
        "country": "IN",
        "timezone_offset_hours": 5.5,
    }


def mask_personal_data(text: str) -> str:
    return re.sub(r"\d{8,}", "[MASKED_NUMBER]", text)


def _parse_iso_hour(ts: str, meta: Dict[str, Any]) -> int:
    """
    Parse ISO timestamp robustly and return hour (0-23).
    Handles timestamps ending with 'Z' by converting to +00:00.
    Falls back to meta['hour'] or current UTC hour if parsing fails.
    """
    if not ts:
        return int(meta.get("hour", datetime.utcnow().hour))
    try:
        # fromisoformat doesn't accept trailing 'Z' — normalize it
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        return dt.hour
    except Exception:
        return int(meta.get("hour", datetime.utcnow().hour))


def rule_based_check(tx: TxIn, user_history: dict) -> Tuple[bool, float, List[str]]:
    reasons: List[str] = []
    amount = tx.amount
    avg = user_history.get("avg_amount", 500.0)
    std = user_history.get("std_amount", max(1.0, avg * 0.6))
    med = user_history.get("median_amount", avg)
    hour = _parse_iso_hour(tx.timestamp, tx.meta)

    # Rules
    if amount > 20000:
        reasons.append("Very large transaction amount")
    if amount > med * 3:
        reasons.append("High compared to user's typical transaction")
    if amount > (avg + 3 * std):
        reasons.append("Amount is far outside typical variance")
    if hour < 6 or hour > 23:
        reasons.append("Transaction at unusual hour")
    if tx.merchant not in user_history.get("merchants", []):
        reasons.append("Merchant is new/unfamiliar")
    if tx.is_international and amount > 1000:
        reasons.append("International high-value transaction")
    if user_history.get("transactions_today", 0) > 10:
        reasons.append("Unusually high transaction frequency today")

    anomaly = len(reasons) > 0
    score = min(1.0, len(reasons) / 5.0)
    return anomaly, score, reasons


def fake_detector(tx: TxIn) -> Tuple[bool, float, List[str]]:
    r = random.random()
    if r < 0.05:
        return True, 0.95, ["Simulated high anomaly (demo)"]
    elif r < 0.2:
        return True, 0.6, ["Simulated medium anomaly (demo)"]
    else:
        return False, 0.05, []


def ml_detector(tx: TxIn) -> Tuple[bool, float, List[str]]:
    model = load_ml_model()
    if model is None:
        raise RuntimeError("ML model not found on server.")
    hour = _parse_iso_hour(tx.timestamp, tx.meta)
    merchant_hash = abs(hash(tx.merchant)) % 1000
    x = np.array([[tx.amount, hour, 1 if tx.is_international else 0, merchant_hash]], dtype=float)
    score_raw = model.decision_function(x)[0]
    pred = model.predict(x)[0]
    is_anom = (pred == -1)
    score = float(1.0 / (1.0 + np.exp(-score_raw)))
    reasons: List[str] = []
    if is_anom:
        reasons.append("ML model flagged as outlier (demo model).")
    return is_anom, score, reasons


@router.post("/rule", response_model=AnomalyResp)
async def detect_rule(tx: TxIn):
    user_history = get_user_history_stub(tx.userId)
    masked_merchant = mask_personal_data(tx.merchant)
    # pass a copy or use masked_merchant only in response/logs; don't mutate input
    anom, score, reasons = rule_based_check(tx, user_history)
    return AnomalyResp(anomaly=anom, score=score, reasons=reasons)


@router.post("/fake", response_model=AnomalyResp)
async def detect_fake(tx: TxIn):
    anom, score, reasons = fake_detector(tx)
    return AnomalyResp(anomaly=anom, score=score, reasons=reasons)


@router.post("/ml", response_model=AnomalyResp)
async def detect_ml(tx: TxIn):
    try:
        anom, score, reasons = ml_detector(tx)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return AnomalyResp(anomaly=anom, score=score, reasons=reasons)
