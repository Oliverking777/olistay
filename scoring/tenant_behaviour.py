import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class PaymentRecord(BaseModel):
    month: str = Field(..., description="e.g. '2024-01'")
    was_on_time: bool
    days_late: int = Field(0, ge=0)
    amount_paid: float = Field(..., gt=0)
    expected_amount: float = Field(..., gt=0)

class LeaseRecord(BaseModel):
    lease_id: str
    start_date: str
    end_date: Optional[str] = None
    was_completed: bool = Field(..., description="True if tenant completed lease without early exit")
    had_disputes: bool = Field(False)
    landlord_rating: Optional[int] = Field(None, ge=1, le=5)

class TenantBehaviourRequest(BaseModel):
    tenant_id: str
    payment_history: List[PaymentRecord] = Field(default=[])
    lease_history: List[LeaseRecord] = Field(default=[])

class TenantBehaviourResponse(BaseModel):
    tenant_id: str
    reliability_score: float
    reliability_grade: str
    reliability_label: str
    total_payments: int
    on_time_payments: int
    late_payments: int
    punctuality_rate: float
    avg_days_late: float
    full_payment_rate: float
    total_leases: int
    completed_leases: int
    lease_completion_rate: float
    dispute_rate: float
    avg_landlord_rating: Optional[float]
    has_history: bool
    flags: List[str]
    summary: str

def compute_punctuality_score(records):
    if not records:
        return 50.0, 0.0, 0.0
    total = len(records)
    on_time = sum(1 for r in records if r.was_on_time)
    late_records = [r for r in records if not r.was_on_time]
    punctuality_rate = on_time / total
    avg_days_late = (
        sum(r.days_late for r in late_records) / len(late_records)
        if late_records else 0.0
    )
    if punctuality_rate >= 0.95:
        score = 100.0
    elif punctuality_rate >= 0.85:
        score = 80.0
    elif punctuality_rate >= 0.70:
        score = 60.0
    elif punctuality_rate >= 0.50:
        score = 35.0
    else:
        score = 10.0
    if avg_days_late > 30:
        score -= 20
    elif avg_days_late > 14:
        score -= 10
    elif avg_days_late > 7:
        score -= 5
    return max(0.0, min(100.0, score)), punctuality_rate, avg_days_late

def compute_payment_amount_score(records):
    if not records:
        return 50.0, 0.0
    full_payments = sum(1 for r in records if r.amount_paid >= r.expected_amount)
    full_payment_rate = full_payments / len(records)
    if full_payment_rate >= 0.98:
        score = 100.0
    elif full_payment_rate >= 0.90:
        score = 80.0
    elif full_payment_rate >= 0.75:
        score = 55.0
    else:
        score = 25.0
    return score, full_payment_rate

def compute_lease_score(leases):
    if not leases:
        return 50.0, 0.0, 0.0, None
    total = len(leases)
    completed = sum(1 for l in leases if l.was_completed)
    disputed = sum(1 for l in leases if l.had_disputes)
    rated = [l.landlord_rating for l in leases if l.landlord_rating is not None]
    completion_rate = completed / total
    dispute_rate = disputed / total
    avg_rating = sum(rated) / len(rated) if rated else None
    if completion_rate >= 0.90:
        score = 100.0
    elif completion_rate >= 0.70:
        score = 75.0
    elif completion_rate >= 0.50:
        score = 50.0
    else:
        score = 20.0
    score -= dispute_rate * 30
    if avg_rating is not None:
        if avg_rating >= 4.5:
            score += 10
        elif avg_rating >= 3.5:
            score += 5
        elif avg_rating < 2.5:
            score -= 15
    return (
        max(0.0, min(100.0, score)),
        completion_rate,
        dispute_rate,
        round(avg_rating, 2) if avg_rating is not None else None
    )

def compute_tenant_reliability(data):
    has_history = bool(data.payment_history or data.lease_history)
    if not has_history:
        return {
            "tenant_id": data.tenant_id,
            "reliability_score": 50.0,
            "reliability_grade": "C",
            "reliability_label": "UNKNOWN",
            "total_payments": 0,
            "on_time_payments": 0,
            "late_payments": 0,
            "punctuality_rate": 0.0,
            "avg_days_late": 0.0,
            "full_payment_rate": 0.0,
            "total_leases": 0,
            "completed_leases": 0,
            "lease_completion_rate": 0.0,
            "dispute_rate": 0.0,
            "avg_landlord_rating": None,
            "has_history": False,
            "flags": ["No rental history found — neutral score assigned"],
            "summary": (
                "This tenant has no recorded rental history on OLISTAY. "
                "A neutral reliability score of 50/100 has been assigned. "
                "Their score will update automatically as they build history."
            )
        }

    punctuality_score, punctuality_rate, avg_days_late = compute_punctuality_score(data.payment_history)
    amount_score, full_payment_rate = compute_payment_amount_score(data.payment_history)
    lease_score, completion_rate, dispute_rate, avg_landlord_rating = compute_lease_score(data.lease_history)

    reliability_score = (
        punctuality_score * 0.45 +
        amount_score      * 0.30 +
        lease_score       * 0.25
    )
    reliability_score = round(reliability_score, 2)

    if reliability_score >= 85:
        grade, label = "A", "EXCELLENT"
    elif reliability_score >= 70:
        grade, label = "B", "GOOD"
    elif reliability_score >= 55:
        grade, label = "C", "AVERAGE"
    elif reliability_score >= 40:
        grade, label = "D", "RISKY"
    else:
        grade, label = "F", "HIGH RISK"

    flags = []
    if punctuality_rate < 0.70:
        flags.append("Frequent late payments recorded")
    if avg_days_late > 14:
        flags.append(f"Average of {avg_days_late:.0f} days late on payments")
    if full_payment_rate < 0.85:
        flags.append("History of partial payments")
    if completion_rate < 0.60:
        flags.append("History of early lease exits")
    if dispute_rate > 0.30:
        flags.append("Multiple disputes with previous landlords")
    if avg_landlord_rating is not None and avg_landlord_rating < 2.5:
        flags.append("Low average rating from previous landlords")
    if punctuality_rate >= 0.95:
        flags.append("Excellent payment punctuality")
    if completion_rate >= 0.90 and data.lease_history:
        flags.append("Consistently completes lease agreements")
    if avg_landlord_rating is not None and avg_landlord_rating >= 4.5:
        flags.append("Highly rated by previous landlords")

    total_payments = len(data.payment_history)
    on_time = sum(1 for r in data.payment_history if r.was_on_time)

    summary = (
        f"Tenant reliability score: {reliability_score}/100 ({label}). "
        f"Based on {total_payments} payment records, this tenant paid on time "
        f"{punctuality_rate*100:.1f}% of the time with an average of "
        f"{avg_days_late:.1f} days late when late. "
        f"Lease completion rate: {completion_rate*100:.1f}% "
        f"across {len(data.lease_history)} lease(s)."
    )

    return {
        "tenant_id":             data.tenant_id,
        "reliability_score":     reliability_score,
        "reliability_grade":     grade,
        "reliability_label":     label,
        "total_payments":        total_payments,
        "on_time_payments":      on_time,
        "late_payments":         total_payments - on_time,
        "punctuality_rate":      round(punctuality_rate, 4),
        "avg_days_late":         round(avg_days_late, 2),
        "full_payment_rate":     round(full_payment_rate, 4),
        "total_leases":          len(data.lease_history),
        "completed_leases":      sum(1 for l in data.lease_history if l.was_completed),
        "lease_completion_rate": round(completion_rate, 4),
        "dispute_rate":          round(dispute_rate, 4),
        "avg_landlord_rating":   avg_landlord_rating,
        "has_history":           has_history,
        "flags":                 flags,
        "summary":               summary,
    }

@router.post("/tenant-score", response_model=TenantBehaviourResponse)
def score_tenant_behaviour(data: TenantBehaviourRequest):
    try:
        result = compute_tenant_reliability(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tenant-score/{tenant_id}/summary")
def get_tenant_summary(tenant_id: str):
    return {
        "tenant_id": tenant_id,
        "reliability_score": 50.0,
        "reliability_label": "UNKNOWN",
        "message": (
            "Start building your rental history on OLISTAY "
            "to improve your reliability score and attract better landlords."
        )
    }