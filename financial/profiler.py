"""
financial/profiler.py
─────────────────────────────────────────────────────────────────────────────
Tenant Financial Profiler — OLISTAY AI Engine (v3)
─────────────────────────────────────────────────────────────────────────────
Computes a complete financial profile for a tenant and determines their
sustainable rent range.

v3 changes (vs v2):
  - Only THREE fields are now mandatory: tenant_id, monthly_income, savings_goal.
    Every other field is optional — the profiler is designed to degrade
    gracefully and still return a usable profile no matter how little the
    tenant chooses to share (inspired by the layered-disclosure pattern of
    Florida APD Form 65G-13.004's Individual Financial Profile: a person can
    submit a single income figure, or a fully itemised income/expense/funds
    breakdown, and the form — and now this profiler — must still work).
  - Income, expenses, and available funds can each be supplied either as a
    single aggregate number (quick path) OR as an itemised breakdown
    (income sources / expense categories / funds sources). If an itemised
    breakdown is present it takes precedence over the matching aggregate
    field.
  - New geographic fields (current_city, current_neighbourhood, gps_lat/lon):
    captured for profile completeness and downstream location-aware modules
    (hidden_costs.py, occupancy_forecaster.py, rent_predictor.py). This
    profiler does not duplicate TCO/location maths — that stays in
    hidden_costs.py — but does use job/income stability to size the
    emergency fund more realistically.
  - New employment fields (job_sector, employer_name, job_title,
    income_stability): an informal/variable income tightens the emergency
    fund target, which is the single most important Cameroonian-market
    adjustment a "job side" field can make to a financial profile.
  - New household fields (num_dependents, num_roommates, shares_housing_costs)
    and a financial-emergency flag.
  - Response now reports a profile_completeness_pct / profile_confidence
    and data_quality_notes so the caller (Spring Boot / frontend)
    can nudge the tenant toward providing more detail without ever blocking
    them from getting a result.

Cameroonian affordability calibration (unchanged from v2):
  - MAX_RENT_RATIO: 0.33 (1/3 of income) — the correct Cameroonian benchmark
    applied by the SIC social landlord and validated by Sardaouna et al. (2024).
    Note: Western models often use 0.28–0.30; Cameroon's informal economy
    reality and the absence of mortgage credit makes 1/3 the established norm.

  - MIN_EMERGENCY_MONTHS: 3 — minimum emergency fund before housing commitment.
    Increased by 1 month automatically when income is flagged informal/variable.

  - Advance payment burden: flagged separately from monthly burden.
    A tenant may afford the monthly rent but not the 3–6 month advance.
    This is the primary access barrier in Cameroon (not monthly affordability).

References:
    Sardaouna et al. (2024) — Social housing in Cameroon — Article 4 of
        Joint Order No. 0760/MINHDU/MINFI (20 Sept 2024):
        "28 to 33% of income for social housing; 10 to 33% for other cases"
    MINDCAF (2018) — Audit des loyers
    Florida APD Form 65G-13.004 A (eff. 12/2022) — Individual Financial
        Profile: structural reference for itemised income / expense /
        available-funds sections (adapted here for a Cameroonian rental
        affordability context, not a disability-subsidy context)
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Affordability thresholds — Cameroonian market ────────────────────────────
# Source: Joint Order No. 0760/MINHDU/MINFI, Article 4 (20 Sept 2024)
# Social housing: 28–33% | General market: 10–33%
MAX_RENT_RATIO       = float(os.getenv("MAX_RENT_TO_INCOME_RATIO",  0.33))  # 1/3 rule
MIN_EMERGENCY_MONTHS = int(os.getenv("MIN_EMERGENCY_FUND_MONTHS",    3))

# Typical advance payment months in the Cameroonian rental market
# (3 months is the most common; 6 months in prestige areas)
TYPICAL_ADVANCE_MONTHS = int(os.getenv("TYPICAL_ADVANCE_MONTHS", 3))

# Job sectors treated as "income instability" risk for emergency-fund sizing.
# Informal/variable income is the Cameroonian-market norm, not an edge case,
# so we size the buffer up rather than penalise the tenant elsewhere.
UNSTABLE_JOB_SECTORS = {
    "informal_self_employed",
    "informal_employee",
    "business_owner",
    "seasonal_worker",
    "unemployed",
}

UNSTABLE_INCOME_LABELS = {"variable", "seasonal", "irregular"}


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL SUB-MODELS — itemised breakdowns
# ─────────────────────────────────────────────────────────────────────────────

class IncomeSource(BaseModel):
    """One additional income line, on top of the tenant's main monthly_income."""
    income_type:    str   = Field(
        ..., description="e.g. side_business, family_support, rental_income, "
                          "freelance, pension, scholarship, other"
    )
    description:    Optional[str] = Field(None, description="Free-text note")
    monthly_amount: float = Field(..., ge=0)


class ExpenseBreakdown(BaseModel):
    """
    Optional itemised monthly expenses. If provided, this REPLACES the
    aggregate `fixed_obligations` figure entirely (it is assumed to be the
    more accurate source once the tenant has gone to the trouble of
    itemising). Categories are intentionally coarse — fine enough to be
    useful, coarse enough that a tenant can fill it in from memory.
    """
    housing_utilities:        float = Field(
        0.0, ge=0, description="Existing rent/utilities/phone/internet "
                                "at the tenant's CURRENT home (not the "
                                "property being evaluated)"
    )
    food_household_supplies:  float = Field(0.0, ge=0)
    transportation:           float = Field(0.0, ge=0)
    personal_health_insurance: float = Field(0.0, ge=0, description="Toiletries, clothing, medical, insurance premiums")
    debt_repayments:          float = Field(0.0, ge=0, description="Loans, tontine/njangi contributions, credit")
    dependents_support:       float = Field(0.0, ge=0, description="School fees, childcare, family support sent out")
    other:                    float = Field(0.0, ge=0)

    def total(self) -> float:
        return (
            self.housing_utilities + self.food_household_supplies +
            self.transportation + self.personal_health_insurance +
            self.debt_repayments + self.dependents_support + self.other
        )


class AvailableFundsBreakdown(BaseModel):
    """
    Optional itemised savings/liquid funds. If provided, this REPLACES the
    aggregate `current_savings` figure entirely.
    """
    checking_account:  float = Field(0.0, ge=0)
    savings_account:   float = Field(0.0, ge=0)
    cash_on_hand:      float = Field(0.0, ge=0)
    mobile_money:      float = Field(0.0, ge=0, description="Orange Money / MTN MoMo balance")
    other:             float = Field(0.0, ge=0)

    def total(self) -> float:
        return (
            self.checking_account + self.savings_account +
            self.cash_on_hand + self.mobile_money + self.other
        )


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TenantProfileRequest(BaseModel):
    # ── OBLIGATORY — the profile cannot be computed without these ──────────
    tenant_id:       str   = Field(..., description="Unique tenant identifier — REQUIRED")
    monthly_income:  float = Field(..., gt=0, description="Main net/gross monthly salary in CFA — REQUIRED")
    savings_goal:    float = Field(
        ..., ge=0,
        description="Tenant's stated savings target in CFA. Use 0 if the "
                    "tenant has no specific savings goal — REQUIRED (must be "
                    "explicitly provided, even if 0)."
    )

    # ── Optional — income detail ────────────────────────────────────────────
    additional_income_sources: Optional[List[IncomeSource]] = Field(
        None, description="Other income on top of monthly_income: side "
                          "business, family support, rental income, etc."
    )
    income_stability: Optional[str] = Field(
        None, description="stable | variable | seasonal | irregular — "
                          "informs emergency fund sizing"
    )

    # ── Optional — employment / "job side" ──────────────────────────────────
    job_sector: Optional[str] = Field(
        None, description="formal_private | formal_public | "
                          "informal_self_employed | informal_employee | "
                          "business_owner | student | unemployed | retired"
    )
    employer_name: Optional[str] = Field(None)
    job_title:     Optional[str] = Field(None)

    # ── Optional — geographic location ──────────────────────────────────────
    current_city:          Optional[str]   = Field(None, description="yaounde | douala | other")
    current_neighbourhood: Optional[str]   = Field(None, description="Tenant's current neighbourhood")
    gps_lat:                Optional[float] = Field(None)
    gps_lon:                Optional[float] = Field(None)

    # ── Optional — household ────────────────────────────────────────────────
    household_size:       int  = Field(1, gt=0)
    has_dependents:       bool = Field(False)
    num_dependents:       int  = Field(0, ge=0)
    num_roommates:        int  = Field(0, ge=0)
    shares_housing_costs: bool = Field(False, description="True if rent/utilities are split with roommates")

    # ── Optional — expenses (aggregate OR itemised) ─────────────────────────
    fixed_obligations: float = Field(
        0.0, ge=0,
        description="Single aggregate monthly obligations figure. Ignored "
                    "if expense_breakdown is also provided."
    )
    expense_breakdown: Optional[ExpenseBreakdown] = Field(
        None, description="Itemised monthly expenses; overrides "
                          "fixed_obligations when present."
    )

    # ── Optional — savings goal timeline & funds (aggregate OR itemised) ───
    goal_timeline_months: int   = Field(12, gt=0)
    current_savings:      float = Field(
        0.0, ge=0,
        description="Single aggregate liquid savings figure. Ignored if "
                    "available_funds_breakdown is also provided."
    )
    available_funds_breakdown: Optional[AvailableFundsBreakdown] = Field(
        None, description="Itemised savings sources; overrides "
                          "current_savings when present."
    )

    # ── Optional — situational flag ─────────────────────────────────────────
    has_financial_emergency: bool = Field(
        False, description="Tenant has flagged an urgent housing/financial situation"
    )


class FinancialProfileResponse(BaseModel):
    tenant_id:                    str
    # Effective figures actually used in the calculation (after merging
    # aggregate + itemised inputs) — always present, never None.
    effective_monthly_income:     float
    effective_fixed_obligations:  float
    effective_current_savings:    float
    income_sources_count:         int
    expense_breakdown_used:       bool
    funds_breakdown_used:         bool
    emergency_fund_months_used:   int

    disposable_income:            float
    monthly_savings_required:     float
    true_disposable:              float
    max_sustainable_rent:         float
    recommended_rent_range_min:   float
    recommended_rent_range_max:   float
    emergency_fund_target:        float
    emergency_fund_status:        str
    months_to_emergency_fund:     Optional[float]
    # Cameroonian-specific advance burden
    typical_advance_amount:       float   # 3 × max_sustainable_rent
    can_afford_typical_advance:   bool
    advance_shortfall:            float   # how much more savings needed for advance
    financial_health:             str
    can_afford_advance:           bool    # kept for backward compat
    advance_burden:                str

    # Echoed context (useful for downstream display / location-aware modules)
    job_sector:             Optional[str]
    current_city:           Optional[str]
    current_neighbourhood:  Optional[str]

    # Transparency about data quality — never blocks a result, just informs
    profile_completeness_pct: float
    profile_confidence:        str
    data_quality_notes:        List[str]

    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# CORE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_financial_profile(data: TenantProfileRequest) -> dict:

    # ── Step 1: merge aggregate + itemised inputs into effective figures ───
    additional_income = (
        sum(s.monthly_amount for s in data.additional_income_sources)
        if data.additional_income_sources else 0.0
    )
    effective_income = data.monthly_income + additional_income

    expense_breakdown_used = data.expense_breakdown is not None
    effective_obligations = (
        data.expense_breakdown.total() if expense_breakdown_used
        else data.fixed_obligations
    )

    funds_breakdown_used = data.available_funds_breakdown is not None
    effective_savings = (
        data.available_funds_breakdown.total() if funds_breakdown_used
        else data.current_savings
    )

    # ── Step 2: income stability → emergency fund sizing ────────────────────
    income_unstable = (
        (data.income_stability or "").lower() in UNSTABLE_INCOME_LABELS
        or (data.job_sector or "").lower() in UNSTABLE_JOB_SECTORS
    )
    emergency_months_used = MIN_EMERGENCY_MONTHS + (1 if income_unstable else 0)

    # ── Step 3: standard affordability math (now on effective figures) ─────
    disposable_income = effective_income - effective_obligations

    monthly_savings_required = (
        data.savings_goal / data.goal_timeline_months
        if data.savings_goal > 0 else 0.0
    )

    true_disposable = disposable_income - monthly_savings_required

    # Max sustainable rent: 1/3 of true disposable income (Cameroonian norm)
    max_sustainable_rent = max(0.0, true_disposable * MAX_RENT_RATIO)

    # Recommended range: 20%–33% of true disposable
    recommended_min = max(0.0, true_disposable * 0.20)
    recommended_max = max_sustainable_rent

    # Emergency fund: N months of total monthly obligations (N bumped up
    # automatically for informal/variable income — see Step 2)
    monthly_expenses = effective_obligations + monthly_savings_required
    emergency_fund_target = monthly_expenses * emergency_months_used

    if effective_savings >= emergency_fund_target:
        emergency_fund_status    = "ADEQUATE"
        months_to_emergency_fund = 0.0
    else:
        shortfall                = emergency_fund_target - effective_savings
        emergency_fund_status    = "INSUFFICIENT"
        months_to_emergency_fund = (
            round(shortfall / true_disposable, 1)
            if true_disposable > 0 else None
        )

    # ── Advance payment burden — primary access barrier in Cameroon ────────
    typical_advance_amount     = max_sustainable_rent * TYPICAL_ADVANCE_MONTHS
    can_afford_typical_advance = effective_savings >= typical_advance_amount
    advance_shortfall = max(0.0, typical_advance_amount - effective_savings)

    if typical_advance_amount == 0 or effective_savings >= typical_advance_amount:
        advance_burden = "LOW"
    elif effective_savings >= typical_advance_amount * 0.5:
        advance_burden = "MEDIUM"
    else:
        advance_burden = "HIGH"

    # ── Financial health ─────────────────────────────────────────────────────
    if (true_disposable >= disposable_income * 0.6
            and emergency_fund_status == "ADEQUATE"
            and can_afford_typical_advance):
        financial_health = "STRONG"
    elif true_disposable >= disposable_income * 0.4:
        financial_health = "MODERATE"
    elif true_disposable >= disposable_income * 0.2:
        financial_health = "TIGHT"
    else:
        financial_health = "CRITICAL"

    if data.has_financial_emergency:
        # An emergency overrides an otherwise-rosy read; the tenant told us
        # directly, so we don't let the numbers argue them out of it.
        if financial_health in ("STRONG", "MODERATE"):
            financial_health = "TIGHT"

    # ── Step 4: data completeness / confidence ──────────────────────────────
    completeness_checks = {
        "job_sector":              bool(data.job_sector),
        "location":                bool(data.current_city or data.current_neighbourhood),
        "income_stability":        bool(data.income_stability),
        "additional_income":       bool(data.additional_income_sources),
        "expense_breakdown":       expense_breakdown_used,
        "funds_breakdown":         funds_breakdown_used,
        "household_detail":        bool(data.num_dependents or data.num_roommates or data.has_dependents),
    }
    completeness_pct = round(
        100 * sum(completeness_checks.values()) / len(completeness_checks), 1
    )
    if completeness_pct >= 70:
        profile_confidence = "HIGH"
    elif completeness_pct >= 35:
        profile_confidence = "MEDIUM"
    else:
        profile_confidence = "LOW"

    notes: List[str] = []
    if not data.job_sector:
        notes.append("Add job_sector for a more accurate income-stability read.")
    if not (data.current_city or data.current_neighbourhood):
        notes.append("Add current_city / current_neighbourhood to unlock location-aware cost estimates.")
    if not expense_breakdown_used and data.fixed_obligations == 0:
        notes.append("No expenses provided — affordability figures assume zero fixed obligations and may be optimistic.")
    if not funds_breakdown_used and data.current_savings == 0:
        notes.append("No savings information provided — advance/caution affordability cannot be properly assessed.")
    if income_unstable:
        notes.append("Income flagged as informal/variable — emergency fund target increased by 1 month as a buffer.")
    if additional_income > 0:
        notes.append(f"{len(data.additional_income_sources)} additional income source(s) totalling {additional_income:,.0f} CFA included.")

    # ── Summary ───────────────────────────────────────────────────────────
    location_note = ""
    if data.current_city or data.current_neighbourhood:
        place = data.current_neighbourhood or data.current_city
        location_note = f" Profile compiled for a tenant currently based in {place}."

    summary = (
        f"Monthly income of {effective_income:,.0f} CFA"
        + (f" (incl. {additional_income:,.0f} CFA from {len(data.additional_income_sources)} additional source(s))" if additional_income > 0 else "")
        + f", obligations of {effective_obligations:,.0f} CFA → "
        f"disposable income {disposable_income:,.0f} CFA. "
        f"After savings commitment of {monthly_savings_required:,.0f} CFA/month, "
        f"sustainable rent range is {recommended_min:,.0f}–{recommended_max:,.0f} CFA. "
        f"Advance payment required: ~{typical_advance_amount:,.0f} CFA "
        f"({'available' if can_afford_typical_advance else f'shortfall: {advance_shortfall:,.0f} CFA'}). "
        f"Financial health: {financial_health}."
        f"{location_note}"
    )

    return {
        "tenant_id":                    data.tenant_id,
        "effective_monthly_income":     round(effective_income, 2),
        "effective_fixed_obligations":  round(effective_obligations, 2),
        "effective_current_savings":    round(effective_savings, 2),
        "income_sources_count":         len(data.additional_income_sources) if data.additional_income_sources else 0,
        "expense_breakdown_used":       expense_breakdown_used,
        "funds_breakdown_used":         funds_breakdown_used,
        "emergency_fund_months_used":   emergency_months_used,

        "disposable_income":          round(disposable_income, 2),
        "monthly_savings_required":   round(monthly_savings_required, 2),
        "true_disposable":            round(true_disposable, 2),
        "max_sustainable_rent":       round(max_sustainable_rent, 2),
        "recommended_rent_range_min": round(recommended_min, 2),
        "recommended_rent_range_max": round(recommended_max, 2),
        "emergency_fund_target":      round(emergency_fund_target, 2),
        "emergency_fund_status":      emergency_fund_status,
        "months_to_emergency_fund": (
            round(months_to_emergency_fund, 1)
            if months_to_emergency_fund is not None else None
        ),

        "typical_advance_amount":       round(typical_advance_amount, 2),
        "can_afford_typical_advance":   can_afford_typical_advance,
        "advance_shortfall":            round(advance_shortfall, 2),
        "financial_health":             financial_health,
        "can_afford_advance":           can_afford_typical_advance,   # backward compat
        "advance_burden":               advance_burden,

        "job_sector":             data.job_sector,
        "current_city":           data.current_city,
        "current_neighbourhood":  data.current_neighbourhood,

        "profile_completeness_pct": completeness_pct,
        "profile_confidence":       profile_confidence,
        "data_quality_notes":       notes,

        "summary": summary,

        # ── Backward-compat passthrough ─────────────────────────────────────
        # Downstream modules (scoring/optimality.py, recommender/content_based.py)
        # read profile["current_savings"] directly — keep this key pointing at
        # the EFFECTIVE (merged) savings figure so they automatically benefit
        # from itemised available_funds_breakdown without any changes on
        # their end.
        "current_savings": round(effective_savings, 2),
    }


@router.post("/profile", response_model=FinancialProfileResponse)
def build_financial_profile(data: TenantProfileRequest):
    """
    Computes a complete financial profile for a tenant.
    Called by Spring Boot when a tenant completes onboarding.

    Only tenant_id, monthly_income, and savings_goal are required — every
    other field is optional and the profile degrades gracefully (with a
    reported profile_confidence) if the tenant chooses not to share more.
    """
    try:
        return compute_financial_profile(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))