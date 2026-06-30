"""
scoring/optimality.py
─────────────────────────────────────────────────────────────────────────────
Multidimensional Housing Optimality Scoring Engine — OLISTAY AI Engine
─────────────────────────────────────────────────────────────────────────────
Scores each candidate property across six dimensions and produces a composite
optimality score.

Cameroonian calibration updates:
  - Financial score thresholds use the 1/3 income rule (not 0.28 Western norm)
  - Advance payment burden is a first-class scoring dimension
    (a tenant may score well on monthly rent but fail on advance affordability)
  - shared_wc / unit_type are scored in the household dimension
  - Generator presence weighted higher (Eneo load-shedding is frequent)
  - Flood risk triggers hard deduction (common in Yaoundé/Douala valleys)
  - Stability score accounts for title type (foncier vs occupation vs none)

v3 update:
  - TenantData now carries job_sector / current_city / income_stability,
    forwarded into the financial profiler (financial/profiler.py v3) so its
    informal/variable-income emergency-fund adjustment applies here too.
    All three remain optional — TenantData keeps working unchanged for
    callers that don't set them.

References:
    Joint Order No. 0760/MINHDU/MINFI (20 Sept 2024) — Article 4
    Sardaouna et al. (2024) — Social housing in Cameroon
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

from financial.profiler import compute_financial_profile, TenantProfileRequest
from financial.hidden_costs import compute_hidden_costs, HiddenCostsRequest

load_dotenv()

router = APIRouter()

# ── Expert-defined default weights ───────────────────────────────────────────
WEIGHTS = {
    "financial":      float(os.getenv("WEIGHT_FINANCIAL",      0.30)),
    "goal_alignment": float(os.getenv("WEIGHT_GOAL_ALIGNMENT", 0.25)),
    "household":      float(os.getenv("WEIGHT_HOUSEHOLD",      0.15)),
    "lifestyle":      float(os.getenv("WEIGHT_LIFESTYLE",      0.15)),
    "safety":         float(os.getenv("WEIGHT_SAFETY",         0.10)),
    "stability":      float(os.getenv("WEIGHT_STABILITY",      0.05)),
}


def _get_active_weights(custom_weights: Optional[dict] = None) -> tuple:
    if custom_weights:
        return custom_weights, "custom_override"
    try:
        from ml_models.feedback_learner import get_current_weights
        w = get_current_weights()
        source = w.pop("_source", "expert_default")
        return w, source
    except Exception:
        return WEIGHTS, "expert_default"


def _get_ml_rent_signal(prop: "PropertyData") -> tuple:
    try:
        from ml_models.rent_predictor import predict_rent
        features = {
            "neighbourhood":    prop.neighbourhood,
            "unit_type":        prop.unit_type,
            "infra_zone":       prop.infra_zone,
            "size_m2":          prop.size_m2,
            "num_bedrooms":     prop.num_bedrooms,
            "num_bathrooms":    prop.num_bathrooms,
            "shared_wc":        prop.shared_wc,
            "has_parking":      prop.has_parking,
            "has_generator":    prop.has_generator,
            "has_water_meter":  prop.has_water_meter,
            "floor_level":      prop.floor_level,
            "near_school":      prop.near_school,
            "near_market":      prop.near_market,
            "near_hospital":    prop.near_hospital,
            "structural_quality": prop.structural_quality,
            "flood_risk":       prop.flood_risk,
            "title_type":       prop.title_type,
            "advance_months":   prop.advance_months,
        }
        predicted    = predict_rent(features)
        delta_pct    = (prop.rent - predicted) / predicted * 100 if predicted > 0 else 0
        assessment   = (
            "OVERPRICED"   if delta_pct >  10 else
            "UNDERPRICED"  if delta_pct < -10 else
            "FAIR"
        )
        return predicted, assessment, round(delta_pct, 1)
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PropertyData(BaseModel):
    property_id:         str
    rent:                float = Field(..., gt=0)
    neighbourhood:       str
    unit_type:           str   = Field("T2", description="chambre / T1 / T2 / T3 / T4 / T5")
    infra_zone:          str   = Field("III", description="MINDCAF Zone I–V")
    num_bedrooms:        int   = Field(..., ge=0)
    num_bathrooms:        int   = Field(..., ge=0)
    shared_wc:           bool  = False
    size_m2:             float = Field(..., gt=0)
    has_generator:       bool  = False
    has_parking:         bool  = False
    has_water_meter:     bool  = True
    near_school:         bool  = False
    near_market:         bool  = False
    near_hospital:       bool  = False
    flood_risk:          bool  = False
    structural_quality:  int   = Field(5, ge=1, le=10)
    noise_level:         int   = Field(5, ge=1, le=10)
    landlord_reputation: int   = Field(5, ge=1, le=10)
    lease_security:      int   = Field(5, ge=1, le=10)
    title_type:          str   = Field("occupation", description="foncier / occupation / none")
    advance_months:      int   = Field(3, ge=1)   # 3 is Cameroonian norm
    caution_months:       int   = Field(1, ge=0)
    transport_score:      int   = Field(5, ge=1, le=10)
    floor_level:          int   = Field(0, ge=0)
    has_gardien:          bool  = False


class TenantData(BaseModel):
    tenant_id:              str
    monthly_income:         float = Field(..., gt=0)
    fixed_obligations:      float = 0.0
    savings_goal:           float = 0.0
    goal_timeline_months:   int   = 12
    household_size:         int   = 1
    current_savings:        float = 0.0
    has_dependents:         bool  = False
    needs_parking:          bool  = False
    needs_school_nearby:    bool  = False
    needs_hospital_nearby:  bool  = False
    needs_generator:        bool  = Field(False, description="Tenant specifically requires backup power")
    current_neighbourhood:  Optional[str] = None
    # ── v3 additions: forwarded into financial/profiler.py ──────────────────
    current_city:           Optional[str] = Field(None, description="yaounde | douala | other")
    job_sector:             Optional[str] = Field(
        None, description="formal_private | formal_public | "
                          "informal_self_employed | informal_employee | "
                          "business_owner | student | unemployed | retired"
    )
    income_stability:       Optional[str] = Field(
        None, description="stable | variable | seasonal | irregular"
    )
    custom_weights:         Optional[dict] = Field(None, description="Override weights; must sum to 1.0")


class ScoringRequest(BaseModel):
    tenant:   TenantData
    property: PropertyData

class BulkScoringRequest(BaseModel):
    tenant:     TenantData
    properties: List[PropertyData]

class CategoryScores(BaseModel):
    financial:      float
    goal_alignment: float
    household:      float
    lifestyle:      float
    safety:         float
    stability:      float

class ScoringResponse(BaseModel):
    tenant_id:       str
    property_id:     str
    total_score:     float
    grade:           str
    recommendation:  str
    category_scores: CategoryScores
    weights_used:    dict
    weight_source:   str
    ml_rent_signal:  Optional[dict]
    financial_summary: str
    tco_summary:     str
    flags:           List[str]


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def score_financial(
    profile: dict,
    prop: PropertyData,
    ml_predicted_rent:  Optional[float] = None,
    ml_price_assessment: Optional[str]  = None,
) -> float:
    """
    Financial compatibility score.

    Primary signal: rent / max_sustainable_rent (1/3 income rule).
    Secondary signals:
      - ML rent predictor penalty for overpriced listings
      - Advance payment burden penalty (Cameroonian-specific)
        A property the tenant can't afford the advance for should score lower
        even if the monthly rent is within range.
    """
    max_rent = profile["max_sustainable_rent"]
    rent     = prop.rent
    if max_rent <= 0:
        return 0.0

    ratio = rent / max_rent
    if ratio <= 0.70:
        base_score = 100.0
    elif ratio <= 0.85:
        base_score = 85.0
    elif ratio <= 1.00:
        base_score = 65.0
    elif ratio <= 1.15:
        base_score = 35.0
    elif ratio <= 1.30:
        base_score = 15.0
    else:
        base_score = 0.0

    # ML overpricing penalty (up to 15 points)
    if ml_price_assessment == "OVERPRICED" and ml_predicted_rent is not None:
        overpriced_pct = (prop.rent - ml_predicted_rent) / ml_predicted_rent * 100
        penalty = min(15.0, overpriced_pct * 0.5)
        base_score = max(0.0, base_score - penalty)

    # Advance payment burden penalty — Cameroonian-specific
    # If tenant can't cover the advance, penalise regardless of monthly affordability
    current_savings = profile.get("current_savings", 0)
    advance_needed  = prop.rent * prop.advance_months
    if current_savings < advance_needed:
        shortfall_ratio = (advance_needed - current_savings) / advance_needed
        advance_penalty = min(20.0, shortfall_ratio * 25.0)
        base_score = max(0.0, base_score - advance_penalty)

    return round(base_score, 2)


def score_goal_alignment(profile: dict, prop: PropertyData) -> float:
    true_disposable   = profile["true_disposable"]
    monthly_savings   = profile["monthly_savings_required"]
    remainder         = true_disposable - prop.rent
    if monthly_savings == 0:
        return 80.0
    savings_coverage = remainder / monthly_savings
    if savings_coverage >= 1.5:
        return 100.0
    elif savings_coverage >= 1.0:
        return 80.0
    elif savings_coverage >= 0.7:
        return 50.0
    elif savings_coverage >= 0.4:
        return 25.0
    else:
        return 0.0


def score_household(tenant: TenantData, prop: PropertyData) -> float:
    """
    Household suitability — updated for Cameroonian unit types.
    Shared WC is a significant negative for families with dependents.
    """
    score = 100.0
    ideal_bedrooms = max(1, (tenant.household_size + 1) // 2)

    bedroom_ratio = prop.num_bedrooms / ideal_bedrooms if ideal_bedrooms > 0 else 1.0
    if bedroom_ratio >= 1.0:
        pass
    elif bedroom_ratio >= 0.75:
        score -= 15
    elif bedroom_ratio >= 0.5:
        score -= 35
    else:
        score -= 60

    if prop.num_bathrooms == 0 and not prop.shared_wc:
        score -= 25
    elif prop.num_bathrooms < max(1, tenant.household_size // 3):
        score -= 10

    # Shared WC penalty — significant for families
    if prop.shared_wc:
        if tenant.household_size > 1:
            score -= 25   # families strongly prefer self-contained
        else:
            score -= 10   # singles: moderate inconvenience

    # Bonus: extra bedroom for dependents
    if tenant.has_dependents and prop.num_bedrooms >= ideal_bedrooms + 1:
        score = min(100.0, score + 10)

    return max(0.0, min(100.0, score))


def score_lifestyle(tenant: TenantData, prop: PropertyData) -> float:
    """
    Lifestyle match — generator weighted higher in Cameroonian context.
    """
    score = 65.0   # lower base than before (Cameroon standard units lack many amenities)

    # Generator: critical in Yaoundé/Douala (Eneo load-shedding)
    if tenant.needs_generator and prop.has_generator:
        score += 15
    elif tenant.needs_generator and not prop.has_generator:
        score -= 20
    elif prop.has_generator:
        score += 8    # nice-to-have even when not required

    if tenant.needs_parking and prop.has_parking:
        score += 10
    elif tenant.needs_parking and not prop.has_parking:
        score -= 15

    if tenant.needs_school_nearby and prop.near_school:
        score += 10
    elif tenant.needs_school_nearby and not prop.near_school:
        score -= 15

    if tenant.needs_hospital_nearby and prop.near_hospital:
        score += 10
    elif tenant.needs_hospital_nearby and not prop.near_hospital:
        score -= 12

    if prop.near_market:
        score += 7    # markets are a major convenience in Cameroon
    if prop.has_water_meter:
        score += 5    # individual meter avoids compound disputes

    return max(0.0, min(100.0, score))


def score_safety(prop: PropertyData) -> float:
    """
    Safety score — flood risk weighted heavily (Yaoundé/Douala valley flooding).
    """
    score = 100.0
    if prop.flood_risk:
        score -= 45   # Cameroonian floods destroy belongings and cut roads
    structural_penalty = (10 - prop.structural_quality) * 4
    score -= structural_penalty
    noise_penalty = (prop.noise_level - 1) * 2
    score -= noise_penalty
    return max(0.0, min(100.0, score))


def score_stability(prop: PropertyData) -> float:
    """
    Stability score — title type added as a key Cameroonian dimension.
    foncier (titled) = secure; occupation = common but riskier; none = risky
    """
    score = 0.0
    score += prop.landlord_reputation * 4
    score += prop.lease_security * 4

    # Title type bonus/penalty
    title_type = prop.title_type.lower()
    if title_type == "foncier":
        score += 20   # full land title: eviction risk minimal
    elif title_type == "occupation":
        score += 8    # occupation permit: moderate protection
    else:
        score -= 10   # no title: highest eviction risk

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_optimality_score(tenant: TenantData, prop: PropertyData) -> dict:

    # Step 1: Financial profile
    fin_request = TenantProfileRequest(
        tenant_id=tenant.tenant_id,
        monthly_income=tenant.monthly_income,
        fixed_obligations=tenant.fixed_obligations,
        savings_goal=tenant.savings_goal,
        goal_timeline_months=tenant.goal_timeline_months,
        household_size=tenant.household_size,
        current_savings=tenant.current_savings,
        has_dependents=tenant.has_dependents,
        # v3: forward geography / job side so profiler.py's
        # informal/variable-income emergency-fund adjustment applies here.
        current_city=tenant.current_city,
        current_neighbourhood=tenant.current_neighbourhood,
        job_sector=tenant.job_sector,
        income_stability=tenant.income_stability,
    )
    profile = compute_financial_profile(fin_request)

    # Step 2: Total cost of occupancy
    tco_request = HiddenCostsRequest(
        property_id=prop.property_id,
        rent=prop.rent,
        neighbourhood=prop.neighbourhood,
        has_generator=prop.has_generator,
        has_water_meter=prop.has_water_meter,
        advance_months=prop.advance_months,
        caution_months=prop.caution_months,
        has_gardien=prop.has_gardien,
        tenant_current_neighbourhood=tenant.current_neighbourhood,
        tenant_monthly_income=tenant.monthly_income,
    )
    tco = compute_hidden_costs(tco_request)

    # Step 3: ML rent signal
    ml_predicted_rent, ml_price_assessment, ml_price_delta_pct = _get_ml_rent_signal(prop)
    ml_rent_signal = None
    if ml_predicted_rent is not None:
        ml_rent_signal = {
            "predicted_rent":                  round(ml_predicted_rent, 0),
            "asking_price":                    prop.rent,
            "price_assessment":                ml_price_assessment,
            "price_delta_pct":                 ml_price_delta_pct,
            "signal_used_in_financial_score":  True,
        }

    # Step 4: Category scores
    cat_scores = {
        "financial":      score_financial(
            profile, prop, ml_predicted_rent, ml_price_assessment
        ),
        "goal_alignment": score_goal_alignment(profile, prop),
        "household":      score_household(tenant, prop),
        "lifestyle":      score_lifestyle(tenant, prop),
        "safety":         score_safety(prop),
        "stability":      score_stability(prop),
    }

    # Step 5: Weighted sum
    weights, weight_source = _get_active_weights(tenant.custom_weights)
    total_score = round(sum(
        cat_scores[cat] * weights.get(cat, 0)
        for cat in cat_scores
    ), 2)

    # Step 6: Grade and flags
    if total_score >= 85:
        grade, recommendation = "A", "EXCELLENT"
    elif total_score >= 70:
        grade, recommendation = "B", "GOOD"
    elif total_score >= 55:
        grade, recommendation = "C", "FAIR"
    elif total_score >= 40:
        grade, recommendation = "D", "POOR"
    else:
        grade, recommendation = "F", "REJECT"

    flags = []
    if cat_scores["financial"] < 50:
        flags.append("Rent exceeds your sustainable budget")
    if not profile["can_afford_typical_advance"]:
        flags.append(
            f"Advance payment ({prop.advance_months} months = "
            f"{prop.rent * prop.advance_months:,.0f} CFA) exceeds current savings"
        )
    if cat_scores["goal_alignment"] < 50:
        flags.append("This property may prevent you from meeting your savings goal")
    if cat_scores["safety"] < 50:
        flags.append("Safety concerns detected")
    if prop.flood_risk:
        flags.append("Property is in a flood-risk zone — common during rainy season")
    if prop.shared_wc and tenant.household_size > 1:
        flags.append("Shared WC/bathroom — may not suit family needs")
    if tco["tco_burden"] in ["HIGH", "CRITICAL"]:
        flags.append(
            f"Total cost of occupancy ({tco['breakdown']['total_monthly_cost']:,.0f} CFA/month) "
            f"is {tco['tco_burden'].lower()} relative to income"
        )
    if ml_price_assessment == "OVERPRICED":
        flags.append(
            f"ML model: asking price is overpriced by ~{abs(ml_price_delta_pct):.0f}% "
            f"vs market rate of {ml_predicted_rent:,.0f} CFA"
        )
    if prop.title_type == "none":
        flags.append("Property has no land title — eviction risk higher")
    if weight_source == "learned":
        flags.append("Scoring weights are data-driven (learned from tenant outcomes)")
    if profile.get("profile_confidence") == "LOW":
        flags.append("Tenant financial profile is sparse — score reflects limited information")

    return {
        "tenant_id":         tenant.tenant_id,
        "property_id":       prop.property_id,
        "total_score":       total_score,
        "grade":             grade,
        "recommendation":    recommendation,
        "category_scores":   cat_scores,
        "weights_used":      weights,
        "weight_source":     weight_source,
        "ml_rent_signal":    ml_rent_signal,
        "financial_summary": profile["summary"],
        "tco_summary":       tco["summary"],
        "flags":             flags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoringResponse)
def score_property(data: ScoringRequest):
    try:
        return compute_optimality_score(data.tenant, data.property)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/score/bulk")
def score_properties_bulk(data: BulkScoringRequest):
    try:
        results = [compute_optimality_score(data.tenant, prop) for prop in data.properties]
        results.sort(key=lambda x: x["total_score"], reverse=True)
        return {
            "tenant_id":     data.tenant.tenant_id,
            "total_scored":  len(results),
            "weight_source": results[0]["weight_source"] if results else "unknown",
            "rankings":      results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))