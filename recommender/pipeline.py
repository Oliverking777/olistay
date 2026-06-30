"""
recommender/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Hybrid Recommendation Pipeline — OLISTAY AI Engine
─────────────────────────────────────────────────────────────────────────────
Four-stage hybrid recommender:
  Stage 1 — Knowledge-based hard filter  (affordability / advance gates)
  Stage 2 — Content-based cosine similarity ranking
  Stage 3 — Collaborative filtering augmentation (cold-start safe)
  Stage 4 — Optimality scoring + LLM narration (delegated to /scoring, /narrator)

v3 update:
  - PipelineTenant now carries job_sector / current_city / income_stability,
    forwarded into the financial profiler (financial/profiler.py v3) so the
    informal/variable-income emergency-fund adjustment applies pipeline-wide,
    not just on the standalone /financial/profile endpoint. All three remain
    optional — existing callers are unaffected.
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

from financial.profiler import compute_financial_profile, TenantProfileRequest
from recommender.content_based import rank_properties_content_based
from recommender.collaborative import compute_collaborative_scores, get_store

load_dotenv()

router = APIRouter()

# ── Knowledge-based hard-filter gates ───────────────────────────────────────
HARD_RENT_CEILING       = float(os.getenv("HARD_RENT_CEILING_MULT",       1.35))
HARD_ADVANCE_CEILING     = float(os.getenv("HARD_ADVANCE_CEILING_MULT",    2.0))
CF_AUGMENTATION_WEIGHT   = float(os.getenv("CF_AUGMENTATION_WEIGHT",      0.20))


class PipelineTenant(BaseModel):
    tenant_id:             str
    monthly_income:        float = Field(..., gt=0)
    fixed_obligations:      float = 0.0
    savings_goal:           float = 0.0
    goal_timeline_months:   int   = 12
    household_size:         int   = 1
    current_savings:        float = 0.0
    has_dependents:         bool  = False
    needs_parking:          bool  = False
    needs_school_nearby:    bool  = False
    needs_hospital_nearby:  bool  = False
    needs_generator:        bool  = False
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


class RecommendRequest(BaseModel):
    tenant:                 PipelineTenant
    candidate_properties:   List[dict]
    top_n:                  int = Field(10, gt=0, le=50)


class RecommendResponse(BaseModel):
    tenant_id:               str
    stage1_candidates_in:    int
    stage1_candidates_out:   int
    stage1_rejected:         List[dict]
    pipeline_method:         str
    cold_start_active:       bool
    cf_augmentation_weight:  float
    top_n:                   int
    recommendations:         List[dict]


def apply_knowledge_filter(tenant: PipelineTenant, properties: List[dict], profile: dict) -> tuple:
    """
    Stage 1: knowledge-based hard filter.
    Rejects properties that are clearly unaffordable, regardless of how well
    they might score on other dimensions. Two gates:
      Gate 1 — rent must not exceed HARD_RENT_CEILING × max_sustainable_rent
      Gate 2 — advance payment must not exceed HARD_ADVANCE_CEILING × current_savings
    """
    max_rent         = profile["max_sustainable_rent"]
    current_savings  = profile.get("current_savings", 0)

    accepted, rejected = [], []
    for prop in properties:
        prop_rent = prop.get("rent", 0)
        advance_months = prop.get("advance_months", 3)
        advance_needed = prop_rent * advance_months

        # Gate 1: rent ceiling
        if max_rent > 0 and prop_rent > max_rent * HARD_RENT_CEILING:
            over_pct = (prop_rent / max_rent - 1) * 100
            rejected.append({
                "property_id": prop.get("property_id"),
                "reason": f"Rent exceeds affordability ceiling by {over_pct:.0f}%"
            })
            continue
        elif max_rent <= 0:
            rejected.append({
                "property_id": prop.get("property_id"),
                "reason": "Tenant has no sustainable rent budget (income ≤ obligations)"
            })
            continue

        # Gate 2: advance payment ceiling
        if current_savings > 0 and advance_needed > current_savings * HARD_ADVANCE_CEILING:
            rejected.append({
                "property_id": prop.get("property_id"),
                "reason": (
                    f"Advance payment ({advance_needed:,.0f} CFA) far exceeds "
                    f"current savings ({current_savings:,.0f} CFA)"
                )
            })
            continue

        accepted.append(prop)

    return accepted, rejected


def run_pipeline(tenant: PipelineTenant, candidate_properties: List[dict], top_n: int = 10) -> dict:

    # ── Stage 0: financial profile (drives Stage 1 + content-based ranking) ─
    fin_req = TenantProfileRequest(
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
    profile = compute_financial_profile(fin_req)

    # ── Stage 1: knowledge-based hard filter ────────────────────────────────
    accepted, rejected = apply_knowledge_filter(tenant, candidate_properties, profile)

    if not accepted:
        return {
            "tenant_id":               tenant.tenant_id,
            "stage1_candidates_in":    len(candidate_properties),
            "stage1_candidates_out":   0,
            "stage1_rejected":         rejected,
            "pipeline_method":         "knowledge_filter_only",
            "cold_start_active":       get_store().is_cold_start,
            "cf_augmentation_weight":  CF_AUGMENTATION_WEIGHT,
            "top_n":                   top_n,
            "recommendations":         [],
        }

    # ── Stage 2: content-based ranking ──────────────────────────────────────
    tenant_dict = tenant.dict()
    ranked = rank_properties_content_based(tenant_dict, profile, accepted)

    # ── Stage 3: collaborative filtering augmentation ───────────────────────
    store = get_store()
    candidate_ids = [p.get("property_id") for p in ranked]
    cf_scores = compute_collaborative_scores(tenant.tenant_id, candidate_ids, store)

    for prop in ranked:
        pid = prop.get("property_id")
        content_score = prop["content_similarity_score"]
        cf_score      = cf_scores.get(pid, 0.5)
        prop["collaborative_score"] = cf_score
        prop["hybrid_score"] = round(
            content_score * (1 - CF_AUGMENTATION_WEIGHT) +
            cf_score * CF_AUGMENTATION_WEIGHT, 4
        )

    ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return {
        "tenant_id":               tenant.tenant_id,
        "stage1_candidates_in":    len(candidate_properties),
        "stage1_candidates_out":   len(accepted),
        "stage1_rejected":         rejected,
        "pipeline_method":         "cold_start_content_only" if store.is_cold_start else "hybrid_content_collaborative",
        "cold_start_active":       store.is_cold_start,
        "cf_augmentation_weight":  CF_AUGMENTATION_WEIGHT,
        "top_n":                   top_n,
        "recommendations":         ranked[:top_n],
    }


@router.post("/recommend", response_model=RecommendResponse)
def recommend(data: RecommendRequest):
    try:
        return run_pipeline(data.tenant, data.candidate_properties, data.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline/info")
def pipeline_info():
    store = get_store()
    return {
        "stages": {
            "1_knowledge_filter":  {
                "hard_rent_ceiling_mult":     HARD_RENT_CEILING,
                "hard_advance_ceiling_mult":  HARD_ADVANCE_CEILING,
            },
            "2_content_based":     "active — 19-dimensional cosine similarity",
            "3_collaborative":     store.stats(),
            "4_optimality_scoring": "delegated to /scoring/score",
        },
        "cf_augmentation_weight": CF_AUGMENTATION_WEIGHT,
    }