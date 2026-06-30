"""
recommender/content_based.py
─────────────────────────────────────────────────────────────────────────────
Content-Based Recommendation Engine — OLISTAY AI Engine
─────────────────────────────────────────────────────────────────────────────
Cameroonian market updates:
  - Generator added as a strong positive feature (not just "nice to have")
  - shared_wc added as negative feature (significant in Cameroon market)
  - title_type_norm added (foncier > occupation > none security gradient)
  - infra_zone_norm added (Zone I–V MINDCAF classification)
  - Advance affordability weighted higher (primary access barrier)
  - Market proximity preference raised (marchés are daily-use in Cameroon)
  - Flood risk retained as strong penalty (rainy season impact is severe)

References:
    Lops, de Gemmis and Semeraro (2011) — Recommender Systems Handbook
    MINDCAF (2018) — Audit des loyers et locations administratives
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

PROPERTY_FEATURE_NAMES = [
    "rent_normalised",          # 1 - (rent / max_sustainable_rent); higher = more affordable
    "size_per_person",          # size_m2 / (household_size × 25); capped at 1
    "bedroom_ratio",            # num_bedrooms / ideal_bedrooms; capped at 1
    "bathroom_adequacy",        # scaled bathroom sufficiency
    "no_shared_wc",             # 1 if self-contained (no shared toilet) — POSITIVE feature
    "has_generator",            # binary — weighted heavily in Cameroon
    "has_water_meter",          # individual Camwater meter
    "has_parking",              # binary
    "near_school",              # binary
    "near_market",              # binary — high daily value in Cameroon
    "near_hospital",            # binary
    "structural_quality_norm",  # structural_quality / 10
    "infra_zone_norm",          # Zone I–V → 1.0–0.0 (I=best)
    "title_security_norm",      # foncier=1.0, occupation=0.5, none=0.0
    "safety_norm",              # composite: structural + flood risk
    "stability_norm",           # landlord_reputation + lease_security
    "transport_score_norm",     # transport_score / 10
    "no_flood_risk",            # 1 if no flood risk (positive framing)
    "advance_affordability",    # 1 if current savings ≥ advance payment needed
]

TENANT_PREFERENCE_NAMES = PROPERTY_FEATURE_NAMES

INFRA_ZONE_VALUES = {"I": 1.0, "II": 0.8, "III": 0.6, "IV": 0.4, "V": 0.2}
TITLE_SECURITY_VALUES = {"foncier": 1.0, "occupation": 0.5, "none": 0.0}


def _safe_divide(num: float, den: float, fallback: float = 0.0) -> float:
    return num / den if den != 0 else fallback


def encode_property_vector(
    prop: dict,
    max_sustainable_rent: float,
    household_size: int,
    current_savings: float,
) -> np.ndarray:
    rent          = prop.get("rent", 0)
    ideal_bedrooms = max(1, (household_size + 1) // 2)
    structural_q  = prop.get("structural_quality", 5)
    flood_risk    = prop.get("flood_risk", False)
    noise         = prop.get("noise_level", 5)
    shared_wc     = prop.get("shared_wc", False)

    # Safety composite (flood risk carries heavy penalty)
    safety_score = max(0.0, 100.0
                       - (45.0 if flood_risk else 0.0)
                       - (10 - structural_q) * 4.0
                       - (noise - 1) * 2.0) / 100.0

    # Stability composite
    landlord_rep  = prop.get("landlord_reputation", 5)
    lease_sec     = prop.get("lease_security", 5)
    stability_score = ((landlord_rep * 5) + (lease_sec * 5)) / 100.0

    # Infrastructure zone
    infra_zone    = str(prop.get("infra_zone", "III")).upper()
    infra_norm    = INFRA_ZONE_VALUES.get(infra_zone, 0.6)

    # Title security
    title_type    = str(prop.get("title_type", "occupation")).lower()
    title_norm    = TITLE_SECURITY_VALUES.get(title_type, 0.5)

    # Advance affordability
    advance_months  = prop.get("advance_months", 3)
    advance_needed  = rent * advance_months
    advance_ok      = 1.0 if current_savings >= advance_needed else max(
        0.0, current_savings / advance_needed if advance_needed > 0 else 0.0
    )

    vector = np.array([
        # Affordability: inverted rent ratio (lower rent relative to budget = higher score)
        1.0 - min(_safe_divide(rent, max_sustainable_rent, 2.0), 1.0),
        # Space
        min(_safe_divide(prop.get("size_m2", 50), household_size * 25), 1.0),
        # Rooms
        min(_safe_divide(prop.get("num_bedrooms", 1), ideal_bedrooms), 1.0),
        min(_safe_divide(prop.get("num_bathrooms", 1), max(1, household_size // 3)), 1.0),
        # Facilities — positively framed
        0.0 if shared_wc else 1.0,                       # no_shared_wc
        1.0 if prop.get("has_generator", False) else 0.0,
        1.0 if prop.get("has_water_meter", True) else 0.0,
        1.0 if prop.get("has_parking", False) else 0.0,
        # Proximity
        1.0 if prop.get("near_school",   False) else 0.0,
        1.0 if prop.get("near_market",   False) else 0.0,
        1.0 if prop.get("near_hospital", False) else 0.0,
        # Quality
        structural_q / 10.0,
        infra_norm,
        title_norm,
        safety_score,
        stability_score,
        _safe_divide(prop.get("transport_score") or 5, 10.0),
        # Risk (positive framing: 1 = no flood risk)
        0.0 if flood_risk else 1.0,
        # Access
        advance_ok,
    ], dtype=np.float64)

    return vector


def encode_tenant_preference_vector(
    tenant: dict,
    financial_profile: dict,
) -> np.ndarray:
    """
    Encode the tenant's ideal property profile as a preference vector.
    Cameroonian-calibrated: generator is weighted near-essential,
    market proximity is highly valued, shared WC is strongly avoided.
    """
    household_size = tenant.get("household_size", 1)

    # Rent: prefer well within budget (0.85 = comfortable buffer below max)
    rent_pref     = 0.85

    # Space
    size_pref     = 1.0
    bedroom_pref  = 1.0
    bathroom_pref = 1.0

    # Self-contained: all tenants prefer no shared WC
    # Stronger preference for families
    no_shared_wc_pref = 1.0 if household_size > 1 else 0.85

    # Generator: near-essential in Yaoundé/Douala (load-shedding context)
    needs_generator = tenant.get("needs_generator", False)
    generator_pref = 1.0 if needs_generator else 0.75  # high preference even if not required

    # Individual water meter: preferred by most tenants
    water_meter_pref = 0.8

    # Parking: depends on tenant
    parking_pref = 1.0 if tenant.get("needs_parking", False) else 0.4

    # Proximity preferences
    school_pref   = 1.0 if tenant.get("needs_school_nearby",   False) else 0.5
    market_pref   = 0.85  # markets are daily-use in Cameroon — high universal preference
    hospital_pref = 1.0 if tenant.get("needs_hospital_nearby", False) else 0.5

    # Quality preferences — all tenants want decent quality
    structural_pref = 0.8
    infra_pref      = 0.75  # prefer Zone II–III minimum
    title_pref      = 0.8   # prefer occupation or foncier
    safety_pref     = 1.0
    stability_pref  = 0.9
    transport_pref  = 0.8

    # No flood risk: universal strong preference
    no_flood_pref   = 1.0

    # Advance: can the tenant afford it?
    advance_pref    = 1.0

    vector = np.array([
        rent_pref,
        size_pref,
        bedroom_pref,
        bathroom_pref,
        no_shared_wc_pref,
        generator_pref,
        water_meter_pref,
        parking_pref,
        school_pref,
        market_pref,
        hospital_pref,
        structural_pref,
        infra_pref,
        title_pref,
        safety_pref,
        stability_pref,
        transport_pref,
        no_flood_pref,
        advance_pref,
    ], dtype=np.float64)

    return vector


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), 0.0, 1.0))


def rank_properties_content_based(
    tenant: dict,
    financial_profile: dict,
    candidate_properties: List[dict],
) -> List[dict]:
    """
    Rank candidate properties by cosine similarity to the tenant's preference vector.
    """
    max_sustainable_rent = financial_profile.get("max_sustainable_rent", 1.0)
    current_savings      = financial_profile.get("current_savings",
                                                  tenant.get("current_savings", 0.0))
    household_size       = tenant.get("household_size", 1)

    tenant_vec = encode_tenant_preference_vector(tenant, financial_profile)

    ranked = []
    for prop in candidate_properties:
        prop_vec   = encode_property_vector(
            prop,
            max_sustainable_rent=max_sustainable_rent,
            household_size=household_size,
            current_savings=current_savings,
        )
        similarity = cosine_similarity(tenant_vec, prop_vec)
        contributions = {
            name: round(float(tenant_vec[i] * prop_vec[i]), 4)
            for i, name in enumerate(PROPERTY_FEATURE_NAMES)
        }
        ranked.append({
            **prop,
            "content_similarity_score": round(similarity, 4),
            "property_vector":          prop_vec.tolist(),
            "feature_contributions":    contributions,
        })

    ranked.sort(key=lambda x: x["content_similarity_score"], reverse=True)
    return ranked