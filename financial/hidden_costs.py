"""
financial/hidden_costs.py
─────────────────────────────────────────────────────────────────────────────
Total Cost of Occupancy — OLISTAY AI Engine
─────────────────────────────────────────────────────────────────────────────
Computes the true monthly cost of renting a property in Cameroon by adding
all recurring costs beyond the listed rent.

Cameroonian-specific cost components:
  - Water (Camwater/CDE): prepaid meter or shared compound bill
  - Electricity (Eneo): prepaid token meter; varies heavily by neighbourhood
  - Generator fuel contribution: if landlord provides generator (common)
  - Building/compound charges (charges de copropriété): varies by building type
  - Transport: estimated monthly commute cost by neighbourhood
  - Caution/advance: one-time upfront cost (3–6 months rent — major barrier)
  - Service charge / gardien: informal security/maintenance in some compounds

The 1/3-of-income affordability rule is the correct Cameroonian benchmark
(applied by the sole social landlord SIC and validated by Sardaouna et al., 2024).

References:
    Sardaouna et al. (2024) — Social housing in Cameroon: state of the affairs
    MINDCAF (2018) — Audit des loyers et locations administratives
    Joint Order No. 0760/MINHDU/MINFI (20 Sept 2024) — Article 4
"""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# ── Neighbourhood cost profiles ───────────────────────────────────────────────
# Water and electricity calibrated to Eneo/Camwater tariff schedules
# and field data from Yaoundé/Douala.
# Transport costs represent estimated monthly moto/taxi/bus spend for a worker
# commuting to central business areas.
NEIGHBOURHOOD_DATA = {
    # Yaoundé prestige
    "bastos": {
        "water":              14000,   # likely individual Camwater meter, higher consumption
        "electricity":        28000,   # Eneo prepaid; large apartments, AC possible
        "generator_contrib":  18000,   # generator fuel share if building has one
        "building_charges":   12000,   # gardiennage + maintenance immeuble
        "transport_score":        8,
        "avg_transport_cost":  8000,   # close to Centre-Ville and Nlongkak
    },
    "omnisport": {
        "water":              10000,
        "electricity":        20000,
        "generator_contrib":  12000,
        "building_charges":    7000,
        "transport_score":        7,
        "avg_transport_cost": 10000,
    },
    # Yaoundé mid-tier
    "nlongkak": {
        "water":               8000,
        "electricity":        14000,
        "generator_contrib":   9000,
        "building_charges":    4000,
        "transport_score":        7,
        "avg_transport_cost": 12000,
    },
    "biyem_assi": {
        "water":               7000,
        "electricity":        13000,
        "generator_contrib":   8000,
        "building_charges":    4000,
        "transport_score":        6,
        "avg_transport_cost": 14000,
    },
    "essos": {
        "water":               6500,
        "electricity":        11000,
        "generator_contrib":   7500,
        "building_charges":    3000,
        "transport_score":        6,
        "avg_transport_cost": 13000,
    },
    "mvog_mbi": {
        "water":               6000,
        "electricity":        10000,
        "generator_contrib":   7000,
        "building_charges":    2500,
        "transport_score":        6,
        "avg_transport_cost": 12000,
    },
    # Yaoundé popular/peripheral
    "melen": {
        "water":               5000,   # often shared compound tap
        "electricity":         9000,
        "generator_contrib":   6000,
        "building_charges":    2000,
        "transport_score":        5,
        "avg_transport_cost": 18000,
    },
    "nkol_foulou": {
        "water":               4500,
        "electricity":         8500,
        "generator_contrib":   5500,
        "building_charges":    1500,
        "transport_score":        4,
        "avg_transport_cost": 20000,
    },
    "odza": {
        "water":               4500,
        "electricity":         8000,
        "generator_contrib":   5000,
        "building_charges":    1500,
        "transport_score":        4,
        "avg_transport_cost": 22000,
    },
    # Douala prestige
    "bonanjo": {
        "water":              12000,
        "electricity":        24000,
        "generator_contrib":  15000,
        "building_charges":   10000,
        "transport_score":        8,
        "avg_transport_cost":  9000,
    },
    "bonapriso": {
        "water":              13000,
        "electricity":        26000,
        "generator_contrib":  16000,
        "building_charges":   11000,
        "transport_score":        8,
        "avg_transport_cost":  9000,
    },
    "akwa": {
        "water":               9000,
        "electricity":        18000,
        "generator_contrib":  11000,
        "building_charges":    6000,
        "transport_score":        7,
        "avg_transport_cost": 10000,
    },
    # Douala mid-tier
    "makepe": {
        "water":               6000,
        "electricity":        11000,
        "generator_contrib":   7500,
        "building_charges":    3000,
        "transport_score":        5,
        "avg_transport_cost": 16000,
    },
    "logpom": {
        "water":               6000,
        "electricity":        10500,
        "generator_contrib":   7000,
        "building_charges":    3000,
        "transport_score":        5,
        "avg_transport_cost": 15000,
    },
    # Douala peripheral
    "yassa": {
        "water":               5000,
        "electricity":         9000,
        "generator_contrib":   6000,
        "building_charges":    2000,
        "transport_score":        4,
        "avg_transport_cost": 22000,
    },
    "bonaberi": {
        "water":               5000,
        "electricity":         8500,
        "generator_contrib":   5500,
        "building_charges":    2000,
        "transport_score":        4,
        "avg_transport_cost": 24000,
    },
    # Fallback for unknown neighbourhoods
    "default": {
        "water":               6500,
        "electricity":        12000,
        "generator_contrib":   8000,
        "building_charges":    3000,
        "transport_score":        5,
        "avg_transport_cost": 16000,
    },
}


class HiddenCostsRequest(BaseModel):
    property_id:                   str
    rent:                          float = Field(..., gt=0, description="Monthly rent in CFA")
    neighbourhood:                 str
    has_generator:                 bool = Field(False, description="Property has backup generator")
    has_water_meter:               bool = Field(True,  description="Individual Camwater meter")
    advance_months:                int  = Field(3, ge=1, description="Months of advance payment required")
    caution_months:                int  = Field(1, ge=0, description="Security deposit months (caution)")
    has_gardien:                   bool = Field(False, description="Compound has a gardien/watchman")
    tenant_current_neighbourhood:  Optional[str] = Field(
        None, description="Tenant's current neighbourhood for transport delta"
    )
    tenant_monthly_income:         float = Field(..., gt=0)


class CostBreakdown(BaseModel):
    rent:                float
    water:               float
    electricity:         float
    generator_contrib:   float
    building_charges:    float
    gardien_contrib:     float
    transport:           float
    transport_delta:     float
    total_monthly_cost:  float
    advance_payment:     float   # advance_months × rent (upfront)
    caution_payment:     float   # caution_months × rent (refundable deposit)
    total_upfront_cost:  float   # advance + caution


class HiddenCostsResponse(BaseModel):
    property_id:           str
    neighbourhood:         str
    breakdown:             CostBreakdown
    tco_to_income_ratio:   float
    tco_burden:            str
    advance_months:        int
    caution_months:        int
    neighbourhood_found:   bool
    summary:               str


def compute_hidden_costs(data: HiddenCostsRequest) -> dict:
    key = data.neighbourhood.lower().replace(" ", "_").replace("-", "_")
    neighbourhood_found = key in NEIGHBOURHOOD_DATA
    costs = NEIGHBOURHOOD_DATA.get(key, NEIGHBOURHOOD_DATA["default"])

    # Water: shared compound tap is ~20% cheaper per unit than individual meter
    water = costs["water"] if data.has_water_meter else costs["water"] * 0.8

    electricity = costs["electricity"]

    # Generator: only charge if the property actually has one
    generator_contrib = costs["generator_contrib"] if data.has_generator else 0.0

    building_charges = costs["building_charges"]

    # Gardien/watchman contribution — common in mid-to-upper compounds
    gardien_contrib = 3000.0 if data.has_gardien else 0.0

    transport = costs["avg_transport_cost"]
    transport_delta = 0.0
    if data.tenant_current_neighbourhood:
        current_key = data.tenant_current_neighbourhood.lower().replace(" ", "_").replace("-", "_")
        current_costs = NEIGHBOURHOOD_DATA.get(current_key, NEIGHBOURHOOD_DATA["default"])
        transport_delta = transport - current_costs["avg_transport_cost"]

    total_monthly = (
        data.rent + water + electricity +
        generator_contrib + building_charges + gardien_contrib + transport
    )

    # Upfront costs — critical in Cameroon; major access barrier
    advance_payment = data.rent * data.advance_months     # non-refundable monthly charges
    caution_payment = data.rent * data.caution_months     # refundable security deposit
    total_upfront   = advance_payment + caution_payment

    # Affordability: SIC/Sardaouna benchmark = max 1/3 of income on housing
    # We use total monthly cost (not just rent) for honest burden calculation
    tco_ratio = total_monthly / data.tenant_monthly_income

    if tco_ratio <= 0.33:
        tco_burden = "LOW"       # within SIC/UN-Habitat 1/3 rule
    elif tco_ratio <= 0.45:
        tco_burden = "MODERATE"  # manageable with discipline
    elif tco_ratio <= 0.60:
        tco_burden = "HIGH"      # stretches budget significantly
    else:
        tco_burden = "CRITICAL"  # unsustainable

    summary = (
        f"Listed rent of {data.rent:,.0f} CFA becomes a total monthly cost of "
        f"{total_monthly:,.0f} CFA once utilities, transport, and building charges are included. "
        f"This is {tco_ratio*100:.1f}% of your income — burden: {tco_burden}. "
        f"You will also need {total_upfront:,.0f} CFA upfront "
        f"({advance_payment:,.0f} CFA advance + {caution_payment:,.0f} CFA caution)."
    )

    return {
        "property_id":         data.property_id,
        "neighbourhood":       data.neighbourhood,
        "breakdown": {
            "rent":               round(data.rent, 2),
            "water":              round(water, 2),
            "electricity":        round(electricity, 2),
            "generator_contrib":  round(generator_contrib, 2),
            "building_charges":   round(building_charges, 2),
            "gardien_contrib":    round(gardien_contrib, 2),
            "transport":          round(transport, 2),
            "transport_delta":    round(transport_delta, 2),
            "total_monthly_cost": round(total_monthly, 2),
            "advance_payment":    round(advance_payment, 2),
            "caution_payment":    round(caution_payment, 2),
            "total_upfront_cost": round(total_upfront, 2),
        },
        "tco_to_income_ratio":  round(tco_ratio, 4),
        "tco_burden":           tco_burden,
        "advance_months":       data.advance_months,
        "caution_months":       data.caution_months,
        "neighbourhood_found":  neighbourhood_found,
        "summary":              summary,
    }


@router.post("/hidden-costs", response_model=HiddenCostsResponse)
def calculate_hidden_costs(data: HiddenCostsRequest):
    try:
        return compute_hidden_costs(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neighbourhoods")
def list_neighbourhoods():
    return {
        "neighbourhoods": [k for k in NEIGHBOURHOOD_DATA.keys() if k != "default"],
        "count":          len(NEIGHBOURHOOD_DATA) - 1,
    }