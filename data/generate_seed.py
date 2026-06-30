"""
generate_seed.py — OLISTAY Rent Seed Data Generator (v2)
─────────────────────────────────────────────────────────────────────────────
Produces a rich synthetic training set calibrated to the Cameroonian urban
rental market (Yaoundé & Douala).

Key upgrades over v1:
  - 8 property types: house, apartment, studio, land, office, shop,
    store, warehouse — each with its own rent logic and feature set
  - Dimensions encoded as length_m × width_m (explicit plot/floor footprint)
  - GPS latitude/longitude per neighbourhood (model learns geo-gradient)
  - 30+ features total: type-specific amenities, build year, condition,
    road frontage, internet_fiber, security_gate, standby_power, etc.
  - 1 200 rows (doubled) for better distributional coverage
  - SIC rent floors enforced; MINDCAF Zone I–V multipliers applied

References:
    MINDCAF (2018) — Audit des loyers et locations administratives
    Sardaouna et al. (2024) — Social housing in Cameroon
    Joint Order No. 0760/MINHDU/MINFI (20 Sept 2024) — Article 4
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

# ─────────────────────────────────────────────────────────────────────────────
# NEIGHBOURHOOD TABLE — base rents (CFA) + GPS centroids
# Base rents represent a mid-size residential unit; commercial multipliers
# are applied later per property type.
# ─────────────────────────────────────────────────────────────────────────────
NEIGHBOURHOODS = {
    # name:  (base_rent, lat, lon, city)
    # Yaoundé — prestige
    "bastos":        (220000,  3.8790,  11.5120, "yaounde"),
    "omnisport":     (160000,  3.8700,  11.5200, "yaounde"),
    # Yaoundé — mid-tier
    "nlongkak":      (110000,  3.8650,  11.5100, "yaounde"),
    "biyem_assi":    ( 95000,  3.8500,  11.4900, "yaounde"),
    "essos":         ( 85000,  3.8800,  11.5300, "yaounde"),
    "mvog_mbi":      ( 75000,  3.8550,  11.5050, "yaounde"),
    "ngousso":       ( 90000,  3.8950,  11.5350, "yaounde"),
    "nkoldongo":     ( 70000,  3.8300,  11.4850, "yaounde"),
    # Yaoundé — popular/peripheral
    "melen":         ( 60000,  3.9100,  11.5400, "yaounde"),
    "nkol_foulou":   ( 55000,  3.8200,  11.4800, "yaounde"),
    "odza":          ( 50000,  3.9300,  11.5600, "yaounde"),
    "olembe":        ( 58000,  3.9500,  11.4950, "yaounde"),
    "ekounou":       ( 52000,  3.8100,  11.5500, "yaounde"),
    # Douala — prestige
    "bonanjo":       (190000,  4.0450,  9.6950,  "douala"),
    "bonapriso":     (210000,  4.0350,  9.6850,  "douala"),
    "akwa":          (140000,  4.0480,  9.7000,  "douala"),
    "deido":         (120000,  4.0600,  9.7100,  "douala"),
    # Douala — mid-tier
    "makepe":        ( 80000,  4.0750,  9.7350,  "douala"),
    "logpom":        ( 70000,  4.0800,  9.7200,  "douala"),
    "cite_sic":      ( 75000,  4.0650,  9.7250,  "douala"),
    "ndogpassi":     ( 65000,  4.0900,  9.7450,  "douala"),
    # Douala — peripheral
    "yassa":         ( 55000,  4.0300,  9.7800,  "douala"),
    "bonaberi":      ( 50000,  4.0550,  9.6600,  "douala"),
    "nyalla":        ( 58000,  4.1000,  9.7600,  "douala"),
}

NEIGHBOURHOOD_KEYS = list(NEIGHBOURHOODS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE ZONE MULTIPLIERS (MINDCAF Zone I–V)
# ─────────────────────────────────────────────────────────────────────────────
ZONE_MULTIPLIERS = {"I": 1.28, "II": 1.12, "III": 1.00, "IV": 0.84, "V": 0.68}
ZONE_PROBS       = [0.08, 0.20, 0.36, 0.24, 0.12]

# ─────────────────────────────────────────────────────────────────────────────
# PROPERTY TYPES — configuration per type
#
# base_mult     : multiplier applied to neighbourhood base rent
# size_range    : (length_min, length_max, width_min, width_max) in metres
# sic_min_rent  : SIC minimum monthly rent floor (CFA)
# has_bedrooms  : whether the type has sleeping rooms
# has_bathrooms : whether the type has bathrooms
# commercial    : True → adds road frontage, shopfront_quality etc.
# ─────────────────────────────────────────────────────────────────────────────
PROPERTY_TYPES = {
    "house": {
        "base_mult": 1.20,
        "length_range": (6,  22),
        "width_range":  (5,  18),
        "sic_min_rent": 30000,
        "has_bedrooms": True,
        "has_bathrooms": True,
        "commercial": False,
    },
    "apartment": {
        "base_mult": 1.00,
        "length_range": (5,  16),
        "width_range":  (4,  12),
        "sic_min_rent": 24685,
        "has_bedrooms": True,
        "has_bathrooms": True,
        "commercial": False,
    },
    "studio": {
        "base_mult": 0.65,
        "length_range": (4,  9),
        "width_range":  (3,  7),
        "sic_min_rent": 13100,
        "has_bedrooms": True,   # 1 bedroom or open-plan
        "has_bathrooms": True,
        "commercial": False,
    },
    "land": {
        "base_mult": 0.40,      # land rent is typically much lower
        "length_range": (8,  60),
        "width_range":  (6,  50),
        "sic_min_rent": 10000,
        "has_bedrooms": False,
        "has_bathrooms": False,
        "commercial": False,
    },
    "office": {
        "base_mult": 2.00,
        "length_range": (6,  30),
        "width_range":  (5,  20),
        "sic_min_rent": 50000,
        "has_bedrooms": False,
        "has_bathrooms": True,
        "commercial": True,
    },
    "shop": {
        "base_mult": 1.80,
        "length_range": (3,  15),
        "width_range":  (3,  10),
        "sic_min_rent": 30000,
        "has_bedrooms": False,
        "has_bathrooms": False,
        "commercial": True,
    },
    "store": {
        "base_mult": 1.50,
        "length_range": (6,  25),
        "width_range":  (5,  20),
        "sic_min_rent": 40000,
        "has_bedrooms": False,
        "has_bathrooms": False,
        "commercial": True,
    },
    "warehouse": {
        "base_mult": 1.20,
        "length_range": (12, 60),
        "width_range":  (10, 40),
        "sic_min_rent": 60000,
        "has_bedrooms": False,
        "has_bathrooms": False,
        "commercial": True,
    },
}

# Sampling weights — residential is dominant in Cameroon market
PROPERTY_TYPE_WEIGHTS = [0.30, 0.25, 0.12, 0.08, 0.08, 0.07, 0.06, 0.04]
PROPERTY_TYPE_KEYS    = list(PROPERTY_TYPES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# BEDROOM DISTRIBUTIONS BY PROPERTY TYPE
# apartment = 2+ rooms by definition
# ─────────────────────────────────────────────────────────────────────────────
BEDROOM_DIST = {
    "house":     ([2, 3, 4, 5, 6],   [0.15, 0.35, 0.30, 0.15, 0.05]),
    "apartment": ([2, 3, 4, 5],       [0.40, 0.35, 0.18, 0.07]),
    "studio":    ([1],                 [1.00]),
    "land":      ([0],                 [1.00]),
    "office":    ([0],                 [1.00]),
    "shop":      ([0],                 [1.00]),
    "store":     ([0],                 [1.00]),
    "warehouse": ([0],                 [1.00]),
}

BATHROOM_DIST = {
    "house":     ([1, 2, 3],          [0.35, 0.45, 0.20]),
    "apartment": ([1, 2],             [0.55, 0.45]),
    "studio":    ([1],                 [1.00]),
    "land":      ([0],                 [1.00]),
    "office":    ([1, 2],             [0.65, 0.35]),
    "shop":      ([0, 1],             [0.70, 0.30]),
    "store":     ([0, 1],             [0.60, 0.40]),
    "warehouse": ([0, 1],             [0.80, 0.20]),
}

FLOOR_DIST = {
    "house":     ([0],                 [1.00]),   # detached houses are ground-level
    "apartment": ([1, 2, 3, 4, 5],   [0.25, 0.30, 0.22, 0.15, 0.08]),
    "studio":    ([0, 1, 2, 3],       [0.30, 0.35, 0.25, 0.10]),
    "office":    ([0, 1, 2, 3, 4, 5],[0.15, 0.25, 0.25, 0.20, 0.10, 0.05]),
    "shop":      ([0, 1],             [0.80, 0.20]),
    "store":     ([0],                 [1.00]),
    "warehouse": ([0],                 [1.00]),
    "land":      ([0],                 [1.00]),
}

ADVANCE_OPTIONS = [1, 2, 3, 3, 3, 6, 6]

# ─────────────────────────────────────────────────────────────────────────────
# RENT FORMULA COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rent(hood_base, prop_type_cfg, prop_type, length_m, width_m,
                 num_bedrooms, num_bathrooms, shared_wc, has_parking,
                 has_generator, has_water_meter, floor_level, near_school,
                 near_market, near_hospital, near_highway, near_university,
                 structural_quality, build_year, condition_score,
                 flood_risk, noise_level, road_frontage_m,
                 shopfront_quality, fiber_internet, security_gate,
                 standby_power_kva, loading_bay, title_type,
                 advance_months, infra_zone, zone_mult, rng):

    area_m2 = length_m * width_m
    base     = hood_base * prop_type_cfg["base_mult"]

    rent = base  # start from neighbourhood×type base

    # ── Area premium ─────────────────────────────────────────────────────────
    # Reference area per type; each m² above reference adds value
    ref_areas = {
        "house": 80, "apartment": 60, "studio": 28,
        "land": 200, "office": 60, "shop": 25, "store": 80, "warehouse": 200
    }
    ref_area  = ref_areas.get(prop_type, 60)
    area_rate = {
        "house": 700, "apartment": 900, "studio": 1200,
        "land": 150, "office": 2500, "shop": 3000, "store": 1800, "warehouse": 600
    }.get(prop_type, 700)
    rent += (area_m2 - ref_area) * area_rate

    # ── Bedrooms / bathrooms ─────────────────────────────────────────────────
    rent += num_bedrooms  * 18000
    rent += num_bathrooms * 8000

    # ── Residential amenities ────────────────────────────────────────────────
    if prop_type in ("house", "apartment", "studio"):
        if shared_wc:
            rent -= 28000
        if has_water_meter:
            rent += 5000
        rent += floor_level * 4000  # upper floor premium in apartments

    # ── Universal amenities ──────────────────────────────────────────────────
    if has_parking:
        parking_val = {"house": 10000, "apartment": 15000, "office": 25000,
                       "shop": 12000, "store": 10000, "warehouse": 8000}.get(prop_type, 10000)
        rent += parking_val
    if has_generator:
        gen_val = {"house": 15000, "apartment": 15000, "office": 35000,
                   "shop": 20000, "store": 18000, "warehouse": 25000}.get(prop_type, 15000)
        rent += gen_val
    if standby_power_kva > 0 and prop_type in ("office", "warehouse"):
        rent += standby_power_kva * 3000   # commercial clients pay per kVA

    # ── Proximity (residential) ──────────────────────────────────────────────
    if prop_type in ("house", "apartment", "studio"):
        rent += near_school   * 7000
        rent += near_market   * 5000
        rent += near_hospital * 6000
        rent += near_university * 10000   # university proximity significant in Cameroon
    rent += near_highway * 8000  # highway access valued by all types

    # ── Commercial-specific ──────────────────────────────────────────────────
    if prop_type_cfg["commercial"]:
        rent += road_frontage_m * 5000          # frontage is priced by the metre
        rent += shopfront_quality * 8000        # 0–5 score
        rent += (1 if fiber_internet else 0) * 20000
        rent += loading_bay * 30000             # loading bay premium for stores/warehouses
    elif fiber_internet:
        rent += 8000   # residential fibre bonus

    # ── Security ─────────────────────────────────────────────────────────────
    rent += (1 if security_gate else 0) * 10000

    # ── Build year / condition ───────────────────────────────────────────────
    # Buildings older than 2000 depreciate; post-2015 buildings command premium
    age_discount = max(0, (2000 - build_year)) * 200
    rent -= age_discount
    rent += (condition_score - 5) * 5000   # condition scored 1–10

    # ── Structural quality ───────────────────────────────────────────────────
    rent += (structural_quality - 5) * 4500

    # ── Risk / negative factors ──────────────────────────────────────────────
    if flood_risk:
        rent -= 25000
    rent -= (noise_level - 1) * 1500

    # ── Title security ───────────────────────────────────────────────────────
    if title_type == "foncier":
        rent += 15000
    elif title_type == "none":
        rent -= 8000

    # ── Infrastructure zone multiplier ───────────────────────────────────────
    rent = rent * zone_mult

    # ── Noise ────────────────────────────────────────────────────────────────
    rent += rng.normal(0, 12000)

    # ── Floor — enforce minimums ─────────────────────────────────────────────
    return max(prop_type_cfg["sic_min_rent"], round(rent / 500) * 500)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
rng  = np.random.default_rng(42)
rows = []

for _ in range(N):
    hood_key  = rng.choice(NEIGHBOURHOOD_KEYS)
    base_rent, lat, lon, city = NEIGHBOURHOODS[hood_key]

    prop_type = rng.choice(PROPERTY_TYPE_KEYS, p=PROPERTY_TYPE_WEIGHTS)
    cfg       = PROPERTY_TYPES[prop_type]

    # Dimensions
    length_m = float(rng.integers(*cfg["length_range"]))
    width_m  = float(rng.integers(*cfg["width_range"]))
    area_m2  = length_m * width_m

    # GPS jitter ±0.005° (~550 m) to simulate intra-neighbourhood variation
    gps_lat = lat + rng.uniform(-0.005, 0.005)
    gps_lon = lon + rng.uniform(-0.005, 0.005)

    # Infra zone
    infra_zone = rng.choice(list(ZONE_MULTIPLIERS.keys()), p=ZONE_PROBS)
    zone_mult  = ZONE_MULTIPLIERS[infra_zone]

    # Bedrooms / bathrooms
    bd_vals, bd_probs = BEDROOM_DIST[prop_type]
    num_bedrooms  = int(rng.choice(bd_vals, p=bd_probs))

    ba_vals, ba_probs = BATHROOM_DIST[prop_type]
    num_bathrooms = int(rng.choice(ba_vals, p=ba_probs))

    # Floor
    fl_vals, fl_probs = FLOOR_DIST.get(prop_type, FLOOR_DIST["studio"])
    floor_level = int(rng.choice(fl_vals, p=fl_probs))

    # Basic amenities — probabilities vary by type
    res = prop_type in ("house", "apartment", "studio")
    com = cfg["commercial"]

    shared_wc       = bool(rng.choice([0, 1], p=[0.75, 0.25])) if res else False
    has_parking     = bool(rng.choice([0, 1], p=[0.55, 0.45])) if res else bool(rng.choice([0, 1], p=[0.30, 0.70]))
    has_generator   = bool(rng.choice([0, 1], p=[0.45, 0.55]))
    has_water_meter = bool(rng.choice([0, 1], p=[0.30, 0.70])) if res else True
    fiber_internet  = bool(rng.choice([0, 1], p=[0.65, 0.35]))
    security_gate   = bool(rng.choice([0, 1], p=[0.45, 0.55]))

    # Commercial-specific
    road_frontage_m   = float(rng.integers(3, 20)) if com else 0.0
    shopfront_quality = int(rng.integers(0, 6))    if com else 0
    loading_bay       = bool(rng.choice([0, 1], p=[0.60, 0.40])) if prop_type in ("store", "warehouse") else False
    standby_power_kva = float(rng.choice([0, 10, 20, 30, 50], p=[0.40, 0.25, 0.20, 0.10, 0.05])) if prop_type in ("office", "warehouse") else 0.0

    # Proximity
    near_school      = bool(rng.choice([0, 1], p=[0.45, 0.55]))
    near_market      = bool(rng.choice([0, 1], p=[0.35, 0.65]))
    near_hospital    = bool(rng.choice([0, 1], p=[0.75, 0.25]))
    near_highway     = bool(rng.choice([0, 1], p=[0.50, 0.50]))
    near_university  = bool(rng.choice([0, 1], p=[0.70, 0.30]))

    # Quality / risk
    structural_quality = int(rng.integers(3, 11))
    condition_score    = int(rng.integers(3, 11))
    build_year         = int(rng.integers(1975, 2025))
    flood_risk         = bool(rng.choice([0, 1], p=[0.80, 0.20]))
    noise_level        = int(rng.integers(1, 11))
    title_type         = rng.choice(["foncier", "occupation", "none"], p=[0.25, 0.50, 0.25])
    advance_months     = int(rng.choice(ADVANCE_OPTIONS))

    rent = compute_rent(
        hood_base=base_rent, prop_type_cfg=cfg, prop_type=prop_type,
        length_m=length_m, width_m=width_m,
        num_bedrooms=num_bedrooms, num_bathrooms=num_bathrooms,
        shared_wc=shared_wc, has_parking=has_parking,
        has_generator=has_generator, has_water_meter=has_water_meter,
        floor_level=floor_level, near_school=near_school,
        near_market=near_market, near_hospital=near_hospital,
        near_highway=near_highway, near_university=near_university,
        structural_quality=structural_quality, build_year=build_year,
        condition_score=condition_score, flood_risk=flood_risk,
        noise_level=noise_level, road_frontage_m=road_frontage_m,
        shopfront_quality=shopfront_quality, fiber_internet=fiber_internet,
        security_gate=security_gate, standby_power_kva=standby_power_kva,
        loading_bay=loading_bay, title_type=title_type,
        advance_months=advance_months, infra_zone=infra_zone,
        zone_mult=zone_mult, rng=rng,
    )

    rows.append({
        # ── Identity / location ──────────────────────────────────────────────
        "property_type":     prop_type,
        "neighbourhood":     hood_key,
        "city":              city,
        "gps_lat":           round(gps_lat, 6),
        "gps_lon":           round(gps_lon, 6),
        "infra_zone":        infra_zone,
        # ── Dimensions ──────────────────────────────────────────────────────
        "length_m":          length_m,
        "width_m":           width_m,
        "area_m2":           area_m2,
        # ── Rooms ───────────────────────────────────────────────────────────
        "num_bedrooms":      num_bedrooms,
        "num_bathrooms":     num_bathrooms,
        "floor_level":       floor_level,
        "shared_wc":         int(shared_wc),
        # ── Amenities ───────────────────────────────────────────────────────
        "has_parking":       int(has_parking),
        "has_generator":     int(has_generator),
        "has_water_meter":   int(has_water_meter),
        "fiber_internet":    int(fiber_internet),
        "security_gate":     int(security_gate),
        "standby_power_kva": standby_power_kva,
        # ── Commercial ──────────────────────────────────────────────────────
        "road_frontage_m":   road_frontage_m,
        "shopfront_quality": shopfront_quality,
        "loading_bay":       int(loading_bay),
        # ── Proximity ───────────────────────────────────────────────────────
        "near_school":       int(near_school),
        "near_market":       int(near_market),
        "near_hospital":     int(near_hospital),
        "near_highway":      int(near_highway),
        "near_university":   int(near_university),
        # ── Quality / risk ──────────────────────────────────────────────────
        "structural_quality": structural_quality,
        "condition_score":   condition_score,
        "build_year":        build_year,
        "flood_risk":        int(flood_risk),
        "noise_level":       noise_level,
        # ── Legal / contractual ──────────────────────────────────────────────
        "title_type":        title_type,
        "advance_months":    advance_months,
        # ── Target ──────────────────────────────────────────────────────────
        "estimated_rent":    int(rent),
    })

df = pd.DataFrame(rows)
df.to_csv("rent_seed.csv", index=False)

print(f"Generated {len(df)} rows, {len(df.columns)} columns.")
print(f"\nProperty type distribution:\n{df['property_type'].value_counts()}")
print(f"\nRent statistics (all types):")
print(f"  Mean   : {df.estimated_rent.mean():>12,.0f} CFA")
print(f"  Median : {df.estimated_rent.median():>12,.0f} CFA")
print(f"  Min    : {df.estimated_rent.min():>12,.0f} CFA")
print(f"  Max    : {df.estimated_rent.max():>12,.0f} CFA")
print(f"\nMean rent by property type:")
print(df.groupby("property_type")["estimated_rent"].mean().sort_values(ascending=False).apply(lambda x: f"{x:,.0f} CFA"))
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample rows:\n{df.head(3).to_string()}")