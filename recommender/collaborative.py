"""
recommender/collaborative.py
─────────────────────────────────────────────────────────────────────────────
Collaborative Filtering Layer — OLISTAY AI Engine
─────────────────────────────────────────────────────────────────────────────
Implements the collaborative filtering stage of OLISTAY's hybrid recommendation
pipeline. This module maintains an in-memory user-item interaction matrix and
computes neighbourhood-based collaborative signals that augment the content-based
ranking at runtime.

Design rationale (Ameisen, 2020; Jannach et al., 2011):
- At platform launch, no historical interaction data exists. The cold-start
  condition is handled explicitly: when fewer than MIN_INTERACTIONS_THRESHOLD
  interactions are recorded, the module returns neutral augmentation weights
  (1.0 for all candidates) and the pipeline relies entirely on content-based
  and knowledge-based signals.
- As interaction data accumulates, the collaborative signal progressively
  augments the content-based ranking. This is the pipeline hybridisation
  strategy described in Chapter 2: content-based provides the primary ranking,
  collaborative signals re-weight the output.

Interaction types recorded:
    VIEW       — tenant viewed property details
    SHORTLIST  — tenant added property to shortlist
    ENQUIRY    — tenant sent an enquiry to landlord
    LEASE      — tenant signed a lease (strongest signal)

Pipeline position: Stage 3 (augments content-based ranking scores).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MIN_INTERACTIONS_THRESHOLD = 10   # minimum interactions before CF activates
MIN_SIMILAR_USERS = 2             # minimum similar users needed to produce signal
SIMILARITY_THRESHOLD = 0.15       # minimum cosine similarity to count as neighbour
CF_AUGMENTATION_WEIGHT = 0.20     # how much CF signal adjusts content-based score
                                  # (0.0 = pure content, 1.0 = pure CF)

# Interaction strength weights (implicit feedback)
INTERACTION_WEIGHTS = {
    "VIEW":       0.1,
    "SHORTLIST":  0.5,
    "ENQUIRY":    0.8,
    "LEASE":      1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY INTERACTION STORE
# In production this would be backed by PostgreSQL via the Spring Boot service.
# The store is initialised empty and populated via the /recommender/interact
# endpoint. It persists for the lifetime of the FastAPI process.
# ─────────────────────────────────────────────────────────────────────────────

class InteractionStore:
    """
    Lightweight in-memory user-item interaction matrix.

    Structure:
        _matrix[tenant_id][property_id] = weighted_score (float, 0.0–1.0)
        _counts[tenant_id] = total interaction count for that tenant
    """

    def __init__(self):
        # user → {property_id: implicit_score}
        self._matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total_interactions: int = 0
        self._interaction_log: List[dict] = []

    def record(
        self,
        tenant_id: str,
        property_id: str,
        interaction_type: str,
    ) -> None:
        """Record a tenant-property interaction and update the matrix."""
        weight = INTERACTION_WEIGHTS.get(interaction_type.upper(), 0.1)
        current = self._matrix[tenant_id].get(property_id, 0.0)
        # Accumulate but cap at 1.0 (multiple interactions strengthen signal)
        self._matrix[tenant_id][property_id] = min(1.0, current + weight)
        self._counts[tenant_id] += 1
        self._total_interactions += 1
        self._interaction_log.append({
            "tenant_id":        tenant_id,
            "property_id":      property_id,
            "interaction_type": interaction_type,
            "weight":           weight,
            "timestamp":        datetime.utcnow().isoformat(),
        })

    def get_user_vector(self, tenant_id: str, all_property_ids: List[str]) -> np.ndarray:
        """Return tenant's interaction vector over the known property catalogue."""
        user_data = self._matrix.get(tenant_id, {})
        return np.array(
            [user_data.get(pid, 0.0) for pid in all_property_ids],
            dtype=np.float64
        )

    def get_all_tenants(self) -> List[str]:
        return list(self._matrix.keys())

    def get_all_property_ids(self) -> List[str]:
        all_ids = set()
        for user_data in self._matrix.values():
            all_ids.update(user_data.keys())
        return list(all_ids)

    @property
    def total_interactions(self) -> int:
        return self._total_interactions

    @property
    def is_cold_start(self) -> bool:
        return self._total_interactions < MIN_INTERACTIONS_THRESHOLD

    def stats(self) -> dict:
        return {
            "total_interactions":    self._total_interactions,
            "unique_tenants":        len(self._matrix),
            "unique_properties":     len(self.get_all_property_ids()),
            "cold_start_active":     self.is_cold_start,
            "threshold":             MIN_INTERACTIONS_THRESHOLD,
        }


# Global singleton store (shared across all requests in this process)
_store = InteractionStore()


def get_store() -> InteractionStore:
    return _store


# ─────────────────────────────────────────────────────────────────────────────
# USER SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), 0.0, 1.0))


def find_similar_users(
    tenant_id: str,
    store: InteractionStore,
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Find the top-K most similar tenants to the active tenant
    based on cosine similarity of their interaction vectors.

    Returns list of (tenant_id, similarity_score) tuples, sorted descending.
    """
    all_property_ids = store.get_all_property_ids()
    if not all_property_ids:
        return []

    active_vec = store.get_user_vector(tenant_id, all_property_ids)
    if np.linalg.norm(active_vec) == 0:
        # Active tenant has no interactions yet — cannot compute similarity
        return []

    similarities = []
    for other_id in store.get_all_tenants():
        if other_id == tenant_id:
            continue
        other_vec = store.get_user_vector(other_id, all_property_ids)
        sim = _cosine_similarity(active_vec, other_vec)
        if sim >= SIMILARITY_THRESHOLD:
            similarities.append((other_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# COLLABORATIVE AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_collaborative_scores(
    tenant_id: str,
    candidate_property_ids: List[str],
    store: Optional[InteractionStore] = None,
) -> Dict[str, float]:
    """
    Compute collaborative augmentation scores for a list of candidate properties.

    Returns a dict mapping property_id → augmentation_score (0.0 – 1.0).

    Cold-start behaviour:
        If fewer than MIN_INTERACTIONS_THRESHOLD total interactions exist in the
        store, returns 0.5 (neutral) for all candidates. The pipeline treats
        this as "no collaborative signal available" and relies on content-based
        and knowledge-based scores only.

    Active behaviour:
        Identifies similar users, collects their interaction scores for each
        candidate property, and computes a similarity-weighted average.
    """
    if store is None:
        store = get_store()

    # Cold-start: return neutral scores
    if store.is_cold_start:
        return {pid: 0.5 for pid in candidate_property_ids}

    similar_users = find_similar_users(tenant_id, store)

    # Not enough similar users found
    if len(similar_users) < MIN_SIMILAR_USERS:
        return {pid: 0.5 for pid in candidate_property_ids}

    all_property_ids = store.get_all_property_ids()
    cf_scores = {}

    for property_id in candidate_property_ids:
        if property_id not in all_property_ids:
            # No interaction data for this property at all
            cf_scores[property_id] = 0.5
            continue

        weighted_sum = 0.0
        weight_total = 0.0

        for similar_tenant_id, similarity in similar_users:
            interaction_score = store._matrix.get(similar_tenant_id, {}).get(property_id, 0.0)
            weighted_sum += similarity * interaction_score
            weight_total += similarity

        if weight_total == 0:
            cf_scores[property_id] = 0.5
        else:
            cf_scores[property_id] = round(weighted_sum / weight_total, 4)

    return cf_scores


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTION RECORDING API
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, HTTPException

router = APIRouter()


class InteractionRequest(BaseModel):
    tenant_id: str
    property_id: str
    interaction_type: str = Field(
        ...,
        description="One of: VIEW, SHORTLIST, ENQUIRY, LEASE"
    )


class InteractionResponse(BaseModel):
    recorded: bool
    tenant_id: str
    property_id: str
    interaction_type: str
    implicit_score: float
    total_interactions: int
    cold_start_active: bool


@router.post("/interact", response_model=InteractionResponse)
def record_interaction(data: InteractionRequest):
    """
    Record a tenant-property interaction.
    Called by the Spring Boot service whenever a tenant views, shortlists,
    enquires about, or signs a lease for a property.
    """
    try:
        store = get_store()
        interaction_type = data.interaction_type.upper()
        if interaction_type not in INTERACTION_WEIGHTS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interaction_type. Must be one of: {list(INTERACTION_WEIGHTS.keys())}"
            )
        store.record(data.tenant_id, data.property_id, interaction_type)
        return {
            "recorded":           True,
            "tenant_id":          data.tenant_id,
            "property_id":        data.property_id,
            "interaction_type":   interaction_type,
            "implicit_score":     store._matrix[data.tenant_id][data.property_id],
            "total_interactions": store.total_interactions,
            "cold_start_active":  store.is_cold_start,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interact/stats")
def interaction_stats():
    """Return current state of the collaborative filtering interaction store."""
    return get_store().stats()