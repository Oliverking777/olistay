"""
ml_models/feedback_learner.py
─────────────────────────────────────────────────────────────────────────────
ML-Based Scoring Weight Learner — OLISTAY AI Engine
─────────────────────────────────────────────────────────────────────────────
Addresses the core architectural critique: the optimality scoring engine
must not remain a static weighted-sum forever. This module implements the
feedback loop that progressively replaces the hand-crafted domain weights
with weights learned from actual tenant lease decisions.

Architecture (Ameisen, 2020 — Chapter 3: start simple, iterate toward ML):
─────────────────────────────────────────────────────────────────────────────
Phase 1 (Cold Start — < MIN_FEEDBACK_THRESHOLD outcomes recorded):
    The scoring engine uses the expert-defined static weights from .env.
    This guarantees meaningful recommendations from day one.

Phase 2 (Learning — ≥ MIN_FEEDBACK_THRESHOLD outcomes recorded):
    A Gradient Boosted Trees model (XGBoost) is trained on recorded
    tenant outcomes (category scores → lease/reject decision). The model
    learns which scoring dimensions most reliably predict tenant satisfaction,
    producing data-driven weights that replace the static defaults.

Feedback signal:
    When a tenant signs a lease for a recommended property → outcome = 1
    When a tenant explicitly rejects a recommended property → outcome = 0
    When a tenant views but takes no action → outcome not recorded (no signal)

The learned weights are persisted to disk and loaded at startup.
This creates a genuine feedback loop: more tenants → better weights →
better recommendations → more tenants.

Training schedule:
    Retrain is triggered automatically when a new outcome is recorded and
    the total count crosses a power-of-two milestone (10, 20, 40, 80, ...),
    ensuring diminishing retraining frequency as the dataset grows.

References:
    Ameisen (2020) — Building Machine Learning Powered Applications
    Jannach et al. (2011) — Recommender Systems, Chapter 3
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime
from collections import defaultdict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "./data/models")
FEEDBACK_LOG_PATH = os.path.join(MODELS_DIR, "feedback_log.json")
LEARNED_WEIGHTS_PATH = os.path.join(MODELS_DIR, "learned_weights.json")
WEIGHT_MODEL_PATH = os.path.join(MODELS_DIR, "weight_learner.joblib")
MIN_FEEDBACK_THRESHOLD = 10  # minimum outcomes before learning activates

# Default expert weights (used when learning has not yet activated)
DEFAULT_WEIGHTS = {
    "financial":      0.30,
    "goal_alignment": 0.25,
    "household":      0.15,
    "lifestyle":      0.15,
    "safety":         0.10,
    "stability":      0.05,
}

CATEGORY_NAMES = list(DEFAULT_WEIGHTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY FEEDBACK STORE
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackStore:
    """
    Stores tenant outcome signals for weight learning.

    Each record maps (tenant_id, property_id) → {category_scores, outcome}.
    Outcome: 1 = lease signed, 0 = property rejected.
    """

    def __init__(self):
        self._records: list = []
        self._learned_weights: Optional[dict] = None
        self._model = None
        self._learning_active: bool = False
        self._load_from_disk()

    def _load_from_disk(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        # Load feedback log
        if os.path.exists(FEEDBACK_LOG_PATH):
            with open(FEEDBACK_LOG_PATH, "r") as f:
                self._records = json.load(f)
            print(f"[FEEDBACK] Loaded {len(self._records)} feedback records.")
        # Load learned weights
        if os.path.exists(LEARNED_WEIGHTS_PATH):
            with open(LEARNED_WEIGHTS_PATH, "r") as f:
                self._learned_weights = json.load(f)
            self._learning_active = True
            print(f"[FEEDBACK] Loaded learned weights: {self._learned_weights}")
        # Load model
        if os.path.exists(WEIGHT_MODEL_PATH):
            self._model = joblib.load(WEIGHT_MODEL_PATH)
            print("[FEEDBACK] Learned weight model loaded from disk.")

    def _save_log(self):
        with open(FEEDBACK_LOG_PATH, "w") as f:
            json.dump(self._records, f)

    def record_outcome(
        self,
        tenant_id: str,
        property_id: str,
        category_scores: dict,
        outcome: int,  # 1 = lease signed, 0 = rejected
        optimality_total: float,
    ) -> dict:
        """Record a tenant outcome and trigger retraining if milestone reached."""
        record = {
            "tenant_id":        tenant_id,
            "property_id":      property_id,
            "outcome":          outcome,
            "optimality_total": optimality_total,
            "timestamp":        datetime.utcnow().isoformat(),
            **{f"score_{cat}": category_scores.get(cat, 50.0) for cat in CATEGORY_NAMES}
        }
        self._records.append(record)
        self._save_log()

        n = len(self._records)
        retrained = False

        # Retrain at power-of-two milestones ≥ threshold
        if n >= MIN_FEEDBACK_THRESHOLD and (n & (n - 1) == 0 or n == MIN_FEEDBACK_THRESHOLD):
            metrics = self._retrain()
            retrained = True
            return {
                "recorded": True,
                "total_records": n,
                "learning_active": self._learning_active,
                "retrained": retrained,
                "training_metrics": metrics,
                "current_weights": self.get_weights(),
            }

        return {
            "recorded": True,
            "total_records": n,
            "learning_active": self._learning_active,
            "retrained": False,
            "training_metrics": None,
            "current_weights": self.get_weights(),
        }

    def _retrain(self) -> dict:
        """
        Train an XGBoost classifier on recorded outcomes.
        Feature importance scores are normalised to produce learned scoring weights.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import roc_auc_score

            df = pd.DataFrame(self._records)
            feature_cols = [f"score_{cat}" for cat in CATEGORY_NAMES]

            # Require at least 2 classes to train
            if df["outcome"].nunique() < 2:
                print("[FEEDBACK] Not enough outcome variety to train (need both 0 and 1).")
                return {"status": "skipped", "reason": "insufficient_outcome_variety"}

            X = df[feature_cols].values
            y = df["outcome"].values

            model = XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X, y)

            # Cross-validation AUC
            if len(df) >= 5:
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(df)), scoring="roc_auc")
                mean_auc = float(np.mean(cv_scores))
            else:
                mean_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

            # Derive weights from feature importances
            importances = model.feature_importances_
            total = importances.sum()
            if total == 0:
                learned_w = DEFAULT_WEIGHTS.copy()
            else:
                learned_w = {
                    CATEGORY_NAMES[i]: round(float(importances[i] / total), 4)
                    for i in range(len(CATEGORY_NAMES))
                }
                # Normalise to sum exactly to 1.0
                s = sum(learned_w.values())
                learned_w = {k: round(v / s, 4) for k, v in learned_w.items()}
                # Force sum to 1 (floating point fix)
                diff = 1.0 - sum(learned_w.values())
                max_key = max(learned_w, key=learned_w.get)
                learned_w[max_key] = round(learned_w[max_key] + diff, 4)

            joblib.dump(model, WEIGHT_MODEL_PATH)
            self._model = model

            with open(LEARNED_WEIGHTS_PATH, "w") as f:
                json.dump(learned_w, f)
            self._learned_weights = learned_w
            self._learning_active = True

            print(f"[FEEDBACK] Retrained on {len(df)} samples. AUC={mean_auc:.3f}")
            print(f"[FEEDBACK] Learned weights: {learned_w}")

            return {
                "status":             "trained",
                "samples":            len(df),
                "mean_cv_auc":        round(mean_auc, 4),
                "learned_weights":    learned_w,
                "default_weights":    DEFAULT_WEIGHTS,
                "weight_shift":       {
                    k: round(learned_w.get(k, 0) - DEFAULT_WEIGHTS.get(k, 0), 4)
                    for k in CATEGORY_NAMES
                },
            }
        except Exception as e:
            print(f"[FEEDBACK] Retraining failed: {e}")
            return {"status": "failed", "error": str(e)}

    def get_weights(self) -> dict:
        """Return current weights — learned if active, else expert defaults."""
        if self._learning_active and self._learned_weights:
            return {**self._learned_weights, "_source": "learned"}
        return {**DEFAULT_WEIGHTS, "_source": "expert_default"}

    def stats(self) -> dict:
        n = len(self._records)
        outcomes = [r["outcome"] for r in self._records]
        return {
            "total_feedback_records":  n,
            "lease_outcomes":          sum(outcomes),
            "reject_outcomes":         n - sum(outcomes),
            "learning_active":         self._learning_active,
            "min_threshold":           MIN_FEEDBACK_THRESHOLD,
            "records_to_first_learning": max(0, MIN_FEEDBACK_THRESHOLD - n),
            "current_weights":         self.get_weights(),
            "weight_model_trained":    self._model is not None,
        }


# Global singleton
_feedback_store = FeedbackStore()


def get_feedback_store() -> FeedbackStore:
    return _feedback_store


def get_current_weights() -> dict:
    """
    Public interface for the scoring engine.
    Returns learned weights if enough feedback exists, else expert defaults.
    """
    store = get_feedback_store()
    return store.get_weights()


# ─────────────────────────────────────────────────────────────────────────────
# API MODELS
# ─────────────────────────────────────────────────────────────────────────────

class OutcomeRequest(BaseModel):
    tenant_id: str
    property_id: str
    category_scores: dict = Field(..., description="Scores from /scoring/score")
    optimality_total: float
    outcome: int = Field(..., ge=0, le=1, description="1=lease_signed, 0=rejected")


class WeightComparisonResponse(BaseModel):
    learning_active: bool
    records_needed_for_learning: int
    expert_weights: dict
    current_weights: dict
    weight_source: str
    interpretation: str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/feedback/outcome")
def record_outcome(data: OutcomeRequest):
    """
    Record a tenant decision outcome.
    Called by Spring Boot when a tenant signs or explicitly rejects a property.
    Triggers automatic model retraining at data milestones.
    """
    try:
        store = get_feedback_store()
        result = store.record_outcome(
            tenant_id=data.tenant_id,
            property_id=data.property_id,
            category_scores=data.category_scores,
            outcome=data.outcome,
            optimality_total=data.optimality_total,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/weights", response_model=WeightComparisonResponse)
def get_weights_comparison():
    """
    Return current scoring weights and compare to expert defaults.
    Exposes the learning progress to system operators.
    """
    store = get_feedback_store()
    current = store.get_weights()
    source = current.pop("_source", "unknown")
    return {
        "learning_active":              store._learning_active,
        "records_needed_for_learning":  max(0, MIN_FEEDBACK_THRESHOLD - len(store._records)),
        "expert_weights":               DEFAULT_WEIGHTS,
        "current_weights":              current,
        "weight_source":                source,
        "interpretation": (
            "Weights are being learned from tenant lease decisions. "
            "The scoring engine now reflects real tenant behaviour, not only expert judgement."
            if store._learning_active else
            f"Using expert-defined weights. Collect {max(0, MIN_FEEDBACK_THRESHOLD - len(store._records))} "
            f"more outcome records to activate learning."
        ),
    }


@router.get("/feedback/stats")
def feedback_stats():
    """Return full feedback store statistics."""
    return get_feedback_store().stats()


@router.post("/feedback/force-retrain")
def force_retrain():
    """Manually trigger a retraining cycle (admin use only)."""
    try:
        store = get_feedback_store()
        if len(store._records) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 feedback records to retrain."
            )
        metrics = store._retrain()
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))