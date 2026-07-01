"""
main.py — OLISTAY AI Engine entry point
─────────────────────────────────────────────────────────────────────────────
FastAPI application mounting all AI microservice routers.

AI Components Summary (for assessor reference):
────────────────────────────────────────────────
  /ml/predict-rent            XGBoost regression — market rent prediction
  /ml/train-rent-model        Trigger monthly retraining from new lease data
  /ml/rent-model/status       Model metrics incl. MAE vs baseline comparison
  /ml/feedback/outcome        Record tenant decision → triggers weight learning
  /ml/feedback/weights        Compare expert vs learned scoring weights
  /scoring/score              Optimality scoring (rule-based + ML signals)
  /recommender/recommend      4-stage hybrid pipeline
  /recommender/pipeline/info  Full AI component status dashboard
  /recommender/interact       Record CF interaction signals
  /narrator/explain           LLM-generated advisory explanation (RAG-backed)
─────────────────────────────────────────────────────────────────────────────
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from financial.profiler import router as financial_router
from financial.hidden_costs import router as costs_router
from scoring.optimality import router as scoring_router
from scoring.tenant_behaviour import router as behaviour_router
from ml_models.rent_predictor import router as rent_router
from ml_models.occupancy_forecaster import router as occupancy_router
from ml_models.feedback_learner import router as feedback_router
from narrator.advisor import router as narrator_router
from recommender.pipeline import router as pipeline_router
from recommender.collaborative import router as collaborative_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[OLISTAY-AI] Starting AI Engine...")
    os.makedirs("./data/models", exist_ok=True)
    os.makedirs("./data/chromadb", exist_ok=True)
    print("[OLISTAY-AI] Service ready.")
    yield
    print("[OLISTAY-AI] Shutting down.")


app = FastAPI(
    title="OLISTAY AI Engine",
    description=(
        "Four-stage hybrid AI recommender for the OLISTAY rental platform. "
        "Combines knowledge-based filtering, XGBoost rent prediction, "
        "content-based cosine similarity ranking, and collaborative filtering "
        "with adaptive weight learning from tenant outcomes."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("SPRING_BOOT_BASE_URL", "http://localhost:8082"),
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(financial_router,   prefix="/financial",    tags=["Financial"])
app.include_router(costs_router,       prefix="/financial",    tags=["Financial"])
app.include_router(scoring_router,     prefix="/scoring",      tags=["Scoring"])
app.include_router(behaviour_router,   prefix="/scoring",      tags=["Scoring"])
app.include_router(rent_router,        prefix="/ml",           tags=["ML Models"])
app.include_router(occupancy_router,   prefix="/ml",           tags=["ML Models"])
app.include_router(feedback_router,    prefix="/ml",           tags=["ML Models"])
app.include_router(narrator_router,    prefix="/narrator",     tags=["Narrator"])
app.include_router(pipeline_router,    prefix="/recommender",  tags=["Recommender"])
app.include_router(collaborative_router, prefix="/recommender", tags=["Recommender"])


@app.get("/")
def root():
    return {
        "service":     "OLISTAY AI Engine",
        "version":     "2.0.0",
        "status":      "running",
        "environment": os.getenv("APP_ENV", "development"),
        "ai_components": {
            "xgboost_rent_predictor":      "active",
            "knowledge_based_filter":      "active",
            "content_based_recommender":   "active",
            "collaborative_filter":        "progressive (cold-start safe)",
            "adaptive_weight_learner":     "progressive (activates at 10 outcomes)",
            "llm_advisor":                 os.getenv("LLM_PROVIDER", "ollama"),
            "rag_knowledge_base":          "chromadb",
        },
    }


@app.get("/health")
def health_check():
    try:
        from ml_models.rent_predictor import _metrics
        rent_mae = _metrics.get("mae_cfa", "not trained")
        rent_r2  = _metrics.get("r2_score", "not trained")
    except Exception:
        rent_mae = rent_r2 = "unavailable"

    try:
        from ml_models.feedback_learner import get_feedback_store
        fb = get_feedback_store()
        fb_stats = {
            "records":          len(fb._records),
            "learning_active":  fb._learning_active,
        }
    except Exception:
        fb_stats = {"error": "unavailable"}

    try:
        from recommender.collaborative import get_store
        cf_stats = get_store().stats()
    except Exception:
        cf_stats = {"error": "unavailable"}

    return {
        "status": "healthy",
        "modules": {
            "financial_profiler":   "active",
            "hidden_costs":         "active",
            "optimality_scoring":   "active",
            "rent_predictor":       {"active": True, "mae_cfa": rent_mae, "r2": rent_r2},
            "tenant_behaviour":     "active",
            "occupancy_forecaster": "active",
            "feedback_learner":     fb_stats,
            "collaborative_filter": cf_stats,
            "narrator_llm":         os.getenv("LLM_PROVIDER", "ollama"),
            "recommender_pipeline": "active",
        },
    }


if __name__ == "__main__":
    import uvicorn

    # Auto-reload is DEV-only. In production (APP_ENV=production) it stays off.
    # Even in dev we must NOT watch ./data — feedback_learner, rent_predictor and
    # ChromaDB write .joblib / .sqlite3 files into ./data/models and
    # ./data/chromadb at runtime. If those live inside the reload-watched tree,
    # a normal request that persists data triggers a server restart mid-request,
    # which the Spring Boot caller sees as "I/O error on POST ... : null".
    dev_reload = os.getenv("APP_ENV", "development").lower() != "production"

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", 8000)),
        reload=dev_reload,
        reload_excludes=["data/*", "*.joblib", "*.sqlite3", "*.json"] if dev_reload else None,
    )