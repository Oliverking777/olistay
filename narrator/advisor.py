import os
import json
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

LLM_PROVIDER      = os.getenv("LLM_PROVIDER",       "ollama")
OLLAMA_URL        = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",       "llama3")
CLAUDE_API_KEY    = os.getenv("CLAUDE_API_KEY",    "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL",      "claude-haiku-4-5-20251001")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL",      "gpt-4o-mini")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",    "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL",      "gemini-1.5-flash")
CHROMA_DIR        = os.getenv("CHROMA_PERSIST_DIR",     "./data/chromadb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "financial_knowledge")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_chroma_collection = None


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            _chroma_collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
        except Exception as e:
            print(f"[NARRATOR] ChromaDB unavailable: {e}")
    return _chroma_collection


def retrieve_financial_knowledge(query: str) -> str:
    try:
        collection = get_chroma_collection()
        if collection is None:
            return ""
        results = collection.query(query_texts=[query], n_results=2)
        docs = results.get("documents", [[]])[0]
        if docs:
            return "\n".join(docs)
    except Exception as e:
        print(f"[NARRATOR] RAG retrieval failed: {e}")
    return ""


def build_prompt(scoring_result: dict, user_profile: dict, knowledge: str) -> str:
    flags_text = (
        "".join(f"- {f}" for f in scoring_result.get("flags", []))
        or "None"
    )
    knowledge_section = (
        f"Relevant financial principle:{knowledge}"
        if knowledge else ""
    )
    prompt = f"""You are a friendly housing advisor for OLISTAY, a rental platform in Cameroon.
Your job is to explain a property recommendation to a tenant in plain, warm language.
You must ONLY use the data provided below. Do NOT invent numbers or advice.
Write exactly 3-4 sentences. Be specific with CFA amounts. Be encouraging but honest.
{knowledge_section}
TENANT PROFILE:
- Monthly income: {user_profile.get('monthly_income', 0):,.0f} CFA
- Fixed obligations: {user_profile.get('fixed_obligations', 0):,.0f} CFA
- Savings goal: {user_profile.get('savings_goal', 0):,.0f} CFA in {user_profile.get('goal_timeline_months', 12)} months
- Household size: {user_profile.get('household_size', 1)} person(s)

PROPERTY SCORE:
- Total score: {scoring_result.get('total_score', 0)}/100
- Grade: {scoring_result.get('grade', 'N/A')}
- Recommendation: {scoring_result.get('recommendation', 'N/A')}
- Financial compatibility: {scoring_result.get('category_scores', {}).get('financial', 0):.0f}/100
- Goal alignment: {scoring_result.get('category_scores', {}).get('goal_alignment', 0):.0f}/100
- Household fit: {scoring_result.get('category_scores', {}).get('household', 0):.0f}/100
- Safety score: {scoring_result.get('category_scores', {}).get('safety', 0):.0f}/100

FINANCIAL SUMMARY: {scoring_result.get('financial_summary', '')}
TCO SUMMARY: {scoring_result.get('tco_summary', '')}

FLAGS:
{flags_text}

Now write your 3-4 sentence advisory explanation for this tenant:"""
    return prompt


def call_ollama(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def call_claude(prompt: str) -> str:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      CLAUDE_MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


def call_openai(prompt: str) -> str:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      OPENAI_MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def call_gemini(prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    response = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

def call_groq(prompt: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      GROQ_MODEL,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def generate_explanation(scoring_result, user_profile, provider=None) -> str:
    active_provider = provider or LLM_PROVIDER
    rag_query = (
        f"rent affordability income {user_profile.get('monthly_income', 0)} CFA "
        f"savings goal housing budget Cameroon"
    )
    knowledge = retrieve_financial_knowledge(rag_query)
    prompt = build_prompt(scoring_result, user_profile, knowledge)
    try:
        if active_provider == "ollama":
            return call_ollama(prompt)
        elif active_provider == "claude":
            return call_claude(prompt)
        elif active_provider == "openai":
            return call_openai(prompt)
        elif active_provider == "gemini":
            return call_gemini(prompt)
        elif active_provider == "groq":
            return call_groq(prompt)
        else:
            return scoring_result.get("financial_summary", "No explanation available.")
    except Exception as e:
        print(f"[NARRATOR] LLM call failed ({active_provider}): {e}")
        return scoring_result.get("financial_summary", "No explanation available.")


class ExplainRequest(BaseModel):
    scoring_result: dict = Field(..., description="Full response from /scoring/score endpoint")
    user_profile: dict = Field(..., description="Tenant profile data")
    provider_override: Optional[str] = Field(None)

class ExplainResponse(BaseModel):
    explanation: str
    provider_used: str
    rag_used: bool
    fallback_used: bool

class KnowledgeChunk(BaseModel):
    text: str
    source: str
    chunk_id: str

class LoadKnowledgeRequest(BaseModel):
    chunks: List[KnowledgeChunk]


@router.post("/explain", response_model=ExplainResponse)
def explain_recommendation(data: ExplainRequest):
    try:
        fallback_used = False
        rag_used = False
        collection = get_chroma_collection()
        if collection and collection.count() > 0:
            rag_used = True
        explanation = generate_explanation(
            data.scoring_result,
            data.user_profile,
            data.provider_override
        )
        if explanation == data.scoring_result.get("financial_summary"):
            fallback_used = True
        return {
            "explanation":   explanation,
            "provider_used": data.provider_override or LLM_PROVIDER,
            "rag_used":      rag_used,
            "fallback_used": fallback_used,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/load")
def load_knowledge(data: LoadKnowledgeRequest):
    try:
        collection = get_chroma_collection()
        if collection is None:
            raise HTTPException(status_code=503, detail="ChromaDB is not available")
        collection.upsert(
            ids=[chunk.chunk_id for chunk in data.chunks],
            documents=[chunk.text for chunk in data.chunks],
            metadatas=[{"source": chunk.source} for chunk in data.chunks]
        )
        return {
            "status":       "loaded",
            "chunks_added": len(data.chunks),
            "total_in_db":  collection.count(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge/status")
def knowledge_status():
    try:
        collection = get_chroma_collection()
        if collection is None:
            return {"available": False, "chunk_count": 0}
        return {
            "available":   True,
            "chunk_count": collection.count(),
            "provider":    LLM_PROVIDER,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}