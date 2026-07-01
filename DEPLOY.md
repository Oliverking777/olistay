# Deploying OLISTAY AI Engine to Render (free tier)

The service is a FastAPI app. The ML model is a small (~1.3 MB) joblib file
committed to the repo, and the LLM runs via the **Groq** hosted API — so there
is nothing heavy to host. It fits Render's free tier comfortably.

## Before you deploy: rotate your API keys

Your old `.env` contained live keys. Regenerate them (the old ones should be
treated as compromised) and use the new values in Render:

- **Groq** → https://console.groq.com/keys
- **Gemini** → https://aistudio.google.com/app/apikey

## Steps

1. Go to https://render.com and sign in with GitHub (free, no card needed).
2. **New → Blueprint**, then select the `olistay` repository.
3. Render reads [`render.yaml`](render.yaml) and provisions the web service.
4. Fill in the env vars marked "set in dashboard":
   - `GROQ_API_KEY` — your **new** Groq key
   - `GEMINI_API_KEY` — your **new** Gemini key
   - `SPRING_BOOT_BASE_URL` — the URL of your frontend/backend (for CORS)
5. Click **Apply / Create**. First build takes a few minutes (installs deps).
6. When live, verify:
   - `https://<your-app>.onrender.com/`        → service info
   - `https://<your-app>.onrender.com/health`  → module health
   - `https://<your-app>.onrender.com/docs`    → interactive API docs

## Notes on the free tier

- The instance **sleeps after 15 min idle** and cold-starts in ~30 s on the
  next request. Fine for demos; upgrade to a paid instance for always-on.
- The filesystem is **ephemeral** — `data/chromadb/` and any retrained model
  reset on redeploy/restart. The seed model and CSV in the repo are always
  present, so predictions work out of the box.
- Free instances have ~512 MB RAM. If the build/boot ever hits a memory limit
  (xgboost + chromadb + onnxruntime), the next cheapest fix is Render's
  Starter instance, or trimming unused LLM SDKs.
