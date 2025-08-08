# Railway Deployment for Finserv API

This repository contains two services to be deployed on Railway:
- python-fastapi (public API) -> exposes /query, /webhook/submission, /health
- node-backend (internal processing) -> exposes /process, /health (private)

## 1) Create two services in Railway
- Service A: FastAPI
  - Root directory: `api/python-fastapi`
  - Start command: `python -m uvicorn app:app --host 0.0.0.0 --port $PORT`
  - Environment variables:
    - PORT: provided by Railway
    - NODE_SERVICE_URL: set to the Railway URL of Service B (e.g., https://<node>.up.railway.app)
    - INTERNAL_SHARED_SECRET: strong random, same value on both services
    - MAX_FILE_SIZE_MB: 20
    - ALLOWED_EXTENSIONS: pdf,docx,eml
    - WEBHOOK_SHARED_SECRET: optional secret required in header `X-Webhook-Secret`

- Service B: Node.js backend
  - Root directory: `api/node-backend`
  - Start command: `node src/server.js`
  - Environment variables:
    - PORT: provided by Railway
    - INTERNAL_SHARED_SECRET: same as Service A
    - GEMINI_API_KEY: your Gemini API key
    - MONGODB_URI: your MongoDB connection string
    - MONGODB_DB: finserv
    - (Pinecone related are optional; currently disabled and using local fallback)

## 2) Networking
- Keep Node service internal if possible; only FastAPI is public.
- In FastAPI, set NODE_SERVICE_URL to the internal/private URL of Node service.

## 3) Webhook URL
- Use FastAPI endpoint: `POST https://<fastapi>.up.railway.app/webhook/submission`
- Headers:
  - `Content-Type: application/json`
  - `X-Webhook-Secret: <WEBHOOK_SHARED_SECRET>` (if configured)
- JSON Body example:
```
{
  "query": "Does this policy cover knee surgery?",
  "urls": ["https://example.com/policy.pdf"],
  "files": [
    {"filename": "policy.pdf", "contentBase64": "<base64-string>"}
  ]
}
```

## 4) Health Checks
- FastAPI: GET /health
- Node: GET /health

## 5) Notes
- Logs are stored in MongoDB Atlas via Node service; failures are tolerated (warnings only).
- Ensure secrets are never committed; set them in Railway dashboard.
- Scale up resources if large files or high concurrency expected.
