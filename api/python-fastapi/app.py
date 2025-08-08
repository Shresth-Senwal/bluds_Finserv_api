"""
FastAPI API layer for LLM-Powered Intelligent Query–Retrieval System.

Responsibilities:
- Validate inputs (files or URLs + query)
- Enforce limits and sanitize
- Forward payload to Node processing service
- Stream/return structured JSON response
- No model keys stored here; uses INTERNAL_SHARED_SECRET

Sample request (multipart for files):
  POST /query
  Content-Type: multipart/form-data
  fields:
    query: "Does this policy cover knee surgery, and what are the conditions?"
    files: [policy.pdf, terms.docx]

Sample request (JSON with URLs):
  POST /query
  {
    "query": "Does this policy cover knee surgery, and what are the conditions?",
    "urls": ["https://example.com/policies/health-123.pdf"]
  }

Sample success response:
  {
    "answer": "Yes, knee surgery is covered subject to pre-authorization...",
    "clauses": [{"id":"c1","text":"...","score":0.91,"source":"policy.pdf#p12"}],
    "rationale": "Based on clause 4.2 (Surgical Benefits) and 7.1 (Exclusions)...",
    "metadata": {
      "confidence": 0.84,
      "model": "gemini-2.5-flash-lite",
      "latency_ms": 1240,
      "sources": ["policy.pdf#p12", "terms.docx#p4"],
      "request_id": "..."
    }
  }
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, field_validator
import os
import json
import asyncio
import urllib.request
import urllib.error
from typing import List, Optional, TypedDict, Mapping, Dict, Any, cast

# Load env
from dotenv import load_dotenv
load_dotenv()

NODE_SERVICE_URL = os.getenv("NODE_SERVICE_URL", "http://localhost:7070")
INTERNAL_SHARED_SECRET = os.getenv("INTERNAL_SHARED_SECRET", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,eml").split(","))
WEBHOOK_SHARED_SECRET = os.getenv("WEBHOOK_SHARED_SECRET", "")

app = FastAPI(title="Query–Retrieval API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"]
)

# Helpers to satisfy type checkers for default_factory
def _empty_httpurl_list() -> List[HttpUrl]:
    return []

def _empty_webhookfile_list() -> List['WebhookFile']:
    return []

class UrlQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    urls: List[HttpUrl] = Field(default_factory=_empty_httpurl_list)

    @field_validator('urls')
    @classmethod
    def _limit_urls(cls, v: List[HttpUrl]) -> List[HttpUrl]:
        if len(v) > 10:
            raise ValueError('Too many URLs; max 10')
        return v

class NodeFile(TypedDict):
    filename: str
    content: str

class NodePayload(TypedDict):
    query: str
    urls: List[str]
    files: List[NodeFile]

# --- New models for Webhook JSON submissions ---
class WebhookFile(BaseModel):
    filename: str
    contentLatin1: Optional[str] = None
    contentBase64: Optional[str] = None

class WebhookSubmission(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    urls: List[HttpUrl] = Field(default_factory=_empty_httpurl_list)
    files: List[WebhookFile] = Field(default_factory=_empty_webhookfile_list)

    @field_validator('urls')
    @classmethod
    def _limit_urls(cls, v: List[HttpUrl]) -> List[HttpUrl]:
        if len(v) > 10:
            raise ValueError('Too many URLs; max 10')
        return v


def _validate_extension(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def _post_json_sync(url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: int = 75) -> Dict[str, Any]:
    data = json.dumps(dict(payload)).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST')
    req.add_header('Content-Type', 'application/json')
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return json.loads(body.decode('utf-8'))


@app.post("/query")
async def query_endpoint(
    request: Request,
    query: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """Accepts multipart with files or pure JSON with URLs. Forwards to Node service.
    Security: adds X-Internal-Shared-Secret header.
    """
    if not INTERNAL_SHARED_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured")

    payload: NodePayload = {"query": "", "urls": [], "files": []}

    # Multipart/form-data path
    if query:
        payload["query"] = query.strip()

    if files:
        for f in files:
            fname = f.filename or "upload.bin"
            if not _validate_extension(fname):
                raise HTTPException(status_code=400, detail=f"Unsupported file: {fname}")
            content = await f.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(status_code=413, detail=f"File too large: {fname}")
            payload["files"].append({"filename": fname, "content": content.decode("latin1")})

    # JSON body path when not multipart
    if (not payload["query"]) and (request.headers.get("content-type") or "").startswith("application/json"):
        try:
            data = await request.json()
            uq = UrlQuery(**data)
            payload["query"] = uq.query.strip()
            urls_list: List[str] = [str(u) for u in uq.urls]
            payload["urls"] = urls_list
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not payload["query"]:
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        result: Dict[str, Any] = await asyncio.to_thread(
            _post_json_sync,
            f"{NODE_SERVICE_URL}/process",
            payload,
            {"X-Internal-Shared-Secret": INTERNAL_SHARED_SECRET},
            75
        )
        return JSONResponse(result)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode('utf-8')
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=e.code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- New webhook endpoint: accepts JSON payload; outputs JSON ---
@app.post("/webhook/submission")
async def webhook_submission(request: Request):
    """Webhook endpoint for JSON submissions.
    Accepts payload: { query, urls[], files[]: { filename, contentLatin1? | contentBase64? } }
    Verifies optional WEBHOOK_SHARED_SECRET via 'X-Webhook-Secret' header.
    Forwards to Node service and returns JSON response.
    """
    if not INTERNAL_SHARED_SECRET:
        raise HTTPException(status_code=500, detail="Server misconfigured")

    if WEBHOOK_SHARED_SECRET:
        provided = request.headers.get('X-Webhook-Secret')
        if provided != WEBHOOK_SHARED_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized webhook")

    try:
        data = await request.json()
        sub = WebhookSubmission(**data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    payload: NodePayload = {"query": sub.query.strip(), "urls": [str(u) for u in sub.urls], "files": []}

    # Convert files to Node payload format
    for wf in sub.files:
        if wf.contentLatin1 is None and wf.contentBase64 is None:
            continue
        content_bytes: bytes
        if wf.contentLatin1 is not None:
            content_bytes = wf.contentLatin1.encode('latin1', errors='ignore')
        else:
            import base64
            try:
                b64: str = cast(str, wf.contentBase64)
                content_bytes = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid base64 for file {wf.filename}")
        # Enforce size limit
        size_mb = len(content_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=413, detail=f"File too large: {wf.filename}")
        if not _validate_extension(wf.filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file: {wf.filename}")
        payload["files"].append({"filename": wf.filename, "content": content_bytes.decode('latin1')})

    try:
        result: Dict[str, Any] = await asyncio.to_thread(
            _post_json_sync,
            f"{NODE_SERVICE_URL}/process",
            payload,
            {"X-Internal-Shared-Secret": INTERNAL_SHARED_SECRET},
            90
        )
        return JSONResponse(result)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode('utf-8')
        except Exception:
            detail = str(e)
        raise HTTPException(status_code=e.code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # Local/dev entrypoint. Railway will set PORT; default to 8000.
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
