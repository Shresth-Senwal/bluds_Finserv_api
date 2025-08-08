# LLM-Powered Intelligent Query–Retrieval API (Backend-Only)

This workspace contains two backend-only services:

- `python-fastapi/`: Public HTTP API layer (FastAPI). Handles uploads, URLs, and queries. Calls Node.js service for heavy processing.
- `node-backend/`: Processing microservice implementing embeddings (BERT), Pinecone semantic search, clause retrieval, and Gemini 2.5 Flash Lite reasoning. Persists logs to MongoDB Atlas.

Both services read configuration from environment variables. No front end is included.

## High-Level Flow
1. FastAPI receives multipart/form-data (files) or JSON (URLs + query), validates inputs, and stores temp files.
2. FastAPI calls Node service `/process` with a signed internal token, sending URLs, extracted text, and query.
3. Node performs:
   - Text extraction (PDF/DOCX/EML)
   - Chunking and BERT embeddings
   - Upsert/search against Pinecone index
   - Clause retrieval/matching
   - Prompt Gemini 2.5 Flash Lite with retrieved context
   - Return structured JSON with answer, clauses, rationale, metadata
   - Persist audit log to MongoDB Atlas
4. FastAPI returns the Node response as-is to clients.

See each subfolder for setup.
