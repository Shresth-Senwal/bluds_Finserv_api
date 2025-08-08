# Project Context: LLM-Powered Intelligent Query–Retrieval System

## Architecture Overview
- **FastAPI Service** (`python-fastapi/app.py`): Public API layer. Handles file/URL input, query validation, and forwards requests to Node.js backend. Returns structured JSON responses.
- **Node.js Backend** (`node-backend/src/server.js`): Handles document parsing, chunking, BERT embeddings, Pinecone semantic search, clause retrieval, LLM reasoning (Gemini 2.5 Flash Lite), and MongoDB Atlas logging.
- **MongoDB Atlas**: Stores audit logs of all API interactions.
- **Pinecone**: Vector database for semantic search and clause retrieval.
- **Gemini 2.5 Flash Lite**: LLM for contextual, explainable answers.

## Data Models
- **Query Request**
  - `query`: string (required)
  - `files`: list of PDF/DOCX/EML files (optional)
  - `urls`: list of document URLs (optional, max 10)
- **API Response**
  - `answer`: string
  - `clauses`: list of `{id, text, score, source}`
  - `rationale`: string
  - `metadata`: `{confidence, model, latency_ms, sources[], request_id}`
- **Audit Log (MongoDB)**
  - `ts`: timestamp
  - `query`: string
  - `urls`: list
  - `file_count`: int
  - `response`: full API response

## Features
- Upload or reference documents (PDF, DOCX, EML) via API.
- Parse natural language queries for insurance, legal, HR, compliance.
- Semantic search and clause matching using BERT embeddings and Pinecone.
- LLM-powered contextual answers with rationale and metadata.
- All interactions logged to MongoDB Atlas.
- Input validation, error handling, and security enforced.

## Endpoints
- `POST /query`: Accepts multipart (files) or JSON (URLs + query). Returns structured JSON answer.
- `GET /health`: Service health check.

## Outstanding Issues & Technical Debt
- **Test Coverage**: Add unit/integration tests for extraction, embedding, Pinecone queries, Gemini LLM prompt shaping, and streaming response handling.
- **Error Handling**: Improve error messages and edge case handling for file parsing, Pinecone, Gemini, and MongoDB failures.
- **Performance**: Profile chunking, embedding, Pinecone queries, and Gemini streaming for large documents. Consider async optimizations.
- **Security**: Rotate secrets regularly. Ensure `.env` is never committed. Monitor MongoDB Atlas for unauthorized access. Audit Pinecone and Gemini API usage.
- **Accessibility**: Not applicable (API only).
- **Internationalization**: All user-facing strings are externalized; ensure Gemini prompt supports multiple languages if required.

## Data Contracts & Integration Points
- **FastAPI → Node.js**: Internal POST `/process` with shared secret. Payload: `{query, files, urls}`.
- **Node.js → Pinecone**: Upsert/query vectors for semantic search.
- **Node.js → Gemini 2.5 Flash Lite**: Uses structured Content objects (role: "user", parts: [{ text }]), includes tool config (`url_context`, `googleSearch`), and supports streaming response. Prompts with query and context, parses JSON answer or wraps plain text. Implementation matches Google GenAI best practices.
- **Node.js → MongoDB Atlas**: Insert audit log for each request.

## Dependencies
- **Python**: fastapi, uvicorn, python-multipart, pydantic, httpx, python-dotenv
- **Node.js**: express, helmet, cors, dotenv, mongodb, pdf-parse, mammoth, mailparser, @xenova/transformers, @pinecone-database/pinecone, @google/generative-ai, joi
- **Config**: All secrets and keys in `.env` files only.

## Technical Notes
- All code is modular, testable, and documented per project standards.
- No hardcoded secrets or credentials; all secrets (MongoDB, Pinecone, Gemini) are stored in `.env` files only.
- Pinecone index is initialized on backend startup with integrated embedding model (`sentence-transformers/all-MiniLM-L6-v2` by default, can be swapped for BERT if supported). API key is loaded from environment.
- Gemini 2.5 Flash Lite integration uses Content objects, tool config, and streaming response for best performance and explainability.
- All file uploads and URLs are validated and sanitized.
- API responses are structured and explainable.
- Audit logs are stored for traceability and compliance.

---

_Last updated: 2025-08-08_
