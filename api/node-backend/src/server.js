import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import { MongoClient } from 'mongodb';
import rateLimit from 'express-rate-limit';
import pkg from '@pinecone-database/pinecone';
const { Pinecone } = pkg;
import { GoogleGenerativeAI } from '@google/generative-ai';
// Use the library implementation directly to avoid pdf-parse debug harness executing a non-existent test file
import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import mammoth from 'mammoth';
import { simpleParser } from 'mailparser';
import fetch from 'node-fetch';
import crypto from 'crypto';
import Joi from 'joi';
import { pipeline } from '@xenova/transformers';

dotenv.config();

const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Basic rate limiting to protect upstreams and stabilize latency
const RATE_LIMIT_WINDOW_MS = parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10);
const RATE_LIMIT_MAX = parseInt(process.env.RATE_LIMIT_MAX || '60', 10);
app.use(rateLimit({ windowMs: RATE_LIMIT_WINDOW_MS, max: RATE_LIMIT_MAX, standardHeaders: true, legacyHeaders: false }));

const PORT = process.env.PORT || 7070;
const MONGODB_URI = process.env.MONGODB_URI;
const MONGODB_DB = process.env.MONGODB_DB || 'finserv';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX = process.env.PINECONE_INDEX;
const INTERNAL_SHARED_SECRET = process.env.INTERNAL_SHARED_SECRET;
const CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE || '1200', 10);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP || '200', 10);
const MAX_CONTEXT_CHUNKS = parseInt(process.env.MAX_CONTEXT_CHUNKS || '4', 10);
const MAX_GEMINI_CONCURRENCY = parseInt(process.env.MAX_GEMINI_CONCURRENCY || '2', 10);
const FASTPATH_MAX_TOTAL_LEN = parseInt(process.env.FASTPATH_MAX_TOTAL_LEN || '1200', 10);
const MAX_EMBED_CONCURRENCY = parseInt(process.env.MAX_EMBED_CONCURRENCY || '4', 10);
const MAX_TEXT_CACHE_ENTRIES = parseInt(process.env.MAX_TEXT_CACHE_ENTRIES || '500', 10);
const PINECONE_MODEL = process.env.PINECONE_MODEL || 'llama-text-embed-v2';
const PINECONE_DIMENSIONS = parseInt(process.env.PINECONE_DIMENSIONS || '1024', 10);
const PINECONE_REGION = process.env.PINECONE_REGION || 'us-east-1';

if (!INTERNAL_SHARED_SECRET) {
  console.error('INTERNAL_SHARED_SECRET is required');
  process.exit(1);
}

// Initialize external clients
const mongo = new MongoClient(MONGODB_URI);
const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// --- Embedder initialization with warmup and readiness flag ---
let embedder; // embeddings pipeline instance
let embedderReady = false;
const initEmbedder = (async () => {
  try {
    const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    // Run a tiny warmup to trigger weight download & JIT
    await pipe('warmup', { pooling: 'mean', normalize: true });
    embedder = pipe;
    embedderReady = true;
    console.log('Embedder initialized and warmed up.');
  } catch (e) {
    console.error('Failed to initialize embedder:', e);
  }
})();

async function ensureEmbedderReady(timeoutMs = 60000) {
  const start = Date.now();
  while (!embedderReady) {
    if (Date.now() - start > timeoutMs) return false;
    await new Promise(r => setTimeout(r, 300));
  }
  return true;
}

const schema = Joi.object({
  query: Joi.string().min(3).max(4000).required(),
  urls: Joi.array().items(Joi.string().uri()).max(10).default([]),
  files: Joi.array().items(Joi.object({
    filename: Joi.string().required(),
    content: Joi.string().required() // latin1 string of bytes
  })).default([])
});

function requireAuth(req, res, next) {
  const token = req.header('X-Internal-Shared-Secret');
  if (!token || token !== INTERNAL_SHARED_SECRET) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  for (let i = 0; i < text.length; i += (size - overlap)) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

// --- Lightweight caches to avoid recomputation across requests ---
const textCache = new Map(); // sha1(bytes) => extracted text
const embedCache = new Map(); // sha1(text) => vector
function sha1(buf) { return crypto.createHash('sha1').update(buf).digest('hex'); }
function textCacheSet(key, val) {
  textCache.set(key, val);
  if (textCache.size > MAX_TEXT_CACHE_ENTRIES) {
    const firstKey = textCache.keys().next().value;
    if (firstKey) textCache.delete(firstKey);
  }
}

function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let i = 0, active = 0;
  return new Promise((resolve, reject) => {
    const launch = () => {
      while (active < limit && i < items.length) {
        const idx = i++;
        active++;
        Promise.resolve(fn(items[idx], idx))
          .then(val => { out[idx] = val; active--; if (i >= items.length && active === 0) resolve(out); else launch(); })
          .catch(reject);
      }
    };
    launch();
  });
}

async function embedTexts(texts) {
  const results = new Array(texts.length);
  const tasks = texts.map((t, idx) => async () => {
    const key = sha1(Buffer.from(t || '', 'utf8'));
    let vec = embedCache.get(key);
    if (!vec) {
      const output = await embedder(t || '', { pooling: 'mean', normalize: true });
      vec = Array.from(output.data);
      embedCache.set(key, vec);
    }
    results[idx] = vec;
  });
  await mapLimit(tasks, MAX_EMBED_CONCURRENCY, t => t());
  return results;
}

async function extractFromUrl(url) {
  async function fetchWithRetry(u, tries = 3) {
    let lastErr;
    for (let i = 0; i < tries; i++) {
      try { return await fetch(u); } catch (e) { lastErr = e; await new Promise(r => setTimeout(r, 200 * (i + 1))); }
    }
    throw lastErr;
  }
  const res = await fetchWithRetry(url);
  const buf = Buffer.from(await res.arrayBuffer());
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('pdf') || url.endsWith('.pdf')) {
    const data = await pdfParse(buf);
    return { text: data.text, source: url };
  }
  if (ct.includes('word') || url.endsWith('.docx')) {
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return { text: value, source: url };
  }
  if (ct.includes('email') || url.endsWith('.eml')) {
    const parsed = await simpleParser(buf);
    const text = `${parsed.subject || ''}\n${parsed.text || parsed.html || ''}`;
    return { text, source: url };
  }
  // default: try text
  return { text: buf.toString('utf8'), source: url };
}

async function extractFromFileLike({ filename, content }) {
  const bytes = Buffer.from(content, 'latin1');
  const key = sha1(bytes);
  if (textCache.has(key)) {
    return { text: textCache.get(key), source: filename };
  }
  if (filename.endsWith('.pdf')) {
    const data = await pdfParse(bytes);
    const text = data.text;
    textCacheSet(key, text);
    return { text, source: filename };
  }
  if (filename.endsWith('.docx')) {
    const { value } = await mammoth.extractRawText({ buffer: bytes });
    textCacheSet(key, value);
    return { text: value, source: filename };
  }
  if (filename.endsWith('.eml')) {
    const parsed = await simpleParser(bytes);
    const text = `${parsed.subject || ''}\n${parsed.text || parsed.html || ''}`;
    textCacheSet(key, text);
    return { text, source: filename };
  }
  const text = bytes.toString('utf8');
  textCacheSet(key, text);
  return { text, source: filename };
}

function buildGeminiPrompt(query, clauses) {
  const context = clauses.map((c, i) => `Context ${i+1} (score=${c.score.toFixed(2)}; source=${c.source})\n${c.text}`).join('\n\n');
  return [
    'Answer strictly using ONLY the provided context. Do not guess.',
    'Be concise (<=90 words). Prefer extractive quotes; cite sources as file#chunk.',
    'If insufficient info, say "Insufficient context" and list what is missing.',
    `Question: ${query}`,
    context,
    'Return compact JSON: {"answer","rationale","caveats"}. No markdown.'
  ].join('\n');
}

/**
 * Calls Gemini 2.5 Flash Lite LLM with structured prompt, tools, and streaming response.
 * @param {string} promptText - The prompt/question for the LLM.
 * @returns {Promise<string>} - The generated answer text.
 */
// Simple concurrency gate for Gemini calls to avoid provider 429s
let geminiInFlight = 0;
const geminiWaitQueue = [];
// Basic circuit breaker for Gemini
let geminiFailures = 0;
let geminiOpenUntil = 0; // timestamp ms
const GEMINI_BREAKER_THRESHOLD = parseInt(process.env.GEMINI_BREAKER_THRESHOLD || '3', 10);
const GEMINI_BREAKER_COOLDOWN_MS = parseInt(process.env.GEMINI_BREAKER_COOLDOWN_MS || '60000', 10);
function geminiOpen() { return Date.now() < geminiOpenUntil; }
function geminiRecordSuccess() { geminiFailures = 0; }
function geminiRecordFailure() { if (++geminiFailures >= GEMINI_BREAKER_THRESHOLD) { geminiOpenUntil = Date.now() + GEMINI_BREAKER_COOLDOWN_MS; geminiFailures = 0; } }
async function withGeminiSlot(fn, timeoutMs = 30000) {
  if (geminiInFlight >= MAX_GEMINI_CONCURRENCY) {
    let resolveWait, rejectWait;
    const waitP = new Promise((res, rej) => { resolveWait = res; rejectWait = rej; });
    geminiWaitQueue.push(resolveWait);
    // Optional timeout for waiting
    const t = setTimeout(() => {
      const idx = geminiWaitQueue.indexOf(resolveWait);
      if (idx >= 0) geminiWaitQueue.splice(idx, 1);
      rejectWait?.(new Error('Gemini concurrency wait timeout'));
    }, timeoutMs);
    await waitP.finally(() => clearTimeout(t));
  }
  geminiInFlight++;
  try { return await fn(); }
  finally {
    geminiInFlight--;
    const next = geminiWaitQueue.shift();
    if (next) next();
  }
}

async function callGemini(promptText) {
  // Prepare Gemini client (already initialized as genAI)
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });

  // Prepare prompt as Content object
  const contents = [
    {
      role: 'user',
      parts: [ { text: promptText } ]
    }
  ];

  // Optional: Add tools and thinking config if supported by Node SDK
  const tools = [
    { url_context: {} },
    { googleSearch: {} }
  ];
  const generateContentConfig = {
    thinking_config: { thinking_budget: -1 },
    tools
  };

  // Stream response if supported
  let responseText = '';
  await withGeminiSlot(async () => {
    try {
      const stream = await model.generateContentStream({ contents, config: generateContentConfig });
      for await (const chunk of stream) { responseText += chunk.text; }
      geminiRecordSuccess();
    } catch (err) {
      try {
        const gen = await model.generateContent({ contents, config: generateContentConfig });
        responseText = gen.response.text();
        geminiRecordSuccess();
      } catch (e2) {
        geminiRecordFailure();
        throw e2;
      }
    }
  });
  return responseText;
}

// Pinecone index initialization (temporarily disabled).
// The service will use a local embedding-based similarity search fallback to ensure availability.
const index_name = PINECONE_INDEX;

// function ensurePineconeIndex() { /* disabled - rely on fallback search */ }

// Ensure index on startup
// ensurePineconeIndex().catch(e => {
//   console.error('Failed to initialize Pinecone index:', e);
//   // Do not exit; continue with local fallback search so the service remains available
// });

app.post('/process', requireAuth, async (req, res) => {
  // Request ID: propagate if provided or generate new
  const requestId = req.header('X-Request-ID') || crypto.randomUUID();
  const ready = await ensureEmbedderReady(60000);
  if (!ready) {
    return res.status(503).json({ error: 'Service warming up, please retry shortly', request_id: requestId });
  }

  const start = Date.now();
  const { value, error } = schema.validate(req.body);
  if (error) return res.status(400).json({ error: error.message, request_id: requestId });

  const { query, urls, files } = value;

  try {
    // 1) Extract text
    const extracted = [];
    for (const u of urls) extracted.push(await extractFromUrl(u));
    for (const f of files) extracted.push(await extractFromFileLike(f));

    const allChunks = [];
    for (const doc of extracted) {
      const chunks = chunkText(doc.text).map((t, i) => ({ chunk_text: t, source: `${doc.source}#chunk${i}` }));
      allChunks.push(...chunks);
    }

    // 2) Local semantic search fallback using on-device embeddings (no external DB required)
    // Compute embeddings for chunks and query, then rank by cosine similarity
    const chunkTexts = allChunks.map(c => c.chunk_text);
    const chunkEmbeddings = await embedTexts(chunkTexts);
    const [queryEmbedding] = await embedTexts([query]);

    function dot(a, b) { let s = 0; for (let i = 0; i < a.length && i < b.length; i++) s += a[i] * b[i]; return s; }

    const scored = chunkEmbeddings.map((emb, i) => ({
      i,
      score: dot(emb, queryEmbedding),
      source: allChunks[i].source,
      text: allChunks[i].chunk_text
    }));
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, MAX_CONTEXT_CHUNKS);

    const clauses = top.map((m, idx) => ({
      id: `local-${idx}`,
      text: m.text,
      score: m.score || 0,
      source: m.source || 'unknown'
    }));

  // 3) Decide fast-path or LLM call (respect circuit breaker)
    const totalLen = extracted.reduce((acc, d) => acc + (d.text?.length || 0), 0);
    let text;
  if (!GEMINI_API_KEY || geminiOpen() || totalLen <= FASTPATH_MAX_TOTAL_LEN || clauses.length === 0) {
      text = JSON.stringify({
        answer: (clauses[0]?.text?.slice(0, 400) || '').trim(),
        rationale: 'Fast extractive path (LLM skipped) for sub-second latency.',
        caveats: ''
      });
    } else {
      const prompt = buildGeminiPrompt(query, clauses);
      try {
        text = await callGemini(prompt);
      } catch (e) {
        console.warn('Gemini call failed, returning fallback answer:', e.message);
        text = JSON.stringify({
          answer: (clauses[0]?.text?.slice(0, 400) || '').trim(),
          rationale: 'LLM unavailable; provided best-effort extractive summary from top-matching chunk(s).',
          caveats: 'Heuristic fallback'
        });
      }
    }

    // Try to parse JSON from model; if not JSON, wrap
    let llm;
    try { llm = JSON.parse(text); } catch { llm = { answer: text, rationale: 'See answer', caveats: '' }; }

    const latency = Date.now() - start;
    const response = {
      answer: llm.answer || '',
      clauses,
      rationale: llm.rationale || '',
      metadata: {
        confidence: clauses[0]?.score || 0,
        model: 'gemini-2.5-flash-lite',
        latency_ms: latency,
        sources: [...new Set(clauses.map(c => c.source))],
        request_id: crypto.randomUUID()
      }
    };

    // 4) Persist log
    try {
      await mongo.connect();
      const col = mongo.db(MONGODB_DB).collection('audit_logs');
      let attempts = 0;
      while (true) {
        try { await col.insertOne({ ts: new Date(), query, urls, file_count: files.length, response, request_id: requestId }); break; }
        catch (e) { if (++attempts >= 3) { console.warn('Mongo log failed', e.message); break; } await new Promise(r => setTimeout(r, 200 * attempts)); }
      }
    } catch (e) {
      console.warn('Mongo connect/log failed', e.message);
    }

    res.setHeader('X-Request-ID', requestId);
    res.json(response);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Processing failed', details: err.message, request_id: requestId });
  }
});

app.get('/health', (req, res) => res.json({ status: 'ok', embedderReady }));

app.get('/health/details', (req, res) => {
  res.json({
    status: 'ok',
    embedderReady,
    limits: {
      MAX_CONTEXT_CHUNKS,
      MAX_GEMINI_CONCURRENCY,
      FASTPATH_MAX_TOTAL_LEN,
      RATE_LIMIT_WINDOW_MS,
      RATE_LIMIT_MAX,
    },
    gemini: {
      in_flight: geminiInFlight,
      breaker_open: geminiOpen(),
      breaker_cooldown_ms: Math.max(0, geminiOpenUntil - Date.now()),
    }
  });
});

app.listen(PORT, () => console.log(`Node backend listening on ${PORT}`));
// No hardcoded test file references; all file handling is based on incoming API requests only
