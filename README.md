# Doc QA

A documentation Q&A system that indexes your docs and answers questions using retrieval-augmented generation (RAG). It ships with a polished web UI, streaming responses, Mermaid diagram generation, conversation history, and a guided setup experience.

```
Your docs  ──►  Index (LanceDB)  ──►  Hybrid search  ──►  LLM  ──►  Streaming answer
  .md .adoc       embeddings           vector + BM25       Cody       with sources,
  .pdf .puml      chunking             reranking           Ollama     diagrams, and
                  change detection     diversity cap                  confidence scores
```

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [LLM Backend](#llm-backend)
  - [Retrieval Settings](#retrieval-settings)
  - [Database (Optional)](#database-optional)
  - [Intelligence Features](#intelligence-features)
  - [Diagram Generation](#diagram-generation)
  - [Answer Verification](#answer-verification)
- [CLI Commands](#cli-commands)
- [Web UI](#web-ui)
- [Environment Variables](#environment-variables)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Supported Document Formats](#supported-document-formats)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
  - [Embedding Model Download Fails](#embedding-model-download-fails)
  - [Corrupt Embedding Model Cache](#corrupt-embedding-model-cache)
  - [Offline Usage](#offline-usage)

---

## Prerequisites

| Dependency | Version | Required | Purpose |
|------------|---------|----------|---------|
| Python | >= 3.11 | Yes | Backend runtime |
| Node.js | >= 18 | Yes | Frontend build + Mermaid validation |
| npm | >= 9 | Yes | Frontend package management |
| PostgreSQL | >= 14 | No | Conversation persistence (optional) |
| Ollama | any | No | Local LLM fallback (optional) |

**LLM access** — you need at least one of:

- **Sourcegraph Cody** — a [Sourcegraph access token](https://sourcegraph.com/sign-up) (free tier available), OR
- **Ollama** — a running [Ollama](https://ollama.com) instance with a pulled model

---

## Quick Start

### 1. Clone and install the backend

```bash
cd doc_qa_tool

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install the package
pip install -e ".[dev]"

# If you want Ollama support:
pip install -e ".[ollama]"

# Create your local config from the template
cp config.yaml.example config.yaml
```

> **Note:** `config.yaml` is gitignored (machine-specific). Each machine gets its own copy from the template.

### 2. Set your LLM token

If using Cody (the default):

```bash
export SRC_ACCESS_TOKEN="your-sourcegraph-access-token"
```

Get a token from **Sourcegraph** > **Settings** > **Access tokens**.

If using Ollama instead, skip this step and [change the config](#llm-backend).

### 3. Index your documentation

```bash
doc-qa index /path/to/your/docs
```

This scans for `.md`, `.adoc`, `.puml`, and `.pdf` files, splits them into chunks, generates embeddings, and stores everything in a local LanceDB database at `./data/doc_qa_db`.

```
Scanning '/path/to/your/docs' for documents...
Found 42 files.
Processing 42 files...

  auth.md: 3 sections -> 5 chunks [new]
  deploy.adoc: 7 sections -> 12 chunks [new]
  ...

Done in 8.3s.
Index: 186 chunks from 42 files.
```

Re-running `doc-qa index` is **incremental** — only new or changed files are re-processed.

### 4. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

This produces an optimized SPA in `frontend/dist/`.

### 5. Download the embedding model (one-time)

The embedding model (~90 MB) is needed for indexing and search. It downloads automatically on first run, but if you're on a restricted network you can download it explicitly:

```bash
doc-qa download-model
```

**If that also fails** (corporate proxy, restricted network), download via your browser and install:

1. Download: https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz
2. Run: `doc-qa install-model /path/to/sentence-transformers-all-MiniLM-L6-v2.tar.gz`

The model is cached in `data/models/` inside the project — it only downloads once and works offline after that. See [Embedding Model Download Fails](#embedding-model-download-fails) for more details.

### 6. Start the server

```bash
doc-qa serve --repo /path/to/your/docs
```

The `--repo` flag is **optional**. Without it, the server starts and you can configure the repository path from the **Settings > Indexing** tab in the web UI.

Open **http://localhost:8000** in your browser. You'll see:

- A welcome screen with example questions and index statistics
- A guided tour on first visit that walks through the settings
- A chat interface with streaming answers

### 7. (Optional) Set up diagram validation

For full Mermaid diagram syntax validation with Node.js:

```bash
cd scripts
npm install
cd ..
```

Without this, diagrams are validated using regex heuristics (still catches most syntax errors).

---

## Configuration

All settings live in **`config.yaml`** at the project root. The server reads it on startup. Most settings can also be changed at runtime through the **Settings** dialog in the web UI (gear icon in the header).

### LLM Backend

```yaml
llm:
  primary: "cody"        # "cody" or "ollama"
  fallback: "ollama"     # used when primary fails

cody:
  access_token_env: "SRC_ACCESS_TOKEN"    # env var name holding your token
  endpoint: "https://sourcegraph.com"
  model: "anthropic::2025-01-01::claude-3.5-sonnet"
  agent_binary: null     # auto-downloads if null

ollama:
  host: "http://localhost:11434"
  model: "qwen2.5:7b"
```

**To use Ollama as the primary backend:**

1. Install [Ollama](https://ollama.com)
2. Pull a model: `ollama pull qwen2.5:7b`
3. Change `config.yaml`:
   ```yaml
   llm:
     primary: "ollama"
     fallback: null
   ```
4. Restart the server

> Changing the LLM backend requires a server restart. The Settings UI will show a "Restart required" badge.

### Retrieval Settings

```yaml
retrieval:
  search_mode: "hybrid"      # "hybrid" | "vector" | "fts"
  top_k: 5                   # final chunks returned to the LLM
  candidate_pool: 20         # candidates fetched before reranking
  min_score: 0.3             # minimum similarity threshold (0.0 – 1.0)
  max_chunks_per_file: 2     # diversity cap — prevents one file dominating
  rerank: true               # enable cross-encoder reranking
```

| Mode | How it works | Best for |
|------|-------------|----------|
| `hybrid` | Vector search + BM25 full-text search + Reciprocal Rank Fusion | General use (default) |
| `vector` | Cosine similarity on embeddings only | Semantic/conceptual queries |
| `fts` | BM25 keyword search only | Exact term/keyword lookups |

These settings can be changed live from the **Retrieval** tab in the Settings dialog — no restart needed.

### Database (Optional)

By default, Doc QA works **without a database**. Conversations exist only in the browser session.

To persist conversation history across sessions:

1. Create a PostgreSQL database:
   ```sql
   CREATE DATABASE doc_qa;
   ```

2. **Option A — config.yaml:**
   ```yaml
   database:
     url: "postgresql+asyncpg://user:password@localhost:5432/doc_qa"
   ```

   **Option B — environment variable:**
   ```bash
   export DOC_QA_DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/doc_qa"
   ```

3. Run migrations:
   ```bash
   doc-qa db upgrade
   ```

4. Restart the server. A conversation sidebar will appear in the UI.

**Or use the Settings UI**: Open the **Database** tab, paste your connection URL, click **Test Connection**, then **Run Migrations**.

### Intelligence Features

```yaml
intelligence:
  enable_intent_classification: true    # detect query type (diagram, code, comparison, etc.)
  intent_confidence_high: 0.85          # above this → specialized generator only
  intent_confidence_medium: 0.65        # above this → specialized + explanation
  enable_multi_intent: true             # split compound queries into sub-queries
  max_sub_queries: 3
```

When enabled, the system automatically detects what the user is asking for:

| Detected Intent | Generator | Example |
|----------------|-----------|---------|
| DIAGRAM | Mermaid diagram + explanation | "Show the OAuth flow" |
| CODE_EXAMPLE | Code with language tags | "Show me a curl example" |
| COMPARISON_TABLE | Markdown table | "Compare JWT vs sessions" |
| PROCEDURAL | Numbered steps | "How do I deploy?" |
| EXPLANATION | Standard prose answer | "What is the rate limit?" |

### Diagram Generation

```yaml
generation:
  enable_diagrams: true
  mermaid_validation: "auto"           # "node" | "regex" | "auto" | "none"
  node_script_path: "scripts/validate_mermaid.mjs"
  max_diagram_retries: 3              # repair attempts for invalid diagrams
```

| Validation mode | Behavior |
|----------------|----------|
| `auto` | Try Node.js validator first, fall back to regex |
| `node` | Node.js only (requires `scripts/node_modules`) |
| `regex` | Fast heuristic checks (bracket balance, type recognition) |
| `none` | Skip validation entirely |

When a diagram fails validation, the system sends it back to the LLM with the error message for repair, up to `max_diagram_retries` times.

Supported diagram types: flowchart, sequence, class, state, ER, gantt, pie, git graph, mindmap, timeline, sankey, XY chart, block, quadrant, requirement, and C4 diagrams.

### Answer Verification

```yaml
verification:
  enable_verification: true            # generate-then-verify (fact-check answers)
  enable_crag: true                    # corrective RAG (re-retrieve on low relevance)
  confidence_threshold: 0.4            # below this → abstain
  max_crag_rewrites: 2                # max query rewrite attempts
  abstain_on_low_confidence: true      # say "I don't know" rather than guess
```

The verification pipeline:

1. **CRAG** — grades retrieved documents for relevance. If mostly irrelevant, rewrites the query and re-retrieves.
2. **Verification** — after generating an answer, asks the LLM to fact-check it against the source documents.
3. **Confidence scoring** — combines retrieval scores and verification results into a 0–1 confidence score.
4. **Abstention** — if confidence is below the threshold, returns "I don't have enough information" instead of a potentially wrong answer.

---

## CLI Commands

```
doc-qa <command> [options]
```

### `index` — Index documentation

```bash
doc-qa index /path/to/docs [--config config.yaml]
```

Scans for supported files, parses sections, chunks text, generates embeddings, and stores in LanceDB. Incremental — only processes new or changed files.

### `query` — Ask a question (CLI mode)

```bash
doc-qa query "How does authentication work?" --repo /path/to/docs
```

Options:
- `--retrieval-only` — show retrieved chunks without sending to the LLM (useful for debugging)

### `serve` — Start the web server

```bash
doc-qa serve [--repo /path/to/docs] [--host 127.0.0.1] [--port 8000]
```

Starts a FastAPI server that serves both the API and the web UI. The `--repo` flag is optional — if omitted, the server falls back to `doc_repo.path` in `config.yaml`, or starts without a repo (configurable from the Settings UI).

On startup, the server pre-downloads the embedding model if needed and then enables **offline mode** — no further internet calls are made (except for Cody/Ollama LLM backends).

- **UI**: http://localhost:8000
- **API docs (Swagger)**: http://localhost:8000/docs

### `download-model` — Download the embedding model

```bash
doc-qa download-model
```

Downloads the embedding model from Google Cloud Storage (~90 MB, one-time) and stores it in `data/models/`. This is useful when:

- You're setting up on a new machine
- The automatic download during `serve` failed (e.g., corporate proxy)
- You want to pre-download before going offline

After downloading, you can copy the `data/models/` folder to other machines for fully offline setup.

### `install-model` — Install model from a downloaded file

```bash
doc-qa install-model /path/to/sentence-transformers-all-MiniLM-L6-v2.tar.gz
```

Installs the embedding model from a `.tar.gz` file you downloaded manually in your browser. This is the **recommended fallback** when automatic downloads fail (corporate networks, proxy issues, restricted internet).

The command handles everything automatically:
1. Cleans up any corrupt cache from previous failed downloads
2. Extracts the tar.gz to the correct location
3. Verifies the model loads and produces embeddings

See [Embedding Model Download Fails](#embedding-model-download-fails) for the full workflow.

### `eval` — Evaluate retrieval quality

```bash
doc-qa eval --test-cases test_cases.json --repo /path/to/docs
```

Runs retrieval evaluation against a test dataset and prints a pass/fail report.

### `db` — Run database migrations

```bash
doc-qa db upgrade [revision]    # upgrade to head (or specific revision)
doc-qa db downgrade [revision]  # downgrade by 1 (or to specific revision)
```

### Global Options

```bash
doc-qa --config /path/to/config.yaml --log-level DEBUG <command>
```

Log levels: `DEBUG`, `INFO`, `WARNING` (default), `ERROR`.

---

## Web UI

### First Visit

On your first visit, a **guided tour** walks you through the settings:

1. **Welcome** — overview of the setup
2. **LLM Backend** — verify your AI model and endpoint
3. **Database** — optionally set up PostgreSQL for conversation persistence
4. **Retrieval** — review search settings (defaults work well)
5. **Advanced** — mention of fine-grained controls
6. **Complete** — close and start asking questions

The tour can be restarted anytime from the **"Take a Tour"** link at the bottom of the page.

### Features

- **Streaming answers** — see the answer build in real-time with phase indicators (classifying, retrieving, generating, verifying)
- **Markdown rendering** — headings, lists, tables, code blocks with syntax highlighting
- **Mermaid diagrams** — rendered inline as interactive SVGs
- **Source attributions** — see which documents each part of the answer came from
- **Confidence badge** — color-coded confidence score on each answer
- **Conversation history** — sidebar with past conversations (requires PostgreSQL)
- **Dark/Light mode** — toggle in the header
- **Settings** — full config editor with per-section save and live reload
- **Connection status** — real-time indicator of backend connectivity
- **New chat** — start a fresh conversation with the `+` button

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `SRC_ACCESS_TOKEN` | If using Cody | — | Sourcegraph access token |
| `SRC_ENDPOINT` | No | `https://sourcegraph.com` | Sourcegraph instance URL |
| `DOC_QA_DATABASE_URL` | No | — | PostgreSQL URL (overrides config) |
| `FASTEMBED_CACHE_PATH` | No | `data/models/` | Override the embedding model cache directory |
| `CODY_AGENT_BINARY` | No | auto-download | Path to Cody agent binary |
| `CODY_AGENT_TRACE_PATH` | No | — | Write JSON-RPC trace log (debug) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Browser                          │
│  React 19 + Tailwind v4 + Streamdown + Framer Motion        │
│  SSE streaming  ──►  Markdown + Mermaid rendering            │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Server                            │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │  REST API    │  │  SSE Stream │  │  Settings API    │    │
│  │  /api/query  │  │  /api/query │  │  GET/PATCH       │    │
│  │  /api/stats  │  │  /stream    │  │  /api/config     │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────────────┘    │
│         │                │                                   │
│  ┌──────▼────────────────▼─────────────────────────────┐    │
│  │              Query Pipeline                          │    │
│  │                                                      │    │
│  │  Intent Classification ──► Query Decomposition       │    │
│  │         │                                            │    │
│  │  Hybrid Retrieval (vector + BM25 + RRF)              │    │
│  │         │                                            │    │
│  │  Reranking ──► CRAG (corrective re-retrieval)        │    │
│  │         │                                            │    │
│  │  Specialized Generation (diagram/code/table/steps)   │    │
│  │         │                                            │    │
│  │  Answer Verification ──► Confidence Scoring           │    │
│  │         │                                            │    │
│  │  Source Attribution                                   │    │
│  └──────┬─────────────────────────────┬────────────────┘    │
│         │                             │                      │
│  ┌──────▼──────┐             ┌───────▼───────┐              │
│  │  LanceDB    │             │  LLM Backend  │              │
│  │  (embedded) │             │  Cody / Ollama│              │
│  └─────────────┘             └───────────────┘              │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐         │
│  │  PostgreSQL (opt.)   │  │  config.yaml          │         │
│  │  conversations       │  │  all settings          │         │
│  └──────────────────────┘  └──────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **No external vector DB** — LanceDB is embedded (zero cloud dependencies)
- **Incremental indexing** — SHA-256 change detection; only re-processes changed files
- **Lazy LLM init** — backend connection only created on first query
- **Pluggable parsers** — registry pattern for document format support
- **Phased streaming** — each pipeline step emits an SSE event for granular UI feedback
- **Config hot-reload** — safe sections (retrieval, intelligence, verification) apply immediately; unsafe sections (LLM, indexing) require restart
- **Same-origin SPA** — FastAPI serves the built frontend; no CORS needed

---

## API Reference

### Query

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/query` | Full Q&A (returns JSON) |
| `GET` | `/api/query/stream?q=...&session_id=...` | Streaming Q&A (SSE) |
| `POST` | `/api/retrieve` | Retrieval only (no LLM) |
| `GET` | `/api/stats` | Index statistics |
| `GET` | `/api/health` | Health check |

### Configuration

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/config` | Current config (secrets redacted) |
| `PATCH` | `/api/config` | Update config sections |
| `POST` | `/api/config/db/test` | Test a database connection |
| `POST` | `/api/config/db/migrate` | Run database migrations |

### Conversations (requires PostgreSQL)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{id}` | Get conversation with messages |
| `PATCH` | `/api/conversations/{id}` | Rename conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |

### SSE Event Types

The streaming endpoint emits events in this order:

```
status("classifying") → intent → status("retrieving") → sources →
status("grading") → status("generating") → answer →
status("verifying") → verified → done
```

| Event | Payload | Description |
|-------|---------|-------------|
| `status` | `{status, session_id?}` | Pipeline phase update |
| `intent` | `{intent, confidence}` | Detected query intent |
| `sources` | `[{file_path, section_title, score}]` | Retrieved source documents |
| `answer` | `{answer, model, session_id, diagrams?}` | Final answer text |
| `verified` | `{passed, reason?}` | Verification result |
| `done` | `{status: "complete", elapsed}` | Stream complete |
| `error` | `{error, type}` | Error occurred |

---

## Supported Document Formats

| Extension | Parser | Features |
|-----------|--------|----------|
| `.md` | Markdown | Front matter extraction, heading hierarchy |
| `.adoc` | AsciiDoc | Heading-based section splitting |
| `.puml` | PlantUML | Diagram text extraction |
| `.pdf` | PDF | Page-by-page text extraction via pdfplumber |

Default scan patterns: `**/*.adoc`, `**/*.md`, `**/*.puml`, `**/*.pdf`

Default exclusions: `**/build/**`, `**/target/**`, `**/node_modules/**`, `**/.git/**`

Customize in `config.yaml`:

```yaml
doc_repo:
  patterns:
    - "**/*.md"
    - "**/*.rst"          # add more patterns
  exclude:
    - "**/vendor/**"      # add more exclusions
```

---

## Development

### Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ollama]"
```

### Frontend setup

```bash
cd frontend
npm install
npm run dev          # starts Vite dev server at localhost:5173
```

The Vite dev server proxies `/api/*` requests to `http://127.0.0.1:8000`, so you can run the backend and frontend separately during development:

**Terminal 1:**
```bash
doc-qa serve --repo /path/to/docs --log-level DEBUG
```

**Terminal 2:**
```bash
cd frontend && npm run dev
```

Then open **http://localhost:5173** (Vite, with hot reload).

### Running tests

**Backend:**
```bash
pytest                           # all tests
pytest tests/test_generators.py  # specific file
pytest -x -q                     # stop on first failure, quiet output
```

**Frontend:**
```bash
cd frontend
npm test                         # run once
npm run test:watch               # watch mode
```

### Linting

```bash
ruff check doc_qa/               # lint
ruff format doc_qa/              # format
cd frontend && npm run lint      # ESLint
cd frontend && npm run typecheck # TypeScript check
```

### Project structure

```
doc_qa_tool/
├── doc_qa/                     # Python backend
│   ├── __main__.py             # CLI entry point (index, query, serve, db, eval)
│   ├── config.py               # Configuration dataclasses + YAML loading
│   ├── api/
│   │   └── server.py           # FastAPI app factory + all endpoints
│   ├── indexing/
│   │   ├── scanner.py          # File discovery with dedup
│   │   ├── chunker.py          # Section-based text chunking
│   │   ├── embedder.py         # fastembed integration
│   │   └── indexer.py          # LanceDB vector store
│   ├── retrieval/
│   │   ├── retriever.py        # Hybrid search (vector + BM25 + RRF)
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   ├── query_pipeline.py   # Full query orchestration
│   │   └── corrective.py       # CRAG (corrective retrieval)
│   ├── llm/
│   │   ├── backend.py          # LLM abstraction (Cody + Ollama)
│   │   └── prompt_templates.py # All prompt templates
│   ├── intelligence/
│   │   ├── intent_classifier.py    # Query intent detection
│   │   ├── query_analyzer.py       # Multi-intent decomposition
│   │   ├── confidence.py           # Confidence scoring
│   │   └── output_detector.py      # Output format detection
│   ├── generation/
│   │   ├── router.py           # Intent → generator mapping
│   │   ├── diagram.py          # Mermaid diagram generator + repair loop
│   │   ├── code_example.py     # Code example generator
│   │   ├── comparison.py       # Comparison table generator
│   │   ├── procedural.py       # Step-by-step generator
│   │   └── explanation.py      # Default prose generator
│   ├── verification/
│   │   ├── verifier.py         # Answer fact-checking
│   │   ├── grader.py           # Document relevance grading
│   │   ├── mermaid_validator.py    # Two-tier Mermaid validation
│   │   └── source_attributor.py    # Sentence → source mapping
│   ├── streaming/
│   │   └── sse.py              # Phased SSE event streaming
│   ├── parsers/
│   │   ├── registry.py         # File extension → parser dispatch
│   │   ├── markdown.py         # Markdown parser
│   │   ├── asciidoc.py         # AsciiDoc parser
│   │   ├── pdf.py              # PDF parser
│   │   └── plantuml.py         # PlantUML parser
│   └── db/
│       ├── engine.py           # Async SQLAlchemy engine
│       ├── models.py           # ORM models (Conversation, Message)
│       ├── repository.py       # DB CRUD operations
│       └── migrations/         # Alembic migration scripts
├── frontend/                   # React SPA
│   ├── src/
│   │   ├── App.tsx             # Root component
│   │   ├── api/                # HTTP + SSE clients
│   │   ├── components/         # UI components (chat, settings, sidebar)
│   │   ├── hooks/              # Custom React hooks
│   │   ├── types/              # TypeScript interfaces
│   │   └── test/               # Frontend tests
│   ├── package.json
│   └── vite.config.ts
├── data/
│   └── models/                 # Embedding model cache (auto-downloaded)
├── scripts/
│   └── validate_mermaid.mjs    # Node.js Mermaid syntax validator
├── tests/                      # Backend tests (pytest)
├── config.yaml                 # Application configuration (gitignored, machine-specific)
├── config.yaml.example         # Template for config.yaml
├── alembic.ini                 # Migration configuration
└── pyproject.toml              # Python package metadata
```

---

## Troubleshooting

### "Index is empty. Run 'doc-qa index' first."

You need to index your documentation before querying:

```bash
doc-qa index /path/to/your/docs
```

### "SRC_ACCESS_TOKEN is not set"

The Cody backend requires a Sourcegraph access token:

```bash
export SRC_ACCESS_TOKEN="sgp_your_token_here"
```

Or switch to Ollama in `config.yaml` if you prefer local inference.

### Ollama connection refused

Make sure Ollama is running and a model is pulled:

```bash
ollama serve                     # start the server
ollama pull qwen2.5:7b           # pull the default model
```

### Frontend shows "Disconnected"

The backend server isn't running. Start it:

```bash
doc-qa serve --repo /path/to/docs
```

### Database migration fails

Check your connection string format:

```
postgresql+asyncpg://username:password@host:port/database
```

Common issues:
- Password contains special characters — URL-encode them (`@` → `%40`)
- PostgreSQL isn't running or the database doesn't exist
- The `asyncpg` driver is required (included in dependencies)

### No documents found during indexing

Check your file patterns in `config.yaml`. The default patterns are:

```yaml
doc_repo:
  patterns:
    - "**/*.adoc"
    - "**/*.md"
    - "**/*.puml"
    - "**/*.pdf"
```

Make sure your documentation files match these patterns and aren't in excluded directories.

### Diagrams not rendering

1. Check that `generation.enable_diagrams` is `true` in config
2. For best validation, install the Node.js validator: `cd scripts && npm install`
3. The frontend renders Mermaid diagrams from ` ```mermaid ` code fences in the answer — they should appear as interactive SVGs

### Embedding model download fails

The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is required for indexing and search. On startup, `doc-qa serve` tries two automatic download methods:

1. **HuggingFace Hub** (default fastembed behavior)
2. **Google Cloud Storage** (fallback — more reliable on corporate networks)

If both fail (e.g., heavily restricted network, proxy issues), follow one of these options:

#### Option A — Browser download + install (recommended)

Two steps — download in your browser, then run one command:

1. **Download** this file in your browser (works even on restricted networks):

   ```
   https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz
   ```

2. **Install** it with one command:

   ```bash
   doc-qa install-model /path/to/sentence-transformers-all-MiniLM-L6-v2.tar.gz
   ```

   On Windows, the path might look like:
   ```powershell
   doc-qa install-model C:\Users\you\Downloads\sentence-transformers-all-MiniLM-L6-v2.tar.gz
   ```

   This command automatically:
   - Cleans up any corrupt cache from previous failed downloads
   - Extracts the model to `data/models/`
   - Verifies it loads correctly and produces embeddings

3. Run `doc-qa serve` — the model is now cached locally and no internet is needed.

#### Option B — Corporate proxy

If your network requires a proxy, set the proxy environment variable before downloading:

**Windows (PowerShell):**
```powershell
$env:HTTPS_PROXY = "http://your-proxy:8080"
doc-qa download-model
```

**macOS / Linux:**
```bash
export HTTPS_PROXY=http://your-proxy:8080
doc-qa download-model
```

#### Option C — Copy from another machine

If you have the model on another machine, copy the entire `data/models/` folder to the same location in the project on the new machine:

```
<project-root>/data/models/fast-all-MiniLM-L6-v2/
├── model.onnx          (~90 MB)
├── config.json
├── tokenizer.json
├── special_tokens_map.json
└── tokenizer_config.json
```

### Corrupt embedding model cache

If the model download was interrupted (partial download), you may see errors like:

```
[ONNXRuntimeError] : 3 : NO_SUCHFILE : Load model from ... model.onnx failed
```

The tool automatically cleans up corrupt caches on startup, but if the issue persists:

1. Delete the cache directory:
   ```bash
   rm -rf data/models/
   ```
   On Windows:
   ```powershell
   Remove-Item -Recurse -Force data\models
   ```

2. Re-download:
   ```bash
   doc-qa download-model
   ```

### Slow first query

The first query triggers lazy initialization:
- LLM backend connection (Cody agent startup or Ollama connection)

The embedding model is pre-loaded during `doc-qa serve` startup, so it doesn't slow down the first query.

### Offline usage

After `doc-qa serve` starts successfully, the tool operates **fully offline** — no internet calls are made except for:

- **Cody** LLM backend (connects to Sourcegraph)
- **Ollama** LLM backend (connects to your local Ollama instance)

The embedding model, index, and all retrieval happen locally. To set up for a fully offline environment:

1. Run `doc-qa download-model` once with internet
2. Index your docs: `doc-qa index /path/to/docs`
3. Disconnect from internet
4. Run `doc-qa serve` — works offline (use Ollama for fully local LLM)

---

## License

MIT
