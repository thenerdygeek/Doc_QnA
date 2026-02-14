/**
 * Canned API responses for E2E mock-mode tests.
 * These mirror the real backend's JSON shapes exactly.
 */

export const health = { status: "ok" };

export const stats = {
  total_chunks: 2_345,
  total_files: 42,
  db_path: "/data/index.db",
  embedding_model: "all-MiniLM-L6-v2",
};

export const conversationsList = [
  {
    id: "conv-1",
    user_id: null,
    title: "How does auth work?",
    created_at: "2026-02-10T10:00:00Z",
    updated_at: "2026-02-10T10:05:00Z",
  },
  {
    id: "conv-2",
    user_id: null,
    title: "REST API examples",
    created_at: "2026-02-09T14:00:00Z",
    updated_at: "2026-02-09T14:10:00Z",
  },
  {
    id: "conv-3",
    user_id: null,
    title: "",
    created_at: "2026-02-08T09:00:00Z",
    updated_at: "2026-02-08T09:01:00Z",
  },
];

export const conversationDetail = {
  id: "conv-1",
  user_id: null,
  title: "How does auth work?",
  created_at: "2026-02-10T10:00:00Z",
  updated_at: "2026-02-10T10:05:00Z",
  messages: [
    {
      id: "msg-1",
      role: "user" as const,
      content: "How does authentication work?",
      metadata: null,
      created_at: "2026-02-10T10:00:00Z",
    },
    {
      id: "msg-2",
      role: "assistant" as const,
      content:
        "Authentication uses JWT tokens. The flow is:\n\n1. User sends credentials to `/api/login`\n2. Server validates and returns a signed JWT\n3. Client includes the token in the `Authorization` header\n\n```python\ntoken = jwt.encode(payload, SECRET)\n```",
      metadata: null,
      created_at: "2026-02-10T10:00:05Z",
    },
  ],
};

export const configData = {
  llm: {
    primary: "cody",
  },
  cody: {
    model: "claude-sonnet-4-20250514",
    endpoint: "https://sourcegraph.example.com",
    access_token: "***REDACTED***",
  },
  ollama: {
    model: "qwen2.5:7b",
    endpoint: "http://localhost:11434",
  },
  retrieval: {
    top_k: 10,
    min_score: 0.3,
    rerank: true,
    rerank_top_k: 5,
  },
  intelligence: {
    intent_classification: true,
    confidence_threshold: 0.7,
  },
  generation: {
    max_tokens: 2048,
    temperature: 0.1,
    enable_diagrams: true,
  },
  verification: {
    enabled: true,
    crag_mode: "corrective",
    abstention_threshold: 0.3,
  },
  indexing: {
    chunk_size: 512,
    chunk_overlap: 64,
    embedding_model: "all-MiniLM-L6-v2",
  },
  database: {
    url: "",
    enabled: false,
  },
};

export const configDataWithDb = {
  ...configData,
  database: {
    url: "postgresql://user:pass@localhost:5432/docqa",
    enabled: true,
  },
};

export const configUpdateSuccess = {
  saved: true,
  restart_required: false,
  restart_sections: [] as string[],
};

export const configUpdateRestart = {
  saved: true,
  restart_required: true,
  restart_sections: ["llm"],
};

export const dbTestSuccess = { ok: true };
export const dbTestFailure = { ok: false, error: "Connection refused" };
export const dbMigrateSuccess = { ok: true, revision: "abc123def" };
