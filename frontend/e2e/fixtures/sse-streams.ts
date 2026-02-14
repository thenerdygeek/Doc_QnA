/**
 * Canned SSE event streams for E2E mock-mode tests.
 * Each function returns a string in SSE wire format (event + data lines).
 */

function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** Full successful Q&A stream with all 9 event types. */
export function fullStream(sessionId = "sess-abc-123"): string {
  return [
    sseEvent("status", { status: "classifying", session_id: sessionId }),
    sseEvent("intent", { intent: "how_to", confidence: 0.92 }),
    sseEvent("status", { status: "retrieving" }),
    sseEvent("sources", {
      sources: [
        { file_path: "docs/auth.md", section_title: "JWT Flow", score: 0.95 },
        {
          file_path: "docs/api.md",
          section_title: "Endpoints",
          score: 0.87,
        },
        {
          file_path: "docs/setup.md",
          section_title: "Configuration",
          score: 0.72,
        },
      ],
      chunks_retrieved: 8,
    }),
    sseEvent("status", { status: "generating" }),
    sseEvent("answer_token", { token: "The " }),
    sseEvent("answer_token", { token: "authentication " }),
    sseEvent("answer_token", { token: "system " }),
    sseEvent("answer_token", { token: "uses " }),
    sseEvent("answer_token", { token: "**JWT tokens**" }),
    sseEvent("answer_token", { token: " for " }),
    sseEvent("answer_token", { token: "secure " }),
    sseEvent("answer_token", { token: "access.\n\n" }),
    sseEvent("answer_token", { token: "## Steps\n\n" }),
    sseEvent("answer_token", { token: "1. Send credentials to `/api/login`\n" }),
    sseEvent("answer_token", { token: "2. Receive signed JWT\n" }),
    sseEvent("answer_token", { token: "3. Include in `Authorization` header\n\n" }),
    sseEvent("answer_token", { token: "```python\n" }),
    sseEvent("answer_token", { token: 'token = jwt.encode(payload, SECRET)\n' }),
    sseEvent("answer_token", { token: "```" }),
    sseEvent("answer", {
      answer:
        'The authentication system uses **JWT tokens** for secure access.\n\n## Steps\n\n1. Send credentials to `/api/login`\n2. Receive signed JWT\n3. Include in `Authorization` header\n\n```python\ntoken = jwt.encode(payload, SECRET)\n```',
      model: "claude-sonnet-4-20250514",
      session_id: sessionId,
      diagrams: [],
    }),
    sseEvent("status", { status: "verifying" }),
    sseEvent("attribution", {
      attributions: [
        {
          sentence: "The authentication system uses JWT tokens for secure access.",
          source_index: 0,
          similarity: 0.94,
        },
        {
          sentence: "Send credentials to /api/login.",
          source_index: 0,
          similarity: 0.91,
        },
      ],
    }),
    sseEvent("verified", { passed: true, confidence: 0.89 }),
    sseEvent("done", { status: "complete", elapsed: 2.34 }),
  ].join("");
}

/** Stream that ends in an error. */
export function errorStream(): string {
  return [
    sseEvent("status", { status: "classifying" }),
    sseEvent("status", { status: "retrieving" }),
    sseEvent("error", { error: "LLM service unavailable", type: "llm_error" }),
  ].join("");
}

/** Stream with unverified result. */
export function unverifiedStream(sessionId = "sess-xyz-789"): string {
  return [
    sseEvent("status", { status: "classifying", session_id: sessionId }),
    sseEvent("intent", { intent: "factual", confidence: 0.78 }),
    sseEvent("status", { status: "retrieving" }),
    sseEvent("sources", {
      sources: [
        { file_path: "docs/config.md", section_title: "Options", score: 0.65 },
      ],
      chunks_retrieved: 3,
    }),
    sseEvent("status", { status: "generating" }),
    sseEvent("answer_token", { token: "The " }),
    sseEvent("answer_token", { token: "configuration " }),
    sseEvent("answer_token", { token: "file " }),
    sseEvent("answer_token", { token: "is in " }),
    sseEvent("answer_token", { token: "`config.yaml`." }),
    sseEvent("answer", {
      answer: "The configuration file is in `config.yaml`.",
      model: "claude-sonnet-4-20250514",
      session_id: sessionId,
    }),
    sseEvent("status", { status: "verifying" }),
    sseEvent("attribution", {
      attributions: [
        {
          sentence: "The configuration file is in config.yaml.",
          source_index: 0,
          similarity: 0.52,
        },
      ],
    }),
    sseEvent("verified", { passed: false, confidence: 0.45 }),
    sseEvent("done", { status: "complete", elapsed: 1.87 }),
  ].join("");
}

/** Minimal stream — just tokens and done (for multi-turn tests). */
export function quickStream(
  answer: string,
  sessionId = "sess-abc-123",
): string {
  const tokens = answer.split(" ").map((word, i, arr) => {
    const suffix = i < arr.length - 1 ? " " : "";
    return sseEvent("answer_token", { token: word + suffix });
  });
  return [
    sseEvent("status", { status: "classifying", session_id: sessionId }),
    sseEvent("status", { status: "generating" }),
    ...tokens,
    sseEvent("answer", {
      answer,
      model: "claude-sonnet-4-20250514",
      session_id: sessionId,
    }),
    sseEvent("verified", { passed: true, confidence: 0.85 }),
    sseEvent("done", { status: "complete", elapsed: 0.5 }),
  ].join("");
}

/* ------------------------------------------------------------------ */
/*  Indexing SSE streams                                               */
/* ------------------------------------------------------------------ */

/** Full successful indexing stream. */
export function indexingFullStream(): string {
  return [
    sseEvent("status", { state: "scanning", repo_path: "/docs/repo" }),
    sseEvent("progress", { state: "scanning", processed: 0, total_files: 3, total_chunks: 0, percent: 0 }),
    sseEvent("status", { state: "indexing" }),
    sseEvent("progress", { state: "indexing", processed: 1, total_files: 3, total_chunks: 5, percent: 33 }),
    sseEvent("file_done", { file: "/docs/repo/intro.md", file_index: 0, total_files: 3, chunks: 5, sections: 3, skipped: false }),
    sseEvent("progress", { state: "indexing", processed: 2, total_files: 3, total_chunks: 9, percent: 67 }),
    sseEvent("file_done", { file: "/docs/repo/guide.md", file_index: 1, total_files: 3, chunks: 4, sections: 2, skipped: false }),
    sseEvent("progress", { state: "indexing", processed: 3, total_files: 3, total_chunks: 12, percent: 100 }),
    sseEvent("file_done", { file: "/docs/repo/api.md", file_index: 2, total_files: 3, chunks: 3, sections: 2, skipped: false }),
    sseEvent("status", { state: "rebuilding_fts" }),
    sseEvent("status", { state: "swapping" }),
    sseEvent("done", { total_files: 3, total_chunks: 12, elapsed: 4.56 }),
  ].join("");
}

/** Indexing stream that gets cancelled. */
export function indexingCancelledStream(): string {
  return [
    sseEvent("status", { state: "scanning", repo_path: "/docs/repo" }),
    sseEvent("status", { state: "indexing" }),
    sseEvent("progress", { state: "indexing", processed: 1, total_files: 5, total_chunks: 3, percent: 20 }),
    sseEvent("file_done", { file: "/docs/repo/readme.md", file_index: 0, total_files: 5, chunks: 3, sections: 2, skipped: false }),
    sseEvent("cancelled", { message: "Cancelled by user" }),
  ].join("");
}

/** Indexing stream that errors out. */
export function indexingErrorStream(): string {
  return [
    sseEvent("status", { state: "scanning", repo_path: "/docs/repo" }),
    sseEvent("error", { error: "Permission denied: /docs/repo", type: "FileNotFoundError" }),
  ].join("");
}
