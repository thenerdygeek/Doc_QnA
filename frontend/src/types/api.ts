// ── Request types ────────────────────────────────────────────────

export interface QueryRequest {
  question: string;
  session_id?: string;
}

export interface RetrievalRequest {
  question: string;
  top_k?: number;
}

// ── Shared sub-types ─────────────────────────────────────────────

export interface SourceInfo {
  file_path: string;
  section_title: string;
  score: number;
}

export interface AttributionInfo {
  sentence: string;
  source_index: number;
  similarity: number;
}

export interface DetectedFormats {
  has_mermaid: boolean;
  has_code_blocks: boolean;
  has_table: boolean;
  has_numbered_list: boolean;
  mermaid_blocks: string[];
  code_blocks: [string, string][];
  code_languages: string[];
}

// ── Response types ───────────────────────────────────────────────

export interface ComponentHealth {
  ok: boolean;
  chunks?: number;
}

export interface HealthResponse {
  status: string;
  components?: {
    index?: ComponentHealth;
  };
}

export interface LLMModel {
  id: string;
  displayName: string;
  provider?: string;
  thinking?: boolean;
  family?: string;
  size?: string;
}

export interface CodyTestResponse {
  ok: boolean;
  error?: string;
  user?: { username: string; email: string; displayName: string };
  models?: LLMModel[];
}

export interface OllamaTestResponse {
  ok: boolean;
  error?: string;
  models?: LLMModel[];
}

export interface StatsResponse {
  total_chunks: number;
  total_files: number;
  db_path: string;
  embedding_model: string;
}

export interface QueryResponse {
  answer: string;
  sources: SourceInfo[];
  chunks_retrieved: number;
  model: string;
  session_id: string;
  error: string | null;
  attributions: AttributionInfo[] | null;
  intent: string | null;
  confidence: number;
  is_abstained: boolean;
  diagrams: string[] | null;
  detected_formats: DetectedFormats | null;
}

export interface RetrievalChunkResponse {
  text: string;
  score: number;
  file_path: string;
  section_title: string;
  chunk_id: string;
}

export interface RetrievalResponse {
  chunks: RetrievalChunkResponse[];
}

// ── Config types ────────────────────────────────────────────────

/** Full config as returned by GET /api/config (secrets redacted). */
export type ConfigData = Record<string, Record<string, unknown>>;

export interface ConfigUpdateResponse {
  saved: boolean;
  restart_required: boolean;
  restart_sections: string[];
}

export interface DbTestResponse {
  ok: boolean;
  error?: string;
}

export interface DbMigrateResponse {
  ok: boolean;
  error?: string;
  revision?: string;
}

// ── Conversation persistence types ──────────────────────────────

export interface ConversationSummary {
  id: string;
  user_id: string | null;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface PersistentMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: PersistentMessage[];
}
