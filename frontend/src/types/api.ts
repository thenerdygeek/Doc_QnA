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

export interface EmbeddingModelInfo {
  configured: string;
  resolved: string;
  ram_gb: number;
  ram_sufficient: boolean;
}

export interface HealthResponse {
  status: string;
  components?: {
    index?: ComponentHealth;
  };
  embedding_model?: EmbeddingModelInfo;
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

// ── Feedback types ──────────────────────────────────────────────

export interface FeedbackRequest {
  query_id: string;
  rating: number; // -1 (bad) | 1 (good)
  comment?: string;
}

// ── Config types ────────────────────────────────────────────────

/** Full config as returned by GET /api/config (secrets redacted). */
export type ConfigData = Record<string, Record<string, unknown>>;

export interface ConfigUpdateResponse {
  saved: boolean;
  restart_required: boolean;
  restart_sections: string[];
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
