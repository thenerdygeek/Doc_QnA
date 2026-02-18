import type { SourceInfo, AttributionInfo } from "./api";

// ── Pipeline phases ──────────────────────────────────────────────

export type PipelineStatus =
  | "classifying"
  | "reformulating"
  | "retrieving"
  | "grading"
  | "generating"
  | "reasoning"
  | "verifying"
  | "complete";

// ── Per-event data interfaces ────────────────────────────────────

export interface SSEStatusData {
  status: PipelineStatus;
  session_id?: string;
}

export interface SSEIntentData {
  intent: string;
  confidence: number;
}

export interface SSESourcesData {
  sources: SourceInfo[];
  chunks_retrieved: number;
}

export interface SSEAnswerTokenData {
  token: string;
}

export interface SSEThinkingTokenData {
  token: string;
}

export interface SSEAnswerData {
  answer: string;
  model: string;
  session_id: string;
  diagrams?: string[];
}

export interface SSEAttributionData {
  attributions: AttributionInfo[];
}

export interface SSEVerifiedData {
  passed: boolean;
  confidence: number;
}

export interface SSEDoneData {
  status: "complete";
  elapsed: number;
  query_id?: string;
}

export interface SSEErrorData {
  error: string;
  type: string;
}

export interface SSERewriteData {
  original: string;
  rewritten: string;
}

export interface CitationInfo {
  number: number;
  chunk_id: string;
  file_path: string;
  section_title: string;
  score: number;
}

export interface SSECitationsData {
  citations: CitationInfo[];
}

// ── Discriminated union ──────────────────────────────────────────

export type SSEEvent =
  | { event: "status"; data: SSEStatusData }
  | { event: "intent"; data: SSEIntentData }
  | { event: "rewrite"; data: SSERewriteData }
  | { event: "sources"; data: SSESourcesData }
  | { event: "thinking_token"; data: SSEThinkingTokenData }
  | { event: "answer_token"; data: SSEAnswerTokenData }
  | { event: "answer"; data: SSEAnswerData }
  | { event: "attribution"; data: SSEAttributionData }
  | { event: "citations"; data: SSECitationsData }
  | { event: "verified"; data: SSEVerifiedData }
  | { event: "done"; data: SSEDoneData }
  | { event: "error"; data: SSEErrorData };
