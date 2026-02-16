import type { SourceInfo, AttributionInfo } from "./api";

// ── Pipeline phases ──────────────────────────────────────────────

export type PipelineStatus =
  | "classifying"
  | "retrieving"
  | "grading"
  | "generating"
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
}

export interface SSEErrorData {
  error: string;
  type: string;
}

// ── Discriminated union ──────────────────────────────────────────

export type SSEEvent =
  | { event: "status"; data: SSEStatusData }
  | { event: "intent"; data: SSEIntentData }
  | { event: "sources"; data: SSESourcesData }
  | { event: "thinking_token"; data: SSEThinkingTokenData }
  | { event: "answer_token"; data: SSEAnswerTokenData }
  | { event: "answer"; data: SSEAnswerData }
  | { event: "attribution"; data: SSEAttributionData }
  | { event: "verified"; data: SSEVerifiedData }
  | { event: "done"; data: SSEDoneData }
  | { event: "error"; data: SSEErrorData };
