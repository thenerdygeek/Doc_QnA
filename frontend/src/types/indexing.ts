// ── Indexing state machine ───────────────────────────────────────

export type IndexingState =
  | "idle"
  | "scanning"
  | "indexing"
  | "rebuilding_fts"
  | "swapping"
  | "done"
  | "cancelled"
  | "error";

// ── Per-event data interfaces ───────────────────────────────────

export interface IndexingStatusData {
  state: IndexingState;
  repo_path?: string;
}

export interface IndexingProgressData {
  state: string;
  processed: number;
  total_files: number;
  total_chunks: number;
  percent: number;
  message?: string;
}

export interface IndexingFileDoneData {
  file: string;
  file_index: number;
  total_files: number;
  chunks: number;
  sections: number;
  skipped: boolean;
}

export interface IndexingDoneData {
  total_files: number;
  total_chunks: number;
  elapsed: number;
  /** Number of files skipped because they were unchanged (incremental indexing) */
  skipped_unchanged?: number;
}

export interface IndexingCancelledData {
  message: string;
}

export interface IndexingErrorData {
  error: string;
  type: string;
}

// ── Discriminated union ─────────────────────────────────────────

export type IndexingSSEEvent =
  | { event: "status"; data: IndexingStatusData }
  | { event: "progress"; data: IndexingProgressData }
  | { event: "file_done"; data: IndexingFileDoneData }
  | { event: "done"; data: IndexingDoneData }
  | { event: "cancelled"; data: IndexingCancelledData }
  | { event: "error"; data: IndexingErrorData };

// ── Status response (poll fallback) ─────────────────────────────

export interface IndexingStatusResponse {
  state: IndexingState;
  repo_path?: string;
  total_files?: number;
  processed_files?: number;
  total_chunks?: number;
  error?: string | null;
}
