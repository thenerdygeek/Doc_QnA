import { useCallback, useRef, useState } from "react";
import { streamIndex } from "@/api/sse-client";
import { api } from "@/api/client";
import type { IndexingSSEEvent, IndexingFileDoneData } from "@/types/indexing";

export type IndexingPhase = "idle" | "running" | "done" | "cancelled" | "error";

export interface IndexingState {
  phase: IndexingPhase;
  /** Backend state (scanning, indexing, rebuilding_fts, swapping, etc.) */
  state: string;
  repoPath: string;
  totalFiles: number;
  processedFiles: number;
  totalChunks: number;
  percent: number;
  /** Most recent file completions (newest first, max 10) */
  recentFiles: IndexingFileDoneData[];
  elapsed: number | null;
  error: string | null;
}

const INITIAL_STATE: IndexingState = {
  phase: "idle",
  state: "idle",
  repoPath: "",
  totalFiles: 0,
  processedFiles: 0,
  totalChunks: 0,
  percent: 0,
  recentFiles: [],
  elapsed: null,
  error: null,
};

const MAX_RECENT_FILES = 10;

export interface UseIndexingReturn extends IndexingState {
  start: (repoPath: string) => void;
  cancel: () => void;
  reconnect: () => void;
  reset: () => void;
}

export function useIndexing(): UseIndexingReturn {
  const [state, setState] = useState<IndexingState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const handleEvent = useCallback((event: IndexingSSEEvent) => {
    setState((prev) => {
      switch (event.event) {
        case "status":
          return {
            ...prev,
            phase: "running",
            state: event.data.state,
            repoPath: event.data.repo_path ?? prev.repoPath,
          };

        case "progress":
          return {
            ...prev,
            phase: "running",
            state: event.data.state,
            processedFiles: event.data.processed,
            totalFiles: event.data.total_files,
            totalChunks: event.data.total_chunks,
            percent: event.data.percent,
          };

        case "file_done":
          return {
            ...prev,
            recentFiles: [event.data, ...prev.recentFiles].slice(
              0,
              MAX_RECENT_FILES,
            ),
          };

        case "done":
          return {
            ...prev,
            phase: "done",
            state: "done",
            totalFiles: event.data.total_files,
            totalChunks: event.data.total_chunks,
            elapsed: event.data.elapsed,
          };

        case "cancelled":
          return {
            ...prev,
            phase: "cancelled",
            state: "cancelled",
          };

        case "error":
          return {
            ...prev,
            phase: "error",
            state: "error",
            error: event.data.error,
          };

        default:
          return prev;
      }
    });
  }, []);

  const start = useCallback(
    (repoPath: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({ ...INITIAL_STATE, phase: "running", state: "scanning", repoPath });

      streamIndex({
        action: "start",
        repoPath,
        onEvent: handleEvent,
        signal: controller.signal,
      }).catch((err) => {
        if (controller.signal.aborted) return;

        // 409 = indexing already running (e.g., after laptop sleep broke the SSE).
        // Automatically reconnect to the existing job instead of showing an error.
        const is409 =
          err instanceof Error && (err.message.includes("409") || err.message.includes("already running"));
        if (is409) {
          streamIndex({
            onEvent: handleEvent,
            signal: controller.signal,
          }).catch((reconnectErr) => {
            if (controller.signal.aborted) return;
            setState((prev) => ({
              ...prev,
              phase: "error",
              state: "error",
              error: "Indexing is in progress but could not reconnect. Restart the server to reset.",
            }));
          });
          return;
        }

        setState((prev) => ({
          ...prev,
          phase: "error",
          state: "error",
          error: err instanceof Error ? err.message : "Stream failed",
        }));
      });
    },
    [handleEvent],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    api.indexing.cancel().catch(() => {
      // Best-effort — SSE abort already sent
    });
  }, []);

  const reconnect = useCallback(() => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Reset to running so UI shows progress
    setState((prev) => ({
      ...prev,
      phase: prev.phase === "idle" ? "idle" : "running",
    }));

    streamIndex({
      onEvent: handleEvent,
      signal: controller.signal,
    }).catch((err) => {
      if (controller.signal.aborted) return;
      // 204 means no active job — stay idle
      if (err?.message?.includes("204")) return;
      // Don't treat connection errors during reconnect as failures
    });
  }, [handleEvent]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL_STATE);
  }, []);

  return { ...state, start, cancel, reconnect, reset };
}
