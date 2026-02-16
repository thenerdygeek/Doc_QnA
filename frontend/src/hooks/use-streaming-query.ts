import { useCallback, useRef, useState } from "react";
import { streamQuery } from "@/api/sse-client";
import type { SSEEvent, PipelineStatus } from "@/types/sse";
import type { SourceInfo, AttributionInfo } from "@/types/api";

export type StreamPhase = "idle" | "streaming" | "complete" | "error";

export interface StreamingQueryState {
  phase: StreamPhase;
  pipelineStatus: PipelineStatus | null;
  thinkingTokens: string;
  tokens: string;
  answer: string | null;
  intent: string | null;
  intentConfidence: number | null;
  sources: SourceInfo[];
  chunksRetrieved: number;
  attributions: AttributionInfo[];
  verification: { passed: boolean; confidence: number } | null;
  model: string | null;
  sessionId: string | null;
  diagrams: string[];
  elapsed: number | null;
  error: string | null;
}

const INITIAL_STATE: StreamingQueryState = {
  phase: "idle",
  pipelineStatus: null,
  thinkingTokens: "",
  tokens: "",
  answer: null,
  intent: null,
  intentConfidence: null,
  sources: [],
  chunksRetrieved: 0,
  attributions: [],
  verification: null,
  model: null,
  sessionId: null,
  diagrams: [],
  elapsed: null,
  error: null,
};

export function useStreamingQuery() {
  const [state, setState] = useState<StreamingQueryState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const handleEvent = useCallback((event: SSEEvent) => {
    setState((prev) => {
      switch (event.event) {
        case "status":
          return { ...prev, pipelineStatus: event.data.status };
        case "intent":
          return {
            ...prev,
            intent: event.data.intent,
            intentConfidence: event.data.confidence,
          };
        case "sources":
          return {
            ...prev,
            sources: event.data.sources,
            chunksRetrieved: event.data.chunks_retrieved,
          };
        case "thinking_token":
          return { ...prev, thinkingTokens: prev.thinkingTokens + event.data.token };
        case "answer_token":
          return { ...prev, tokens: prev.tokens + event.data.token };
        case "answer":
          return {
            ...prev,
            answer: event.data.answer,
            model: event.data.model,
            sessionId: event.data.session_id,
            diagrams: event.data.diagrams ?? [],
          };
        case "attribution":
          return { ...prev, attributions: event.data.attributions };
        case "verified":
          return {
            ...prev,
            verification: {
              passed: event.data.passed,
              confidence: event.data.confidence,
            },
          };
        case "done":
          return {
            ...prev,
            phase: "complete",
            pipelineStatus: "complete",
            elapsed: event.data.elapsed,
          };
        case "error":
          return { ...prev, phase: "error", error: event.data.error };
        default:
          return prev;
      }
    });
  }, []);

  const submit = useCallback(
    async (question: string, sessionId?: string) => {
      // Abort any running stream
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setState({ ...INITIAL_STATE, phase: "streaming" });

      try {
        await streamQuery({
          question,
          sessionId,
          onEvent: handleEvent,
          signal: controller.signal,
        });
      } catch (err) {
        if (controller.signal.aborted) return;
        setState((prev) => ({
          ...prev,
          phase: "error",
          error: err instanceof Error ? err.message : "Stream failed",
        }));
      }
    },
    [handleEvent],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, phase: "complete" }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL_STATE);
  }, []);

  return { ...state, submit, cancel, reset };
}
