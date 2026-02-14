import { renderHook, act } from "@testing-library/react";
import { useStreamingQuery } from "./use-streaming-query";
import { vi } from "vitest";
import type { SSEEvent } from "@/types/sse";

// Mock the SSE client
vi.mock("@/api/sse-client", () => ({
  streamQuery: vi.fn(),
}));

import { streamQuery } from "@/api/sse-client";

const mockStreamQuery = vi.mocked(streamQuery);

beforeEach(() => {
  mockStreamQuery.mockReset();
});

describe("useStreamingQuery", () => {
  it("starts with idle phase", () => {
    const { result } = renderHook(() => useStreamingQuery());
    expect(result.current.phase).toBe("idle");
    expect(result.current.tokens).toBe("");
    expect(result.current.answer).toBeNull();
    expect(result.current.sources).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it("transitions to streaming phase on submit", async () => {
    mockStreamQuery.mockResolvedValue(undefined);

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test question");
    });

    // After streamQuery resolves, it remains in streaming unless done event fires
    expect(result.current.phase).toBe("streaming");
  });

  it("accumulates answer_token events", async () => {
    let capturedOnEvent: ((event: SSEEvent) => void) | null = null;
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      capturedOnEvent = onEvent;
      // Simulate receiving tokens
      onEvent({ event: "answer_token", data: { token: "Hello " } });
      onEvent({ event: "answer_token", data: { token: "world" } });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(capturedOnEvent).not.toBeNull();
    expect(result.current.tokens).toBe("Hello world");
  });

  it("handles pipeline status events", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "status", data: { status: "retrieving" } });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.pipelineStatus).toBe("retrieving");
  });

  it("handles sources event", async () => {
    const sources = [
      { file_path: "docs/api.md", section_title: "REST API", score: 0.92 },
    ];
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "sources",
        data: { sources, chunks_retrieved: 5 },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.sources).toEqual(sources);
    expect(result.current.chunksRetrieved).toBe(5);
  });

  it("handles done event and transitions to complete", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "done", data: { status: "complete", elapsed: 2.5 } });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.phase).toBe("complete");
    expect(result.current.elapsed).toBe(2.5);
  });

  it("handles error event", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "error",
        data: { error: "No relevant docs found", type: "retrieval" },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.error).toBe("No relevant docs found");
  });

  it("handles verification event", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "verified",
        data: { passed: true, confidence: 0.95 },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.verification).toEqual({
      passed: true,
      confidence: 0.95,
    });
  });

  it("cancel aborts and transitions to complete", async () => {
    mockStreamQuery.mockImplementation(
      () => new Promise(() => {}), // never resolves
    );

    const { result } = renderHook(() => useStreamingQuery());
    act(() => {
      void result.current.submit("test");
    });

    act(() => {
      result.current.cancel();
    });

    expect(result.current.phase).toBe("complete");
  });

  it("reset returns to idle state", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "answer_token", data: { token: "data" } });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });
    expect(result.current.tokens).toBe("data");

    act(() => {
      result.current.reset();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.tokens).toBe("");
  });

  it("handles stream rejection as error", async () => {
    mockStreamQuery.mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.error).toBe("Network error");
  });
});

describe("useStreamingQuery – gap coverage", () => {
  it("submit while already streaming aborts previous", async () => {
    let callCount = 0;
    const signals: AbortSignal[] = [];

    mockStreamQuery.mockImplementation(async ({ signal }) => {
      callCount++;
      signals.push(signal);
      // Hang indefinitely on first call, resolve instantly on second
      if (callCount === 1) {
        return new Promise(() => {});
      }
    });

    const { result } = renderHook(() => useStreamingQuery());

    // Start first stream (will hang)
    act(() => {
      void result.current.submit("first question");
    });

    // Start second stream — should abort the first
    await act(async () => {
      await result.current.submit("second question");
    });

    expect(signals[0].aborted).toBe(true);
    expect(callCount).toBe(2);
  });

  it("session ID extracted from answer event", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "answer",
        data: {
          answer: "The answer",
          model: "claude-3",
          session_id: "sess-abc",
          diagrams: [],
        },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.sessionId).toBe("sess-abc");
  });

  it("diagrams extracted from answer event", async () => {
    const diagrams = ["graph TD; A-->B;", "sequenceDiagram; A->>B: Hello"];
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "answer",
        data: {
          answer: "Here is a diagram",
          model: "claude-3",
          session_id: "sess-1",
          diagrams,
        },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.diagrams).toEqual(diagrams);
  });

  it("intent + confidence from intent event", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "intent",
        data: { intent: "factual_lookup", confidence: 0.87 },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.intent).toBe("factual_lookup");
    expect(result.current.intentConfidence).toBe(0.87);
  });

  it("attribution event populates attributions array", async () => {
    const attributions = [
      { sentence: "Sentence one.", source_index: 0, similarity: 0.95 },
      { sentence: "Sentence two.", source_index: 1, similarity: 0.82 },
    ];
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "attribution", data: { attributions } });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.attributions).toEqual(attributions);
  });

  it("multiple answer_tokens concatenate in order", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "answer_token", data: { token: "A" } });
      onEvent({ event: "answer_token", data: { token: "B" } });
      onEvent({ event: "answer_token", data: { token: "C" } });
      onEvent({ event: "answer_token", data: { token: "D" } });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.tokens).toBe("ABCD");
  });

  it("error event sets phase='error' + message", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "error",
        data: { error: "Rate limit exceeded", type: "server" },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.error).toBe("Rate limit exceeded");
  });

  it("cancel while idle is no-op", () => {
    const { result } = renderHook(() => useStreamingQuery());
    expect(result.current.phase).toBe("idle");

    act(() => {
      result.current.cancel();
    });

    // Cancel sets phase to "complete" regardless
    expect(result.current.phase).toBe("complete");
  });

  it("reset clears ALL state", async () => {
    mockStreamQuery.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "answer_token", data: { token: "partial" } });
      onEvent({
        event: "intent",
        data: { intent: "factual", confidence: 0.9 },
      });
      onEvent({
        event: "sources",
        data: {
          sources: [
            { file_path: "a.md", section_title: "A", score: 0.8 },
          ],
          chunks_retrieved: 3,
        },
      });
      onEvent({
        event: "answer",
        data: {
          answer: "Full answer",
          model: "claude-3",
          session_id: "sess-x",
          diagrams: ["graph LR; A-->B"],
        },
      });
      onEvent({
        event: "attribution",
        data: {
          attributions: [
            { sentence: "s", source_index: 0, similarity: 0.9 },
          ],
        },
      });
      onEvent({
        event: "verified",
        data: { passed: true, confidence: 0.99 },
      });
      onEvent({
        event: "done",
        data: { status: "complete", elapsed: 1.5 },
      });
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    // Verify state was populated
    expect(result.current.phase).toBe("complete");
    expect(result.current.tokens).not.toBe("");
    expect(result.current.sessionId).toBe("sess-x");

    act(() => {
      result.current.reset();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.tokens).toBe("");
    expect(result.current.answer).toBeNull();
    expect(result.current.intent).toBeNull();
    expect(result.current.intentConfidence).toBeNull();
    expect(result.current.sources).toEqual([]);
    expect(result.current.chunksRetrieved).toBe(0);
    expect(result.current.attributions).toEqual([]);
    expect(result.current.verification).toBeNull();
    expect(result.current.model).toBeNull();
    expect(result.current.sessionId).toBeNull();
    expect(result.current.diagrams).toEqual([]);
    expect(result.current.elapsed).toBeNull();
    expect(result.current.error).toBeNull();
    expect(result.current.pipelineStatus).toBeNull();
  });

  it("stream error (non-abort) sets error phase", async () => {
    mockStreamQuery.mockImplementation(async () => {
      throw new Error("Connection reset");
    });

    const { result } = renderHook(() => useStreamingQuery());
    await act(async () => {
      await result.current.submit("test");
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.error).toBe("Connection reset");
  });
});
