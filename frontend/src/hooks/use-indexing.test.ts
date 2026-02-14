import { renderHook, act } from "@testing-library/react";
import { useIndexing } from "./use-indexing";
import { vi } from "vitest";
import type { IndexingSSEEvent } from "@/types/indexing";

// Mock both the SSE client and API client
vi.mock("@/api/sse-client", () => ({
  streamQuery: vi.fn(),
  streamIndex: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  api: {
    indexing: {
      cancel: vi.fn(),
      status: vi.fn(),
    },
  },
}));

import { streamIndex } from "@/api/sse-client";
import { api } from "@/api/client";

const mockStreamIndex = vi.mocked(streamIndex);
const mockCancel = vi.mocked(api.indexing.cancel);

beforeEach(() => {
  mockStreamIndex.mockReset();
  mockCancel.mockReset();
  mockCancel.mockResolvedValue({ ok: true });
});

describe("useIndexing", () => {
  it("starts with idle phase", () => {
    const { result } = renderHook(() => useIndexing());
    expect(result.current.phase).toBe("idle");
    expect(result.current.state).toBe("idle");
    expect(result.current.repoPath).toBe("");
    expect(result.current.totalFiles).toBe(0);
    expect(result.current.processedFiles).toBe(0);
    expect(result.current.totalChunks).toBe(0);
    expect(result.current.percent).toBe(0);
    expect(result.current.recentFiles).toEqual([]);
    expect(result.current.elapsed).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("start sets running phase and calls streamIndex", () => {
    mockStreamIndex.mockImplementation(async () => {});

    const { result } = renderHook(() => useIndexing());
    act(() => {
      result.current.start("/home/user/repo");
    });

    expect(result.current.phase).toBe("running");
    expect(result.current.state).toBe("scanning");
    expect(result.current.repoPath).toBe("/home/user/repo");
    expect(mockStreamIndex).toHaveBeenCalledTimes(1);
    expect(mockStreamIndex).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "start",
        repoPath: "/home/user/repo",
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("status event updates state and repoPath", async () => {
    let capturedOnEvent: ((event: IndexingSSEEvent) => void) | null = null;
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      capturedOnEvent = onEvent;
      onEvent({
        event: "status",
        data: { state: "indexing", repo_path: "/updated/path" },
      });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/initial/path");
    });

    expect(capturedOnEvent).not.toBeNull();
    expect(result.current.phase).toBe("running");
    expect(result.current.state).toBe("indexing");
    expect(result.current.repoPath).toBe("/updated/path");
  });

  it("progress event updates counts", async () => {
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "progress",
        data: {
          state: "indexing",
          processed: 25,
          total_files: 100,
          total_chunks: 500,
          percent: 25.0,
        },
      });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    expect(result.current.phase).toBe("running");
    expect(result.current.state).toBe("indexing");
    expect(result.current.processedFiles).toBe(25);
    expect(result.current.totalFiles).toBe(100);
    expect(result.current.totalChunks).toBe(500);
    expect(result.current.percent).toBe(25.0);
  });

  it("file_done event adds to recentFiles", async () => {
    const fileData = {
      file: "src/main.ts",
      file_index: 1,
      total_files: 50,
      chunks: 3,
      sections: 2,
      skipped: false,
    };

    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      onEvent({ event: "file_done", data: fileData });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    expect(result.current.recentFiles).toHaveLength(1);
    expect(result.current.recentFiles[0]).toEqual(fileData);
  });

  it("recentFiles caps at 10 entries", async () => {
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      for (let i = 0; i < 12; i++) {
        onEvent({
          event: "file_done",
          data: {
            file: `file-${i}.ts`,
            file_index: i,
            total_files: 50,
            chunks: 1,
            sections: 1,
            skipped: false,
          },
        });
      }
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    expect(result.current.recentFiles).toHaveLength(10);
    // Newest first: file-11 should be at index 0
    expect(result.current.recentFiles[0].file).toBe("file-11.ts");
    // Oldest kept: file-2 should be at index 9
    expect(result.current.recentFiles[9].file).toBe("file-2.ts");
  });

  it("done event transitions to done phase", async () => {
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "done",
        data: { total_files: 80, total_chunks: 1200, elapsed: 12.5 },
      });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    expect(result.current.phase).toBe("done");
    expect(result.current.state).toBe("done");
    expect(result.current.totalFiles).toBe(80);
    expect(result.current.totalChunks).toBe(1200);
    expect(result.current.elapsed).toBe(12.5);
  });

  it("cancelled event transitions to cancelled phase", async () => {
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "cancelled",
        data: { message: "User cancelled" },
      });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    expect(result.current.phase).toBe("cancelled");
    expect(result.current.state).toBe("cancelled");
  });

  it("error event sets phase and error message", async () => {
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "error",
        data: { error: "Permission denied", type: "filesystem" },
      });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.state).toBe("error");
    expect(result.current.error).toBe("Permission denied");
  });

  it("cancel aborts and calls api.indexing.cancel", () => {
    const signals: AbortSignal[] = [];
    mockStreamIndex.mockImplementation(async ({ signal }) => {
      signals.push(signal);
      return new Promise(() => {}); // never resolves
    });

    const { result } = renderHook(() => useIndexing());
    act(() => {
      result.current.start("/repo");
    });

    act(() => {
      result.current.cancel();
    });

    expect(signals[0].aborted).toBe(true);
    expect(mockCancel).toHaveBeenCalledTimes(1);
  });

  it("reset returns to initial state", async () => {
    mockStreamIndex.mockImplementation(async ({ onEvent }) => {
      onEvent({
        event: "progress",
        data: {
          state: "indexing",
          processed: 10,
          total_files: 50,
          total_chunks: 200,
          percent: 20,
        },
      });
      onEvent({
        event: "file_done",
        data: {
          file: "a.ts",
          file_index: 0,
          total_files: 50,
          chunks: 2,
          sections: 1,
          skipped: false,
        },
      });
    });

    const { result } = renderHook(() => useIndexing());
    await act(async () => {
      result.current.start("/repo");
    });

    // Verify state was populated
    expect(result.current.phase).toBe("running");
    expect(result.current.processedFiles).toBe(10);
    expect(result.current.recentFiles).toHaveLength(1);

    act(() => {
      result.current.reset();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.state).toBe("idle");
    expect(result.current.repoPath).toBe("");
    expect(result.current.totalFiles).toBe(0);
    expect(result.current.processedFiles).toBe(0);
    expect(result.current.totalChunks).toBe(0);
    expect(result.current.percent).toBe(0);
    expect(result.current.recentFiles).toEqual([]);
    expect(result.current.elapsed).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("start while running aborts previous stream", () => {
    let callCount = 0;
    const signals: AbortSignal[] = [];

    mockStreamIndex.mockImplementation(async ({ signal }) => {
      callCount++;
      signals.push(signal);
      // Hang indefinitely on first call, resolve instantly on second
      if (callCount === 1) {
        return new Promise(() => {});
      }
    });

    const { result } = renderHook(() => useIndexing());

    // Start first stream (will hang)
    act(() => {
      result.current.start("/repo/first");
    });

    // Start second stream — should abort the first
    act(() => {
      result.current.start("/repo/second");
    });

    expect(signals[0].aborted).toBe(true);
    expect(callCount).toBe(2);
    expect(result.current.repoPath).toBe("/repo/second");
  });

  it("reconnect calls streamIndex without action", () => {
    mockStreamIndex.mockImplementation(async () => {});

    const { result } = renderHook(() => useIndexing());

    // First start to move out of idle
    act(() => {
      result.current.start("/repo");
    });

    mockStreamIndex.mockReset();
    mockStreamIndex.mockImplementation(async () => {});

    act(() => {
      result.current.reconnect();
    });

    expect(mockStreamIndex).toHaveBeenCalledTimes(1);
    const callArgs = mockStreamIndex.mock.calls[0][0];
    expect(callArgs.action).toBeUndefined();
    expect(callArgs.signal).toBeInstanceOf(AbortSignal);
    // Phase should remain running (was running before reconnect)
    expect(result.current.phase).toBe("running");
  });

  it("reconnect from idle keeps idle phase", () => {
    mockStreamIndex.mockImplementation(async () => {});

    const { result } = renderHook(() => useIndexing());
    expect(result.current.phase).toBe("idle");

    act(() => {
      result.current.reconnect();
    });

    expect(result.current.phase).toBe("idle");
  });

  it("stream rejection sets error phase", async () => {
    mockStreamIndex.mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useIndexing());
    // start fires the stream asynchronously via .catch; we need to flush the microtask
    await act(async () => {
      result.current.start("/repo");
      // Allow the rejected promise + .catch setState to flush
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(result.current.phase).toBe("error");
    expect(result.current.state).toBe("error");
    expect(result.current.error).toBe("Network error");
  });
});
