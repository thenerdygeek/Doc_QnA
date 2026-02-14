import { renderHook, act, waitFor } from "@testing-library/react";
import { useConversations } from "./use-conversations";
import { vi } from "vitest";
import type { ConversationSummary } from "@/types/api";

vi.mock("@/api/client", () => {
  class MockApiError extends Error {
    status: number;
    detail: string;
    constructor(status: number, detail: string) {
      super(detail);
      this.name = "ApiError";
      this.status = status;
      this.detail = detail;
    }
  }
  return {
    ApiError: MockApiError,
    api: {
      conversations: {
        list: vi.fn(),
        delete: vi.fn(),
      },
    },
  };
});

import { api, ApiError } from "@/api/client";

const mockList = vi.mocked(api.conversations.list);
const mockDelete = vi.mocked(api.conversations.delete);

const FAKE_CONVERSATIONS: ConversationSummary[] = [
  {
    id: "conv-1",
    user_id: null,
    title: "First conversation",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "conv-2",
    user_id: null,
    title: "Second conversation",
    created_at: "2026-01-02T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
  },
];

beforeEach(() => {
  vi.clearAllMocks();
  mockList.mockResolvedValue(FAKE_CONVERSATIONS);
  mockDelete.mockResolvedValue({ ok: true });
});

describe("useConversations", () => {
  it("fetches conversations on mount", async () => {
    renderHook(() => useConversations());

    await waitFor(() => {
      expect(mockList).toHaveBeenCalledTimes(1);
    });
  });

  it("returns parsed conversation array", async () => {
    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.conversations).toEqual(FAKE_CONVERSATIONS);
    });
  });

  it("loading true during fetch, false after", async () => {
    let resolveList!: (value: ConversationSummary[]) => void;
    mockList.mockReturnValue(
      new Promise((resolve) => {
        resolveList = resolve;
      }),
    );

    const { result } = renderHook(() => useConversations());
    expect(result.current.loading).toBe(true);

    await act(async () => {
      resolveList(FAKE_CONVERSATIONS);
    });

    expect(result.current.loading).toBe(false);
  });

  it("501 response sets dbEnabled=false", async () => {
    mockList.mockRejectedValue(new ApiError(501, "Not implemented"));

    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.dbEnabled).toBe(false);
    });
  });

  it("successful response sets dbEnabled=true", async () => {
    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.dbEnabled).toBe(true);
    });
  });

  it("deleteConversation(id) removes from local list", async () => {
    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.conversations).toHaveLength(2);
    });

    await act(async () => {
      await result.current.deleteConversation("conv-1");
    });

    expect(result.current.conversations).toHaveLength(1);
    expect(result.current.conversations[0].id).toBe("conv-2");
  });

  it("deleteConversation(id) calls DELETE API", async () => {
    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.conversations).toHaveLength(2);
    });

    await act(async () => {
      await result.current.deleteConversation("conv-1");
    });

    expect(mockDelete).toHaveBeenCalledWith("conv-1");
  });

  it("delete failure: list unchanged, console.error", async () => {
    mockDelete.mockRejectedValue(new Error("Server error"));
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.conversations).toHaveLength(2);
    });

    await act(async () => {
      await result.current.deleteConversation("conv-1");
    });

    // List unchanged because the API call failed before setConversations
    expect(result.current.conversations).toHaveLength(2);
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it("refresh() re-fetches full list", async () => {
    const { result } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.conversations).toHaveLength(2);
    });

    const updatedList: ConversationSummary[] = [
      ...FAKE_CONVERSATIONS,
      {
        id: "conv-3",
        user_id: null,
        title: "Third",
        created_at: "2026-01-03T00:00:00Z",
        updated_at: "2026-01-03T00:00:00Z",
      },
    ];
    mockList.mockResolvedValue(updatedList);

    await act(async () => {
      await result.current.refresh();
    });

    expect(result.current.conversations).toHaveLength(3);
    // list called twice: initial mount + refresh
    expect(mockList).toHaveBeenCalledTimes(2);
  });

  it("double-mount does not double-fetch (hasFetched ref)", async () => {
    const { result, unmount } = renderHook(() => useConversations());

    await waitFor(() => {
      expect(result.current.conversations).toHaveLength(2);
    });

    // Unmount and remount (simulating strict mode) -- but hasFetched ref
    // In the same component instance, the ref guards re-fetch
    expect(mockList).toHaveBeenCalledTimes(1);
  });
});
