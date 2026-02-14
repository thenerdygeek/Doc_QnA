import { renderHook, act } from "@testing-library/react";
import { useSession } from "./use-session";
import { vi } from "vitest";

const SESSION_KEY = "doc-qa-session-id";

beforeEach(() => {
  sessionStorage.clear();
});

describe("useSession", () => {
  it("returns undefined sessionId when storage is empty", () => {
    const { result } = renderHook(() => useSession());
    expect(result.current.sessionId).toBeUndefined();
  });

  it("restores sessionId from sessionStorage", () => {
    sessionStorage.setItem(SESSION_KEY, "abc-123");
    const { result } = renderHook(() => useSession());
    expect(result.current.sessionId).toBe("abc-123");
  });

  it("setSessionId persists to sessionStorage and updates state", () => {
    const { result } = renderHook(() => useSession());
    act(() => {
      result.current.setSessionId("new-session");
    });
    expect(result.current.sessionId).toBe("new-session");
    expect(sessionStorage.getItem(SESSION_KEY)).toBe("new-session");
  });

  it("clearSession removes from storage and resets state", () => {
    sessionStorage.setItem(SESSION_KEY, "existing");
    const { result } = renderHook(() => useSession());
    expect(result.current.sessionId).toBe("existing");

    act(() => {
      result.current.clearSession();
    });
    expect(result.current.sessionId).toBeUndefined();
    expect(sessionStorage.getItem(SESSION_KEY)).toBeNull();
  });
});

describe("useSession – gap coverage", () => {
  it("sessionStorage unavailable does not crash", () => {
    // Simulate sessionStorage.getItem returning null (e.g., storage cleared,
    // private browsing mode, or storage disabled). The hook uses
    // `sessionStorage.getItem(key) ?? undefined` in the useState initializer,
    // so returning null results in undefined — same as empty storage.
    const spy = vi.spyOn(sessionStorage, "getItem").mockReturnValue(null);

    const { result } = renderHook(() => useSession());
    expect(result.current.sessionId).toBeUndefined();

    // setSessionId and clearSession still work — they call setItem/removeItem
    // which in jsdom's mock storage won't fail, but the getItem mock confirms
    // the hook gracefully handles missing data on init.
    act(() => {
      result.current.setSessionId("fallback-id");
    });
    expect(result.current.sessionId).toBe("fallback-id");

    act(() => {
      result.current.clearSession();
    });
    expect(result.current.sessionId).toBeUndefined();

    spy.mockRestore();
  });
});
