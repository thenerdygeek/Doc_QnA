import { render, screen, act, fireEvent } from "@testing-library/react";
import { vi } from "vitest";
import { ConnectionStatus } from "./connection-status";

// ── Mock api.health ─────────────────────────────────────────────────
const mockHealth = vi.fn();
vi.mock("@/api/client", () => ({
  api: {
    health: (...args: unknown[]) => mockHealth(...args),
  },
}));

describe("ConnectionStatus", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("returns null while status is 'checking' (initial state)", () => {
    // health never resolves
    mockHealth.mockReturnValue(new Promise(() => {}));
    const { container } = render(<ConnectionStatus />);
    expect(container.innerHTML).toBe("");
  });

  it("shows 'Connected' and green dot when health succeeds with chunks", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 42 } },
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Connected")).toBeInTheDocument();
  });

  it("shows 'Degraded' and amber dot when index is empty", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 0 } },
    });
    render(<ConnectionStatus />);
    expect(await screen.findByText("Degraded")).toBeInTheDocument();
  });

  it("shows animated ping on connected (animate-ping class)", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 10 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Connected");
    const container = screen.getByRole("status");
    const ping = container.querySelector(".animate-ping");
    expect(ping).toBeInTheDocument();
  });

  it("shows animated ping on degraded too", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 0 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Degraded");
    const container = screen.getByRole("status");
    const ping = container.querySelector(".animate-ping");
    expect(ping).toBeInTheDocument();
  });

  it("shows 'Offline' and red dot when health fails", async () => {
    mockHealth.mockRejectedValue(new Error("Network error"));
    render(<ConnectionStatus />);
    expect(await screen.findByText("Offline")).toBeInTheDocument();
  });

  it("has no animated ping when disconnected", async () => {
    mockHealth.mockRejectedValue(new Error("fail"));
    render(<ConnectionStatus />);
    await screen.findByText("Offline");
    const container = screen.getByRole("status");
    const ping = container.querySelector(".animate-ping");
    expect(ping).not.toBeInTheDocument();
  });

  it("has role='status' on container", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 5 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Connected");
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("has aria-label='Backend connected' when connected", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 5 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Connected");
    expect(screen.getByRole("status")).toHaveAttribute(
      "aria-label",
      "Backend connected",
    );
  });

  it("has aria-label='Backend degraded' when degraded", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 0 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Degraded");
    expect(screen.getByRole("status")).toHaveAttribute(
      "aria-label",
      "Backend degraded",
    );
  });

  it("has aria-label='Backend disconnected' when disconnected", async () => {
    mockHealth.mockRejectedValue(new Error("fail"));
    render(<ConnectionStatus />);
    await screen.findByText("Offline");
    expect(screen.getByRole("status")).toHaveAttribute(
      "aria-label",
      "Backend disconnected",
    );
  });

  it("text is hidden on mobile (hidden sm:inline class)", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 5 } },
    });
    render(<ConnectionStatus />);
    const text = await screen.findByText("Connected");
    expect(text.className).toContain("hidden");
    expect(text.className).toContain("sm:inline");
  });

  it("polls health endpoint on mount", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 5 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Connected");
    expect(mockHealth).toHaveBeenCalledTimes(1);
  });

  it("re-polls every 30 seconds", async () => {
    vi.useFakeTimers();
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 5 } },
    });

    render(<ConnectionStatus />);

    // Flush initial check() call (microtask)
    await act(async () => {});
    const initialCalls = mockHealth.mock.calls.length;
    expect(initialCalls).toBeGreaterThanOrEqual(1);

    // Advance 30 seconds — triggers interval
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    expect(mockHealth.mock.calls.length).toBe(initialCalls + 1);

    // Advance another 30 seconds
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    expect(mockHealth.mock.calls.length).toBe(initialCalls + 2);

    vi.useRealTimers();
  });

  it("cleanup cancels polling interval on unmount", async () => {
    vi.useFakeTimers();
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 5 } },
    });

    const { unmount } = render(<ConnectionStatus />);

    // Flush initial check() call
    await act(async () => {});
    const callsBeforeUnmount = mockHealth.mock.calls.length;

    unmount();

    // Advance time — should NOT call health again
    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    expect(mockHealth.mock.calls.length).toBe(callsBeforeUnmount);

    vi.useRealTimers();
  });

  // ── Tooltip tests ─────────────────────────────────────────────

  it("shows tooltip with chunk count on hover when connected", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 1234 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Connected");

    const container = screen.getByRole("status");
    fireEvent.mouseEnter(container);

    const tooltip = screen.getByRole("tooltip");
    expect(tooltip).toBeInTheDocument();
    expect(tooltip.textContent).toContain("1,234 chunks");
  });

  it("shows tooltip with 'Empty' when degraded", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 0 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Degraded");

    const container = screen.getByRole("status");
    fireEvent.mouseEnter(container);

    const tooltip = screen.getByRole("tooltip");
    expect(tooltip.textContent).toContain("Empty");
  });

  it("shows tooltip with 'Offline' when disconnected", async () => {
    mockHealth.mockRejectedValue(new Error("fail"));
    render(<ConnectionStatus />);
    await screen.findByText("Offline");

    const container = screen.getByRole("status");
    fireEvent.mouseEnter(container);

    const tooltip = screen.getByRole("tooltip");
    expect(tooltip.textContent).toContain("Offline");
  });

  it("hides tooltip on mouse leave", async () => {
    mockHealth.mockResolvedValue({
      status: "ok",
      components: { index: { ok: true, chunks: 10 } },
    });
    render(<ConnectionStatus />);
    await screen.findByText("Connected");

    const container = screen.getByRole("status");
    fireEvent.mouseEnter(container);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    fireEvent.mouseLeave(container);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  // ── Backward compatibility ────────────────────────────────────

  it("handles legacy health response without components", async () => {
    // Old format: just { status: "ok" }
    mockHealth.mockResolvedValue({ status: "ok" });
    render(<ConnectionStatus />);
    // Without components, index defaults to ok=false, chunks=0 → degraded
    expect(await screen.findByText("Degraded")).toBeInTheDocument();
  });
});
