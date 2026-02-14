import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";
import { WelcomeScreen } from "./welcome-screen";
import { api } from "@/api/client";

// ── Mock framer-motion ──────────────────────────────────────────
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => {
      const safe: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(props)) {
        if (k === "className" || k === "role" || k.startsWith("aria-") || k.startsWith("data-")) {
          safe[k] = v;
        }
      }
      return <div {...(safe as React.HTMLAttributes<HTMLDivElement>)}>{children}</div>;
    },
    button: ({
      children,
      ...props
    }: React.PropsWithChildren<Record<string, unknown>>) => {
      const safe: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(props)) {
        if (
          k === "className" ||
          k === "type" ||
          k === "onClick" ||
          k === "onMouseMove" ||
          k === "role" ||
          k.startsWith("aria-") ||
          k.startsWith("data-")
        ) {
          safe[k] = v;
        }
      }
      return (
        <button {...(safe as React.ButtonHTMLAttributes<HTMLButtonElement>)}>
          {children}
        </button>
      );
    },
    span: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
  },
  AnimatePresence: ({ children }: React.PropsWithChildren) => <>{children}</>,
  useReducedMotion: () => true,
}));

// ── Mock API client ─────────────────────────────────────────────
vi.mock("@/api/client", () => ({
  api: {
    stats: vi.fn(),
  },
}));

const mockStats = api.stats as ReturnType<typeof vi.fn>;

const defaultStats = { total_files: 0, total_chunks: 0, db_path: "", embedding_model: "" };

/** Render and flush the pending stats promise so React doesn't warn about act(). */
async function renderWelcome(onSelectQuestion = vi.fn()) {
  let result: ReturnType<typeof render>;
  await act(async () => {
    result = render(<WelcomeScreen onSelectQuestion={onSelectQuestion} />);
  });
  return { ...result!, onSelectQuestion };
}

// ── Tests ───────────────────────────────────────────────────────

describe("WelcomeScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockStats.mockReset();
  });

  it('renders heading "Ask your docs anything"', async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    expect(screen.getByText("Ask your docs anything")).toBeInTheDocument();
  });

  it("renders subtitle", async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    expect(
      screen.getByText(/Ask anything about your documentation/),
    ).toBeInTheDocument();
  });

  it("fetches stats on mount via api.stats()", async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    expect(mockStats).toHaveBeenCalledTimes(1);
  });

  it("shows stats when available with file count + chunk count", async () => {
    mockStats.mockResolvedValue({
      total_files: 42,
      total_chunks: 1234,
      db_path: "/db",
      embedding_model: "text-embedding",
    });
    await renderWelcome();

    expect(screen.getByText(/42 files indexed/)).toBeInTheDocument();
    expect(screen.getByText(/1,234 chunks/)).toBeInTheDocument();
  });

  it("stats fetch failure doesn't crash", async () => {
    mockStats.mockRejectedValue(new Error("Network error"));
    const { container } = await renderWelcome();
    // Should still render the heading without crashing
    expect(screen.getByText("Ask your docs anything")).toBeInTheDocument();
    // No stats should be displayed
    expect(screen.queryByText(/files indexed/)).not.toBeInTheDocument();
  });

  it("unmounted component doesn't setState (mounted guard)", async () => {
    let resolveStats!: (value: unknown) => void;
    mockStats.mockReturnValue(
      new Promise((r) => {
        resolveStats = r;
      }),
    );

    const { unmount } = render(<WelcomeScreen onSelectQuestion={vi.fn()} />);
    unmount();

    // Resolve after unmount - should not cause any error thanks to mounted guard
    await act(async () => {
      resolveStats({
        total_files: 10,
        total_chunks: 100,
        db_path: "",
        embedding_model: "",
      });
    });

    expect(mockStats).toHaveBeenCalled();
  });

  it("renders exactly 4 example buttons", async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(4);
  });

  it("clicking button calls onSelectQuestion with question text", async () => {
    mockStats.mockResolvedValue(defaultStats);
    const onSelect = vi.fn();
    await renderWelcome(onSelect);
    const btn = screen.getByLabelText("Ask: How does the authentication flow work?");
    fireEvent.click(btn);
    expect(onSelect).toHaveBeenCalledWith(
      "How does the authentication flow work?",
    );
  });

  it("each button has icon + label + preview text", async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    // Check all 4 labels exist
    expect(screen.getByText("Explain a concept")).toBeInTheDocument();
    expect(screen.getByText("Code example")).toBeInTheDocument();
    expect(screen.getByText("Compare options")).toBeInTheDocument();
    expect(screen.getByText("Step-by-step")).toBeInTheDocument();
    // Check preview texts exist
    expect(
      screen.getByText("How does the authentication flow work?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Show me an example of the REST API usage"),
    ).toBeInTheDocument();
  });

  it("question preview has line-clamp-2 class", async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    const preview = screen.getByText(
      "How does the authentication flow work?",
    );
    expect(preview.className).toContain("line-clamp-2");
  });

  it('each button has aria-label="Ask: {question}"', async () => {
    mockStats.mockResolvedValue(defaultStats);
    await renderWelcome();
    expect(
      screen.getByLabelText("Ask: How does the authentication flow work?"),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Ask: Show me an example of the REST API usage"),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(
        "Ask: What are the differences between v1 and v2 APIs?",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Ask: How do I set up the development environment?"),
    ).toBeInTheDocument();
  });

  it('decorative orbs have aria-hidden="true"', async () => {
    mockStats.mockResolvedValue(defaultStats);
    const { container } = await renderWelcome();
    const hiddenEls = container.querySelectorAll('[aria-hidden="true"]');
    expect(hiddenEls.length).toBeGreaterThanOrEqual(1);
    // The decorative orb container should contain the animated orbs
    const orbContainer = Array.from(hiddenEls).find((el) =>
      el.querySelector(".orb"),
    );
    expect(orbContainer).toBeInTheDocument();
  });

  it("chunk count uses toLocaleString formatting", async () => {
    mockStats.mockResolvedValue({
      total_files: 5,
      total_chunks: 98765,
      db_path: "",
      embedding_model: "",
    });
    await renderWelcome();

    // toLocaleString should produce "98,765" in en-US
    expect(
      screen.getByText((_, el) => {
        return el?.textContent === "98,765 chunks";
      }),
    ).toBeInTheDocument();
  });
});
