import { render, screen, within } from "@testing-library/react";
import { MessageList, type Message } from "./message-list";
import type { StreamingQueryState } from "@/hooks/use-streaming-query";

// ── Mock framer-motion ──────────────────────────────────────────
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => {
      const safe: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(props)) {
        if (
          typeof v !== "object" &&
          typeof v !== "function" &&
          k !== "initial" &&
          k !== "animate" &&
          k !== "transition" &&
          k !== "whileHover"
        ) {
          safe[k] = v;
        }
        if (k === "className" || k === "aria-label" || k === "role") {
          safe[k] = v;
        }
      }
      return <div {...(safe as React.HTMLAttributes<HTMLDivElement>)}>{children}</div>;
    },
  },
  AnimatePresence: ({ children }: React.PropsWithChildren) => <>{children}</>,
  useReducedMotion: () => false,
}));

// ── Mock ScrollArea (just renders children) ─────────────────────
vi.mock("@/components/ui/scroll-area", () => ({
  ScrollArea: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => (
    <div data-testid="scroll-area" {...(props as React.HTMLAttributes<HTMLDivElement>)}>
      {children}
    </div>
  ),
}));

// ── Mock child components ───────────────────────────────────────
vi.mock("./streaming-answer", () => ({
  StreamingAnswer: (props: Record<string, unknown>) => (
    <div data-testid="streaming-answer" data-tokens={props.tokens} data-final={props.finalAnswer} data-is-streaming={String(props.isStreaming)} />
  ),
}));

vi.mock("./sources-list", () => ({
  SourcesList: (props: Record<string, unknown>) => (
    <div data-testid="sources-list" data-count={Array.isArray(props.sources) ? props.sources.length : 0} />
  ),
}));

vi.mock("./attribution-list", () => ({
  AttributionList: (props: Record<string, unknown>) => (
    <div data-testid="attribution-list" data-count={Array.isArray(props.attributions) ? props.attributions.length : 0} />
  ),
}));

vi.mock("./status-indicator", () => ({
  StatusIndicator: (props: Record<string, unknown>) => (
    <div data-testid="status-indicator" data-status={props.status} />
  ),
}));

vi.mock("./confidence-badge", () => ({
  ConfidenceBadge: (props: Record<string, unknown>) => (
    <div data-testid="confidence-badge" data-passed={String(props.passed)} data-confidence={props.confidence} />
  ),
}));

vi.mock("./error-display", () => ({
  ErrorDisplay: (props: Record<string, unknown>) => (
    <div data-testid="error-display" data-error={props.error}>
      {props.onRetry && <button data-testid="error-retry" onClick={props.onRetry as () => void}>Retry</button>}
    </div>
  ),
}));

// ── Helpers ─────────────────────────────────────────────────────

function makeStreaming(overrides: Partial<StreamingQueryState> = {}): StreamingQueryState {
  return {
    phase: "streaming",
    pipelineStatus: "generating",
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
    ...overrides,
  };
}

function userMsg(content: string, id?: string): Message {
  return { id: id ?? "u1", role: "user", content };
}

function assistantMsg(
  content: string,
  streaming?: StreamingQueryState,
  id?: string,
): Message {
  return { id: id ?? "a1", role: "assistant", content, streaming };
}

// ── Stub scrollIntoView (not implemented in jsdom) ──────────────
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

// ── Tests ───────────────────────────────────────────────────────

describe("MessageList", () => {
  it("renders user messages with right-aligned class (justify-end)", () => {
    render(<MessageList messages={[userMsg("Hello")]} />);
    const el = screen.getByLabelText("Your question");
    expect(el.className).toContain("justify-end");
  });

  it("user message shows question text in <p>", () => {
    render(<MessageList messages={[userMsg("My question")]} />);
    const p = screen.getByText("My question");
    expect(p.tagName).toBe("P");
  });

  it('user message has aria-label="Your question"', () => {
    render(<MessageList messages={[userMsg("test")]} />);
    expect(screen.getByLabelText("Your question")).toBeInTheDocument();
  });

  it("assistant message has robot avatar icon (Sparkles)", () => {
    render(<MessageList messages={[assistantMsg("hi")]} />);
    // Sparkles is rendered inside a div with specific classes
    const response = screen.getByLabelText("Assistant response");
    const avatarContainer = response.querySelector(".hidden.sm\\:flex");
    expect(avatarContainer).toBeInTheDocument();
  });

  it("avatar hidden on mobile (hidden sm:flex class)", () => {
    render(<MessageList messages={[assistantMsg("hi")]} />);
    const response = screen.getByLabelText("Assistant response");
    const avatarDiv = response.querySelector('[class*="hidden"]');
    expect(avatarDiv).toBeInTheDocument();
    expect(avatarDiv!.className).toContain("sm:flex");
  });

  it('assistant message has aria-label="Assistant response"', () => {
    render(<MessageList messages={[assistantMsg("test")]} />);
    expect(screen.getByLabelText("Assistant response")).toBeInTheDocument();
  });

  it("streaming msg: StatusIndicator rendered", () => {
    const streaming = makeStreaming({ pipelineStatus: "generating" });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    const indicator = screen.getByTestId("status-indicator");
    expect(indicator).toBeInTheDocument();
    expect(indicator.dataset.status).toBe("generating");
  });

  it("streaming msg: StreamingAnswer rendered with tokens", () => {
    const streaming = makeStreaming({ tokens: "Hello world" });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    const answer = screen.getByTestId("streaming-answer");
    expect(answer).toBeInTheDocument();
    expect(answer.dataset.tokens).toBe("Hello world");
  });

  it("error phase: ErrorDisplay rendered INSTEAD OF StreamingAnswer", () => {
    const streaming = makeStreaming({
      phase: "error",
      error: "Something went wrong",
    });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    expect(screen.getByTestId("error-display")).toBeInTheDocument();
    expect(screen.queryByTestId("streaming-answer")).not.toBeInTheDocument();
  });

  it("error phase: onRetry uses previous user message content", () => {
    const streaming = makeStreaming({
      phase: "error",
      error: "fail",
    });
    const onRetry = vi.fn();
    const msgs: Message[] = [
      userMsg("What is X?", "u1"),
      assistantMsg("", streaming, "a1"),
    ];
    render(<MessageList messages={msgs} onRetry={onRetry} />);
    const retryBtn = screen.getByTestId("error-retry");
    retryBtn.click();
    expect(onRetry).toHaveBeenCalledWith("What is X?");
  });

  it("onRetry not provided for first assistant msg (i=0 guard)", () => {
    const streaming = makeStreaming({
      phase: "error",
      error: "fail",
    });
    const onRetry = vi.fn();
    // assistant at index 0 means i=0 so onRetry should NOT be passed
    render(
      <MessageList messages={[assistantMsg("", streaming, "a1")]} onRetry={onRetry} />,
    );
    expect(screen.queryByTestId("error-retry")).not.toBeInTheDocument();
  });

  it("sources shown when streaming.sources.length > 0", () => {
    const streaming = makeStreaming({
      sources: [{ file_path: "a.md", section_title: "Intro", score: 0.9 }],
    });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    expect(screen.getByTestId("sources-list")).toBeInTheDocument();
  });

  it("sources hidden when empty array", () => {
    const streaming = makeStreaming({ sources: [] });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    expect(screen.queryByTestId("sources-list")).not.toBeInTheDocument();
  });

  it("attributions shown when streaming.attributions.length > 0", () => {
    const streaming = makeStreaming({
      attributions: [{ sentence: "test", source_index: 0, similarity: 0.95 }],
    });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    expect(screen.getByTestId("attribution-list")).toBeInTheDocument();
  });

  it("ConfidenceBadge shown when verification exists", () => {
    const streaming = makeStreaming({
      verification: { passed: true, confidence: 0.95 },
    });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    const badge = screen.getByTestId("confidence-badge");
    expect(badge).toBeInTheDocument();
    expect(badge.dataset.passed).toBe("true");
  });

  it('elapsed time "Answered in X.Xs" shown on complete', () => {
    const streaming = makeStreaming({
      phase: "complete",
      elapsed: 2.345,
    });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    expect(screen.getByText("Answered in 2.3s")).toBeInTheDocument();
  });

  it("elapsed time hidden when phase != complete", () => {
    const streaming = makeStreaming({
      phase: "streaming",
      elapsed: 1.5,
    });
    render(<MessageList messages={[assistantMsg("", streaming)]} />);
    expect(screen.queryByText(/Answered in/)).not.toBeInTheDocument();
  });

  it("empty messages array renders no messages", () => {
    render(<MessageList messages={[]} />);
    expect(screen.queryByLabelText("Your question")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Assistant response")).not.toBeInTheDocument();
  });

  it("multiple messages render in DOM order", () => {
    const msgs: Message[] = [
      userMsg("First", "u1"),
      assistantMsg("Second", undefined, "a1"),
      userMsg("Third", "u2"),
    ];
    render(<MessageList messages={msgs} />);
    const all = screen.getAllByLabelText(/Your question|Assistant response/);
    expect(all).toHaveLength(3);
    expect(all[0]).toHaveAttribute("aria-label", "Your question");
    expect(all[1]).toHaveAttribute("aria-label", "Assistant response");
    expect(all[2]).toHaveAttribute("aria-label", "Your question");
  });

  it("auto-scroll ref div exists at bottom", () => {
    const { container } = render(<MessageList messages={[userMsg("hi")]} />);
    const logDiv = container.querySelector('[role="log"]');
    // The ref div is the last child of the log container
    const lastChild = logDiv!.lastElementChild;
    expect(lastChild).toBeInTheDocument();
    expect(lastChild!.tagName).toBe("DIV");
    // It should be an empty div (the scroll anchor)
    expect(lastChild!.children).toHaveLength(0);
    expect(lastChild!.textContent).toBe("");
  });

  it('container has role="log" and aria-live="polite"', () => {
    render(<MessageList messages={[]} />);
    const log = screen.getByRole("log");
    expect(log).toHaveAttribute("aria-live", "polite");
  });

  it("non-streaming assistant uses msg.content for answer", () => {
    render(
      <MessageList messages={[assistantMsg("Direct content answer")]} />,
    );
    const answer = screen.getByTestId("streaming-answer");
    expect(answer.dataset.final).toBe("Direct content answer");
  });
});
