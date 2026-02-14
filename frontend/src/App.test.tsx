import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

// ── Constants ──────────────────────────────────────────────────────────
const MESSAGES_KEY = "doc-qa-messages";

// ── Mock return-value factories ────────────────────────────────────────

function makeStreamMock(overrides: Record<string, unknown> = {}) {
  return {
    phase: "idle" as string,
    pipelineStatus: null,
    tokens: "",
    answer: null as string | null,
    intent: null,
    intentConfidence: null,
    sources: [],
    chunksRetrieved: 0,
    attributions: [],
    verification: null,
    model: null,
    sessionId: null as string | null,
    diagrams: [],
    elapsed: null,
    error: null,
    submit: vi.fn(),
    cancel: vi.fn(),
    reset: vi.fn(),
    ...overrides,
  };
}

function makeSessionMock(overrides: Record<string, unknown> = {}) {
  return {
    sessionId: undefined as string | undefined,
    setSessionId: vi.fn(),
    clearSession: vi.fn(),
    ...overrides,
  };
}

function makeConversationsMock(overrides: Record<string, unknown> = {}) {
  return {
    conversations: [],
    loading: false,
    dbEnabled: null as boolean | null,
    refresh: vi.fn(),
    deleteConversation: vi.fn(),
    ...overrides,
  };
}

function makeSettingsMock(overrides: Record<string, unknown> = {}) {
  return {
    config: null,
    loading: false,
    open: false,
    setOpen: vi.fn(),
    updateSection: vi.fn(),
    saving: false,
    testDbConnection: vi.fn(),
    dbTestResult: null,
    runMigrations: vi.fn(),
    migrateResult: null,
    restartRequired: [],
    ...overrides,
  };
}

function makeTourMock(overrides: Record<string, unknown> = {}) {
  return {
    active: false,
    currentStep: 0,
    step: { id: "welcome", tab: null, title: "Welcome", description: "", required: true },
    totalSteps: 6,
    next: vi.fn(),
    back: vi.fn(),
    skip: vi.fn(),
    start: vi.fn(),
    finish: vi.fn(),
    isFirstVisit: false,
    ...overrides,
  };
}

// ── Shared mutable mock return values (reassigned per-test) ───────────

let mockStream = makeStreamMock();
let mockSession = makeSessionMock();
let mockConversations = makeConversationsMock();
let mockSettings = makeSettingsMock();
let mockTour = makeTourMock();
let mockApiConversationsGet: ReturnType<typeof vi.fn>;

// ── vi.mock calls (hoisted) ───────────────────────────────────────────

vi.mock("@/hooks/use-streaming-query", () => ({
  useStreamingQuery: () => mockStream,
}));

vi.mock("@/hooks/use-session", () => ({
  useSession: () => mockSession,
}));

vi.mock("@/hooks/use-conversations", () => ({
  useConversations: () => mockConversations,
}));

vi.mock("@/hooks/use-settings", () => ({
  useSettings: () => mockSettings,
}));

vi.mock("@/hooks/use-tour", () => ({
  useTour: () => mockTour,
}));

vi.mock("@/api/client", () => ({
  api: {
    conversations: {
      get: (...args: unknown[]) => mockApiConversationsGet(...args),
    },
  },
}));

// ── Mock child components ─────────────────────────────────────────────

vi.mock("@/components/chat-input", () => ({
  ChatInput: ({ onSubmit, onStop, isStreaming }: any) => (
    <div data-testid="chat-input" data-streaming={isStreaming}>
      <button data-testid="submit-btn" onClick={() => onSubmit("test question")}>
        Submit
      </button>
      <button data-testid="stop-btn" onClick={onStop}>
        Stop
      </button>
    </div>
  ),
}));

vi.mock("@/components/message-list", () => ({
  MessageList: ({ messages, onRetry }: any) => (
    <div data-testid="message-list" data-msg-count={messages.length}>
      {messages.map((m: any, i: number) => (
        <div
          key={m.id}
          data-testid={`msg-${i}`}
          data-role={m.role}
          data-has-streaming={!!m.streaming}
        >
          {m.content}
        </div>
      ))}
      {onRetry && (
        <button data-testid="retry-btn" onClick={() => onRetry("prev question")}>
          Retry
        </button>
      )}
    </div>
  ),
}));

vi.mock("@/components/welcome-screen", () => ({
  WelcomeScreen: ({ onSelectQuestion }: any) => (
    <div data-testid="welcome-screen">
      <button data-testid="example-btn" onClick={() => onSelectQuestion("example q")}>
        Example
      </button>
    </div>
  ),
}));

vi.mock("@/components/settings-dialog", () => ({
  SettingsDialog: (props: any) => (
    <div data-testid="settings-dialog" data-open={props.open} />
  ),
}));

vi.mock("@/components/conversation-sidebar", () => ({
  ConversationSidebar: (props: any) => (
    <div data-testid="sidebar" data-open={props.open}>
      <button data-testid="sidebar-select" onClick={() => props.onSelect("conv-1")}>
        Select
      </button>
      <button data-testid="sidebar-delete" onClick={() => props.onDelete("conv-1")}>
        Delete
      </button>
      <button data-testid="sidebar-new" onClick={props.onNew}>
        New
      </button>
    </div>
  ),
}));

vi.mock("@/components/connection-status", () => ({
  ConnectionStatus: () => <div data-testid="connection-status" />,
}));

vi.mock("@/components/theme-toggle", () => ({
  ThemeToggle: () => <div data-testid="theme-toggle" />,
}));

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
    span: ({ children, ...rest }: any) => <span {...rest}>{children}</span>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

vi.mock("lucide-react", () => ({
  Compass: () => <span data-testid="compass-icon" />,
  Menu: () => <span data-testid="menu-icon" />,
  MessageSquarePlus: () => <span data-testid="new-chat-icon" />,
  Settings: () => <span data-testid="settings-icon" />,
}));

// ── Import App AFTER mocks ────────────────────────────────────────────

import App from "./App";

// ── Test suite ────────────────────────────────────────────────────────

beforeEach(() => {
  sessionStorage.clear();
  mockStream = makeStreamMock();
  mockSession = makeSessionMock();
  mockConversations = makeConversationsMock();
  mockSettings = makeSettingsMock();
  mockTour = makeTourMock();
  mockApiConversationsGet = vi.fn();
});

// ── 1. Messages restored from sessionStorage on mount ─────────────────
describe("sessionStorage restore", () => {
  it("restores messages from sessionStorage on mount", () => {
    const saved = [
      { id: "1", role: "user", content: "hello" },
      { id: "2", role: "assistant", content: "world" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    render(<App />);

    // Should show MessageList (not WelcomeScreen) since messages exist
    expect(screen.getByTestId("message-list")).toBeInTheDocument();
    expect(screen.getByTestId("msg-0")).toHaveTextContent("hello");
    expect(screen.getByTestId("msg-1")).toHaveTextContent("world");
  });

  // ── 2. Corrupt sessionStorage JSON doesn't crash ──────────────────
  it("handles corrupt sessionStorage JSON without crashing", () => {
    sessionStorage.setItem(MESSAGES_KEY, "not valid json{{{");
    expect(() => render(<App />)).not.toThrow();
    // Falls back to empty → shows welcome screen
    expect(screen.getByTestId("welcome-screen")).toBeInTheDocument();
  });

  // ── 3. Empty content assistant messages filtered on restore ────────
  it("filters out assistant messages with empty content on restore", () => {
    const saved = [
      { id: "1", role: "user", content: "hello" },
      { id: "2", role: "assistant", content: "" },
      { id: "3", role: "assistant", content: "   " },
      { id: "4", role: "user", content: "second" },
      { id: "5", role: "assistant", content: "real answer" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    render(<App />);

    const list = screen.getByTestId("message-list");
    // Empty and whitespace-only assistant messages are filtered, user messages kept
    expect(list.getAttribute("data-msg-count")).toBe("3");
  });
});

// ── 4-5. submitQuestion ──────────────────────────────────────────────
describe("submitQuestion", () => {
  it("adds user + empty assistant pair on submit", async () => {
    const user = userEvent.setup();
    render(<App />);

    // Initially shows welcome screen (no messages)
    expect(screen.getByTestId("welcome-screen")).toBeInTheDocument();

    await user.click(screen.getByTestId("submit-btn"));

    // After submit, we should see the message list with 2 messages
    expect(screen.getByTestId("message-list")).toBeInTheDocument();
    expect(screen.getByTestId("message-list").getAttribute("data-msg-count")).toBe("2");
    expect(screen.getByTestId("msg-0")).toHaveAttribute("data-role", "user");
    expect(screen.getByTestId("msg-0")).toHaveTextContent("test question");
    expect(screen.getByTestId("msg-1")).toHaveAttribute("data-role", "assistant");
  });

  it("calls stream.submit with question and sessionId", async () => {
    mockSession = makeSessionMock({ sessionId: "sess-42" });
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("submit-btn"));

    expect(mockStream.submit).toHaveBeenCalledWith("test question", "sess-42");
  });
});

// ── 6-7. handleRetry ────────────────────────────────────────────────
describe("handleRetry", () => {
  it("removes last message and adds new assistant on retry", async () => {
    const saved = [
      { id: "1", role: "user", content: "original question" },
      { id: "2", role: "assistant", content: "bad answer" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));
    const user = userEvent.setup();
    render(<App />);

    // Verify 2 messages initially
    expect(screen.getByTestId("message-list").getAttribute("data-msg-count")).toBe("2");

    await user.click(screen.getByTestId("retry-btn"));

    // After retry: removes last, adds new assistant → still 2 messages
    expect(screen.getByTestId("message-list").getAttribute("data-msg-count")).toBe("2");
    // The first message is kept, the second is a new empty assistant
    expect(screen.getByTestId("msg-0")).toHaveTextContent("original question");
  });

  it("calls stream.submit on retry", async () => {
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "a" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("retry-btn"));

    expect(mockStream.submit).toHaveBeenCalledWith("prev question", mockSession.sessionId);
  });
});

// ── 8. handleNewChat ────────────────────────────────────────────────
describe("handleNewChat", () => {
  it("resets stream, messages, and session", async () => {
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "a" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));
    const user = userEvent.setup();
    render(<App />);

    // Messages exist, so New Chat button should be visible
    const newChatBtn = screen.getByRole("button", { name: "New conversation" });
    await user.click(newChatBtn);

    expect(mockStream.reset).toHaveBeenCalled();
    expect(mockSession.clearSession).toHaveBeenCalled();
    // Messages should be cleared → welcome screen shown
    expect(screen.getByTestId("welcome-screen")).toBeInTheDocument();
  });
});

// ── 9. Session ID synced from stream ────────────────────────────────
describe("session sync", () => {
  it("syncs session ID from stream to useSession", () => {
    mockStream = makeStreamMock({ sessionId: "stream-sess-1" });
    render(<App />);

    expect(mockSession.setSessionId).toHaveBeenCalledWith("stream-sess-1");
  });

  // ── 10. Session ID not re-set if same value ─────────────────────────
  it("does not call setSessionId again when stream sessionId is unchanged", () => {
    mockStream = makeStreamMock({ sessionId: "stream-sess-1" });
    const { rerender } = render(<App />);

    expect(mockSession.setSessionId).toHaveBeenCalledTimes(1);

    // Rerender with same sessionId on stream
    rerender(<App />);

    // Still only called once — the ref guard prevents duplicate calls
    expect(mockSession.setSessionId).toHaveBeenCalledTimes(1);
  });
});

// ── 11. Final answer persisted ──────────────────────────────────────
describe("final answer persistence", () => {
  it("persists final answer to last assistant message when stream completes", async () => {
    // Pre-populate with a non-empty assistant (survives restore filter)
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "old answer" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    // Stream is in complete phase with a final answer
    mockStream = makeStreamMock({ phase: "complete", answer: "the answer" });
    render(<App />);

    // The useEffect should update the last assistant message content
    await waitFor(() => {
      expect(screen.getByTestId("msg-1")).toHaveTextContent("the answer");
    });
  });
});

// ── 12-13. Sidebar refresh after stream completes ───────────────────
describe("sidebar refresh", () => {
  it("refreshes conversations after stream completes when dbEnabled=true", () => {
    mockStream = makeStreamMock({ phase: "complete" });
    mockConversations = makeConversationsMock({ dbEnabled: true });
    render(<App />);

    expect(mockConversations.refresh).toHaveBeenCalled();
  });

  it("does not refresh when dbEnabled=false", () => {
    mockStream = makeStreamMock({ phase: "complete" });
    mockConversations = makeConversationsMock({ dbEnabled: false });
    render(<App />);

    expect(mockConversations.refresh).not.toHaveBeenCalled();
  });
});

// ── 14-15. handleSelectConversation ──────────────────────────────────
describe("handleSelectConversation", () => {
  it("loads messages from API when selecting a conversation", async () => {
    mockConversations = makeConversationsMock({ dbEnabled: true });
    mockApiConversationsGet.mockResolvedValue({
      id: "conv-1",
      messages: [
        { id: "m1", role: "user", content: "loaded q" },
        { id: "m2", role: "assistant", content: "loaded a" },
      ],
    });

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("sidebar-select"));

    await waitFor(() => {
      expect(mockApiConversationsGet).toHaveBeenCalledWith("conv-1");
    });
    await waitFor(() => {
      expect(screen.getByTestId("message-list")).toBeInTheDocument();
    });
    expect(mockStream.reset).toHaveBeenCalled();
    expect(mockSession.setSessionId).toHaveBeenCalledWith("conv-1");
  });

  it("handles API failure without crashing", async () => {
    mockConversations = makeConversationsMock({ dbEnabled: true });
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    mockApiConversationsGet.mockRejectedValue(new Error("network error"));

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("sidebar-select"));

    await waitFor(() => {
      expect(consoleError).toHaveBeenCalledWith(
        "Failed to load conversation:",
        expect.any(Error),
      );
    });

    consoleError.mockRestore();
  });
});

// ── 16-17. handleDeleteConversation ──────────────────────────────────
describe("handleDeleteConversation", () => {
  it("clears chat when deleting the active conversation", async () => {
    mockConversations = makeConversationsMock({
      dbEnabled: true,
      deleteConversation: vi.fn().mockResolvedValue(undefined),
    });
    mockSession = makeSessionMock({ sessionId: "conv-1" });
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("sidebar-delete"));

    await waitFor(() => {
      expect(mockConversations.deleteConversation).toHaveBeenCalledWith("conv-1");
    });
    // Deleting the active conversation triggers handleNewChat
    expect(mockStream.reset).toHaveBeenCalled();
    expect(mockSession.clearSession).toHaveBeenCalled();
  });

  it("does not clear chat when deleting a non-active conversation", async () => {
    mockConversations = makeConversationsMock({
      dbEnabled: true,
      deleteConversation: vi.fn().mockResolvedValue(undefined),
    });
    // Active session is different from the one being deleted
    mockSession = makeSessionMock({ sessionId: "conv-other" });

    // Pre-populate messages so we can verify they persist
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "a" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("sidebar-delete"));

    await waitFor(() => {
      expect(mockConversations.deleteConversation).toHaveBeenCalledWith("conv-1");
    });

    // handleNewChat should NOT have been called
    expect(mockStream.reset).not.toHaveBeenCalled();
    expect(mockSession.clearSession).not.toHaveBeenCalled();
    // Messages should still be there
    expect(screen.getByTestId("message-list")).toBeInTheDocument();
  });
});

// ── 18-19. displayMessages streaming injection ──────────────────────
describe("displayMessages streaming state", () => {
  it("injects streaming state into last assistant message when not idle", async () => {
    // Use submitQuestion to create user+empty assistant pair (avoids sessionStorage filter)
    mockStream = makeStreamMock({ phase: "streaming" });
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("submit-btn"));

    // The last assistant message should have streaming set (phase !== "idle")
    expect(screen.getByTestId("msg-1")).toHaveAttribute("data-has-streaming", "true");
  });

  it("does not inject streaming into non-streaming assistant messages", () => {
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "done" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    // Phase is idle, so no streaming injection
    mockStream = makeStreamMock({ phase: "idle" });
    render(<App />);

    expect(screen.getByTestId("msg-1")).toHaveAttribute("data-has-streaming", "false");
  });
});

// ── 20-21. New Chat button visibility ───────────────────────────────
describe("New Chat button visibility", () => {
  it("hides New Chat button when no messages exist", () => {
    render(<App />);

    expect(screen.queryByRole("button", { name: "New conversation" })).not.toBeInTheDocument();
  });

  it("shows New Chat button when messages exist", () => {
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "a" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    render(<App />);

    expect(screen.getByRole("button", { name: "New conversation" })).toBeInTheDocument();
  });
});

// ── 22-23. Sidebar visibility ───────────────────────────────────────
describe("sidebar visibility", () => {
  it("hides sidebar when dbEnabled is not true", () => {
    mockConversations = makeConversationsMock({ dbEnabled: false });
    render(<App />);

    expect(screen.queryByTestId("sidebar")).not.toBeInTheDocument();
  });

  it("shows sidebar when dbEnabled is true", () => {
    mockConversations = makeConversationsMock({ dbEnabled: true });
    render(<App />);

    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
  });
});

// ── 24-25. Tour link ────────────────────────────────────────────────
describe("Take a Tour", () => {
  it("hides tour link when tour is active", () => {
    mockTour = makeTourMock({ active: true });
    render(<App />);

    expect(screen.queryByText("Take a Tour")).not.toBeInTheDocument();
  });

  it("clicking tour link starts tour and opens settings", async () => {
    mockTour = makeTourMock({ active: false });
    const user = userEvent.setup();
    render(<App />);

    const tourBtn = screen.getByText("Take a Tour");
    await user.click(tourBtn);

    expect(mockTour.start).toHaveBeenCalled();
    expect(mockSettings.setOpen).toHaveBeenCalledWith(true);
  });
});

// ── 26. Tour auto-opens settings on first visit ─────────────────────
describe("tour auto-open settings", () => {
  it("auto-opens settings dialog when tour is active and isFirstVisit", () => {
    mockTour = makeTourMock({ active: true, isFirstVisit: true });
    render(<App />);

    expect(mockSettings.setOpen).toHaveBeenCalledWith(true);
  });
});

// ── 27-28. Persist / clear sessionStorage ───────────────────────────
describe("sessionStorage persistence", () => {
  it("persists messages to sessionStorage when messages change", async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByTestId("submit-btn"));

    const stored = sessionStorage.getItem(MESSAGES_KEY);
    expect(stored).not.toBeNull();
    const parsed = JSON.parse(stored!);
    expect(parsed).toHaveLength(2);
    expect(parsed[0].role).toBe("user");
    expect(parsed[0].content).toBe("test question");
  });

  it("removes sessionStorage key when messages are empty", async () => {
    // Start with messages
    const saved = [
      { id: "1", role: "user", content: "q" },
      { id: "2", role: "assistant", content: "a" },
    ];
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(saved));

    const user = userEvent.setup();
    render(<App />);

    // Clear via New Chat button
    const newChatBtn = screen.getByRole("button", { name: "New conversation" });
    await user.click(newChatBtn);

    expect(sessionStorage.getItem(MESSAGES_KEY)).toBeNull();
  });
});
