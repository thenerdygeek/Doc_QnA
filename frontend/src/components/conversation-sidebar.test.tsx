import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { ConversationSidebar } from "./conversation-sidebar";
import type { ConversationSummary } from "@/types/api";

// jsdom doesn't provide ResizeObserver (needed by Radix ScrollArea)
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

// ── Framer Motion mock ──────────────────────────────────────────────
vi.mock("framer-motion", () => {
  const filterDomProps = (props: Record<string, unknown>) => {
    const invalid = [
      "initial",
      "animate",
      "exit",
      "transition",
      "whileHover",
      "whileTap",
      "variants",
      "layout",
    ];
    const clean: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(props)) {
      if (!invalid.includes(k)) clean[k] = v;
    }
    return clean;
  };

  return {
    AnimatePresence: ({ children }: React.PropsWithChildren) => (
      <>{children}</>
    ),
    motion: {
      div: ({
        children,
        ...props
      }: React.PropsWithChildren<Record<string, unknown>>) => (
        <div {...filterDomProps(props)}>{children}</div>
      ),
      button: ({
        children,
        ...props
      }: React.PropsWithChildren<Record<string, unknown>>) => (
        <button {...filterDomProps(props)}>{children}</button>
      ),
    },
    useReducedMotion: () => false,
  };
});

// ── Helpers ─────────────────────────────────────────────────────────
function makeConv(
  overrides: Partial<ConversationSummary> = {},
): ConversationSummary {
  return {
    id: "conv-1",
    user_id: null,
    title: "First chat",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

const defaultProps = () => ({
  conversations: [
    makeConv({ id: "conv-1", title: "First chat" }),
    makeConv({ id: "conv-2", title: "Second chat" }),
  ],
  activeId: "conv-1",
  onSelect: vi.fn(),
  onNew: vi.fn(),
  onDelete: vi.fn(),
  open: true,
  onClose: vi.fn(),
});

describe("ConversationSidebar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ── Rendering ───────────────────────────────────────────────────
  it("renders conversation titles", () => {
    render(<ConversationSidebar {...defaultProps()} />);
    expect(screen.getByText("First chat")).toBeInTheDocument();
    expect(screen.getByText("Second chat")).toBeInTheDocument();
  });

  it("falls back to 'Untitled' when title is empty", () => {
    const props = defaultProps();
    props.conversations = [makeConv({ id: "c-empty", title: "" })];
    render(<ConversationSidebar {...props} />);
    expect(screen.getByText("Untitled")).toBeInTheDocument();
  });

  it("applies bg-accent class to active conversation", () => {
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);
    const activeBtn = screen.getByText("First chat").closest("button")!;
    expect(activeBtn.className).toContain("bg-accent");
  });

  it("applies base class to inactive conversation", () => {
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);
    const inactiveBtn = screen.getByText("Second chat").closest("button")!;
    expect(inactiveBtn.className).toContain("text-foreground");
    // Active conversation should not have the base-only class
    const activeBtn = screen.getByText("First chat").closest("button")!;
    expect(activeBtn.className).toContain("bg-accent");
  });

  // ── Interactions ────────────────────────────────────────────────
  it("calls onSelect with conversation id when clicked", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    await user.click(screen.getByText("Second chat").closest("button")!);
    expect(props.onSelect).toHaveBeenCalledWith("conv-2");
  });

  it("calls onNew when New Chat button is clicked", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    await user.click(screen.getByLabelText("New conversation"));
    expect(props.onNew).toHaveBeenCalledOnce();
  });

  it("calls onDelete with conversation id on delete button click", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    const deleteBtns = screen.getAllByLabelText("Delete conversation");
    await user.click(deleteBtns[0]);
    expect(props.onDelete).toHaveBeenCalledWith("conv-1");
  });

  it("delete click stops propagation (onSelect NOT called)", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    const deleteBtns = screen.getAllByLabelText("Delete conversation");
    await user.click(deleteBtns[0]);
    expect(props.onDelete).toHaveBeenCalledWith("conv-1");
    expect(props.onSelect).not.toHaveBeenCalled();
  });

  it("delete keyboard: Enter key calls onDelete", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    const deleteBtns = screen.getAllByLabelText("Delete conversation");
    deleteBtns[0].focus();
    await user.keyboard("{Enter}");
    expect(props.onDelete).toHaveBeenCalledWith("conv-1");
  });

  it("delete keyboard: Space key calls onDelete", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    const deleteBtns = screen.getAllByLabelText("Delete conversation");
    deleteBtns[0].focus();
    await user.keyboard(" ");
    expect(props.onDelete).toHaveBeenCalledWith("conv-1");
  });

  // ── Empty state ─────────────────────────────────────────────────
  it("shows 'No conversations yet' when list is empty", () => {
    const props = defaultProps();
    props.conversations = [];
    render(<ConversationSidebar {...props} />);
    expect(screen.getByText("No conversations yet")).toBeInTheDocument();
  });

  // ── Close button ────────────────────────────────────────────────
  it("calls onClose when close button is clicked", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);

    await user.click(screen.getByLabelText("Close sidebar"));
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("close button has md:hidden class (mobile only)", () => {
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);
    const closeBtn = screen.getByLabelText("Close sidebar");
    expect(closeBtn.className).toContain("md:hidden");
  });

  // ── formatTimeAgo ───────────────────────────────────────────────
  it("formatTimeAgo: < 1 min shows 'just now'", () => {
    const props = defaultProps();
    props.conversations = [
      makeConv({ id: "t1", updated_at: new Date().toISOString() }),
    ];
    render(<ConversationSidebar {...props} />);
    expect(screen.getByText("just now")).toBeInTheDocument();
  });

  it("formatTimeAgo: 5 min shows '5m ago'", () => {
    const props = defaultProps();
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    props.conversations = [makeConv({ id: "t2", updated_at: fiveMinAgo })];
    render(<ConversationSidebar {...props} />);
    expect(screen.getByText("5m ago")).toBeInTheDocument();
  });

  it("formatTimeAgo: 2 hours shows '2h ago'", () => {
    const props = defaultProps();
    const twoHoursAgo = new Date(
      Date.now() - 2 * 60 * 60_000,
    ).toISOString();
    props.conversations = [makeConv({ id: "t3", updated_at: twoHoursAgo })];
    render(<ConversationSidebar {...props} />);
    expect(screen.getByText("2h ago")).toBeInTheDocument();
  });

  it("formatTimeAgo: 3 days shows '3d ago'", () => {
    const props = defaultProps();
    const threeDaysAgo = new Date(
      Date.now() - 3 * 24 * 60 * 60_000,
    ).toISOString();
    props.conversations = [
      makeConv({ id: "t4", updated_at: threeDaysAgo }),
    ];
    render(<ConversationSidebar {...props} />);
    expect(screen.getByText("3d ago")).toBeInTheDocument();
  });

  it("formatTimeAgo: 8+ days shows locale date string", () => {
    const props = defaultProps();
    const tenDaysAgo = new Date(
      Date.now() - 10 * 24 * 60 * 60_000,
    );
    props.conversations = [
      makeConv({ id: "t5", updated_at: tenDaysAgo.toISOString() }),
    ];
    render(<ConversationSidebar {...props} />);
    const expected = tenDaysAgo.toLocaleDateString();
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  // ── Open / closed state ─────────────────────────────────────────
  it("sidebar has -translate-x-full class when open=false", () => {
    const props = defaultProps();
    props.open = false;
    render(<ConversationSidebar {...props} />);
    const aside = screen.getByRole("complementary");
    expect(aside.className).toContain("-translate-x-full");
  });

  it("mobile backdrop is visible when open", () => {
    const props = defaultProps();
    const { container } = render(<ConversationSidebar {...props} />);
    const backdrop = container.querySelector(".bg-black\\/40");
    expect(backdrop).toBeInTheDocument();
  });

  it("backdrop click calls onClose", async () => {
    const user = userEvent.setup();
    const props = defaultProps();
    const { container } = render(<ConversationSidebar {...props} />);
    const backdrop = container.querySelector(".bg-black\\/40")!;
    await user.click(backdrop);
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  // ── Semantics / Accessibility ───────────────────────────────────
  it("sidebar has semantic <aside> element", () => {
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);
    expect(screen.getByRole("complementary")).toBeInTheDocument();
  });

  it("delete button has aria-label='Delete conversation'", () => {
    const props = defaultProps();
    render(<ConversationSidebar {...props} />);
    const deleteBtns = screen.getAllByLabelText("Delete conversation");
    expect(deleteBtns.length).toBeGreaterThan(0);
    expect(deleteBtns[0]).toHaveAttribute(
      "aria-label",
      "Delete conversation",
    );
  });
});
