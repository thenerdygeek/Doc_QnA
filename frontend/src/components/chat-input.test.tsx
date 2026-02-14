import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChatInput } from "./chat-input";
import { vi } from "vitest";

// Framer Motion mocks to avoid animation timing issues in tests
vi.mock("framer-motion", () => ({
  motion: {
    div: ({
      children,
      ...props
    }: React.PropsWithChildren<Record<string, unknown>>) => (
      <div {...filterDomProps(props)}>{children}</div>
    ),
  },
  AnimatePresence: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

function filterDomProps(props: Record<string, unknown>) {
  const invalid = [
    "initial",
    "animate",
    "exit",
    "transition",
    "whileHover",
    "whileTap",
    "variants",
  ];
  const clean: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) {
    if (!invalid.includes(k)) clean[k] = v;
  }
  return clean;
}

describe("ChatInput", () => {
  const defaultProps = {
    onSubmit: vi.fn(),
    onStop: vi.fn(),
    isStreaming: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the textarea with placeholder", () => {
    render(<ChatInput {...defaultProps} />);
    expect(
      screen.getByPlaceholderText("Ask about your docs..."),
    ).toBeInTheDocument();
  });

  it("renders send button when not streaming", () => {
    render(<ChatInput {...defaultProps} />);
    expect(screen.getByLabelText("Send question")).toBeInTheDocument();
  });

  it("renders stop button when streaming", () => {
    render(<ChatInput {...defaultProps} isStreaming />);
    expect(screen.getByLabelText("Stop generating")).toBeInTheDocument();
  });

  it("send button is disabled when input is empty", () => {
    render(<ChatInput {...defaultProps} />);
    expect(screen.getByLabelText("Send question")).toBeDisabled();
  });

  it("send button enables when text is entered", async () => {
    const user = userEvent.setup();
    render(<ChatInput {...defaultProps} />);

    await user.type(
      screen.getByPlaceholderText("Ask about your docs..."),
      "Hello",
    );
    expect(screen.getByLabelText("Send question")).toBeEnabled();
  });

  it("calls onSubmit with trimmed text on form submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput {...defaultProps} onSubmit={onSubmit} />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.type(textarea, "  How does auth work?  ");
    await user.click(screen.getByLabelText("Send question"));

    expect(onSubmit).toHaveBeenCalledWith("How does auth work?");
  });

  it("clears input after submit", async () => {
    const user = userEvent.setup();
    render(<ChatInput {...defaultProps} />);

    const textarea = screen.getByPlaceholderText(
      "Ask about your docs...",
    ) as HTMLTextAreaElement;
    await user.type(textarea, "question");
    await user.click(screen.getByLabelText("Send question"));

    expect(textarea.value).toBe("");
  });

  it("submits on Enter key", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput {...defaultProps} onSubmit={onSubmit} />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.type(textarea, "test{Enter}");

    expect(onSubmit).toHaveBeenCalledWith("test");
  });

  it("does not submit on Shift+Enter", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput {...defaultProps} onSubmit={onSubmit} />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.type(textarea, "line1{Shift>}{Enter}{/Shift}line2");

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("calls onStop when stop button is clicked", async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();
    render(<ChatInput {...defaultProps} onStop={onStop} isStreaming />);

    await user.click(screen.getByLabelText("Stop generating"));
    expect(onStop).toHaveBeenCalled();
  });

  it("disables textarea when disabled prop is true", () => {
    render(<ChatInput {...defaultProps} disabled />);
    expect(
      screen.getByPlaceholderText("Ask about your docs..."),
    ).toBeDisabled();
  });
});

describe("ChatInput – gap coverage", () => {
  const defaultProps = {
    onSubmit: vi.fn(),
    onStop: vi.fn(),
    isStreaming: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("Escape key calls onStop when streaming", async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();
    render(<ChatInput {...defaultProps} onStop={onStop} isStreaming />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.click(textarea);
    await user.keyboard("{Escape}");

    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("Escape key does nothing when not streaming", async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();
    render(<ChatInput {...defaultProps} onStop={onStop} isStreaming={false} />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.click(textarea);
    await user.keyboard("{Escape}");

    expect(onStop).not.toHaveBeenCalled();
  });

  it("submit is blocked during streaming", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput {...defaultProps} onSubmit={onSubmit} isStreaming />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.type(textarea, "hello{Enter}");

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("whitespace-only input does not submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ChatInput {...defaultProps} onSubmit={onSubmit} />);

    const textarea = screen.getByPlaceholderText("Ask about your docs...");
    await user.type(textarea, "   {Enter}");

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("textarea height resets to 'auto' after submit", async () => {
    const user = userEvent.setup();
    render(<ChatInput {...defaultProps} />);

    const textarea = screen.getByPlaceholderText(
      "Ask about your docs...",
    ) as HTMLTextAreaElement;
    await user.type(textarea, "question");
    await user.click(screen.getByLabelText("Send question"));

    expect(textarea.style.height).toBe("auto");
  });

  it("form has role='search' attribute", () => {
    render(<ChatInput {...defaultProps} />);
    expect(screen.getByRole("search")).toBeInTheDocument();
  });

  it("form has aria-label='Ask a question'", () => {
    render(<ChatInput {...defaultProps} />);
    expect(screen.getByLabelText("Ask a question")).toBeInTheDocument();
  });

  it("helper text 'Enter to send' has sm:block class", () => {
    const { container } = render(<ChatInput {...defaultProps} />);
    const helperText = container.querySelector("p");
    expect(helperText).not.toBeNull();
    expect(helperText!.className).toContain("sm:block");
  });
});
