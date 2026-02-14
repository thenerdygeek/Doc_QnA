import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { ThemeToggle } from "./theme-toggle";

// ── Mock useTheme hook ──────────────────────────────────────────────
const mockToggleTheme = vi.fn();
let mockResolvedTheme: "light" | "dark" = "dark";

vi.mock("@/hooks/use-theme", () => ({
  useTheme: () => ({
    resolvedTheme: mockResolvedTheme,
    toggleTheme: mockToggleTheme,
  }),
}));

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
      span: ({
        children,
        ...props
      }: React.PropsWithChildren<Record<string, unknown>>) => (
        <span data-testid="motion-span" {...filterDomProps(props)}>
          {children}
        </span>
      ),
    },
  };
});

describe("ThemeToggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockResolvedTheme = "dark";
  });

  it("renders Moon icon when resolvedTheme is 'dark'", () => {
    mockResolvedTheme = "dark";
    render(<ThemeToggle />);
    // The button aria-label indicates the action to switch TO light mode (means we are in dark)
    expect(screen.getByLabelText("Switch to light mode")).toBeInTheDocument();
  });

  it("renders Sun icon when resolvedTheme is 'light'", () => {
    mockResolvedTheme = "light";
    render(<ThemeToggle />);
    expect(screen.getByLabelText("Switch to dark mode")).toBeInTheDocument();
  });

  it("has aria-label='Switch to light mode' in dark mode", () => {
    mockResolvedTheme = "dark";
    render(<ThemeToggle />);
    const btn = screen.getByRole("button");
    expect(btn).toHaveAttribute("aria-label", "Switch to light mode");
  });

  it("has aria-label='Switch to dark mode' in light mode", () => {
    mockResolvedTheme = "light";
    render(<ThemeToggle />);
    const btn = screen.getByRole("button");
    expect(btn).toHaveAttribute("aria-label", "Switch to dark mode");
  });

  it("calls toggleTheme on click", async () => {
    const user = userEvent.setup();
    render(<ThemeToggle />);

    await user.click(screen.getByRole("button"));
    expect(mockToggleTheme).toHaveBeenCalledOnce();
  });

  it("AnimatePresence wraps icon with motion.span", () => {
    render(<ThemeToggle />);
    const motionSpan = screen.getByTestId("motion-span");
    expect(motionSpan).toBeInTheDocument();
    // The motion.span should be inside the button
    expect(screen.getByRole("button")).toContainElement(motionSpan);
  });
});
