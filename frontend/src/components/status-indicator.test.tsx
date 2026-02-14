import { render, screen } from "@testing-library/react";
import { StatusIndicator } from "./status-indicator";
import { vi } from "vitest";

vi.mock("framer-motion", () => ({
  AnimatePresence: ({ children }: React.PropsWithChildren) => <>{children}</>,
  motion: {
    div: ({
      children,
      className,
      role,
      "aria-live": ariaLive,
      "aria-label": ariaLabel,
    }: Record<string, unknown> & React.PropsWithChildren) => (
      <div
        className={className as string}
        role={role as string}
        aria-live={ariaLive as "off" | "polite" | "assertive" | undefined}
        aria-label={ariaLabel as string}
      >
        {children}
      </div>
    ),
  },
}));

describe("StatusIndicator", () => {
  it("renders nothing when status is null", () => {
    const { container } = render(<StatusIndicator status={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when status is complete", () => {
    const { container } = render(<StatusIndicator status="complete" />);
    expect(container.firstChild).toBeNull();
  });

  it("renders classifying status", () => {
    render(<StatusIndicator status="classifying" />);
    expect(screen.getByText("Classifying intent\u2026")).toBeInTheDocument();
  });

  it("renders retrieving status", () => {
    render(<StatusIndicator status="retrieving" />);
    expect(
      screen.getByText("Searching documentation\u2026"),
    ).toBeInTheDocument();
  });

  it("renders generating status", () => {
    render(<StatusIndicator status="generating" />);
    expect(screen.getByText("Generating answer\u2026")).toBeInTheDocument();
  });

  it("renders verifying status", () => {
    render(<StatusIndicator status="verifying" />);
    expect(screen.getByText("Verifying answer\u2026")).toBeInTheDocument();
  });

  it("has status role for accessibility", () => {
    render(<StatusIndicator status="retrieving" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});

describe("StatusIndicator – gap coverage", () => {
  it("renders grading pipeline status with correct text", () => {
    render(<StatusIndicator status="grading" />);
    expect(screen.getByText("Grading relevance\u2026")).toBeInTheDocument();
  });
});
