import { render, screen } from "@testing-library/react";
import { StatusIndicator } from "./status-indicator";
import { vi } from "vitest";

vi.mock("framer-motion", () => ({
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
  useReducedMotion: () => false,
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

  it("renders classifying step as active", () => {
    render(<StatusIndicator status="classifying" />);
    expect(screen.getByText("Classify")).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders retrieving step with Search label", () => {
    render(<StatusIndicator status="retrieving" />);
    expect(screen.getByText("Search")).toBeInTheDocument();
  });

  it("renders generating step", () => {
    render(<StatusIndicator status="generating" />);
    expect(screen.getByText("Generate")).toBeInTheDocument();
  });

  it("renders verifying step", () => {
    render(<StatusIndicator status="verifying" />);
    expect(screen.getByText("Verify")).toBeInTheDocument();
  });

  it("has status role for accessibility", () => {
    render(<StatusIndicator status="retrieving" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows all 5 pipeline step labels", () => {
    render(<StatusIndicator status="grading" />);
    expect(screen.getByText("Classify")).toBeInTheDocument();
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByText("Grade")).toBeInTheDocument();
    expect(screen.getByText("Generate")).toBeInTheDocument();
    expect(screen.getByText("Verify")).toBeInTheDocument();
  });
});

describe("StatusIndicator – gap coverage", () => {
  it("renders grading pipeline status with Grade label", () => {
    render(<StatusIndicator status="grading" />);
    expect(screen.getByText("Grade")).toBeInTheDocument();
  });
});
