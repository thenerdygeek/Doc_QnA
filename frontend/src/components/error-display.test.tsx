import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErrorDisplay } from "./error-display";
import { vi } from "vitest";

vi.mock("framer-motion", () => ({
  motion: {
    div: ({
      children,
      className,
      role,
    }: React.PropsWithChildren<{ className?: string; role?: string }>) => (
      <div className={className} role={role}>
        {children}
      </div>
    ),
  },
}));

describe("ErrorDisplay", () => {
  it("renders error message", () => {
    render(<ErrorDisplay error="Connection failed" />);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Connection failed")).toBeInTheDocument();
  });

  it("has alert role for accessibility", () => {
    render(<ErrorDisplay error="Error" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("renders retry button when onRetry is provided", () => {
    render(<ErrorDisplay error="Error" onRetry={() => {}} />);
    expect(screen.getByLabelText("Retry question")).toBeInTheDocument();
  });

  it("does not render retry button when onRetry is absent", () => {
    render(<ErrorDisplay error="Error" />);
    expect(screen.queryByLabelText("Retry question")).not.toBeInTheDocument();
  });

  it("calls onRetry when retry button is clicked", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<ErrorDisplay error="Error" onRetry={onRetry} />);

    await user.click(screen.getByLabelText("Retry question"));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
