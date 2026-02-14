import { render, screen } from "@testing-library/react";
import { ConfidenceBadge } from "./confidence-badge";

describe("ConfidenceBadge", () => {
  it("renders 'Verified' when passed is true", () => {
    render(<ConfidenceBadge passed confidence={0.95} />);
    expect(screen.getByText("Verified (95%)")).toBeInTheDocument();
  });

  it("renders 'Unverified' when passed is false", () => {
    render(<ConfidenceBadge passed={false} confidence={0.42} />);
    expect(screen.getByText("Unverified (42%)")).toBeInTheDocument();
  });

  it("rounds confidence percentage", () => {
    render(<ConfidenceBadge passed confidence={0.876} />);
    expect(screen.getByText("Verified (88%)")).toBeInTheDocument();
  });

  it("applies emerald styling when verified", () => {
    const { container } = render(
      <ConfidenceBadge passed confidence={0.9} />,
    );
    const badge = container.firstElementChild!;
    expect(badge.className).toContain("emerald");
  });

  it("applies amber styling when unverified", () => {
    const { container } = render(
      <ConfidenceBadge passed={false} confidence={0.3} />,
    );
    const badge = container.firstElementChild!;
    expect(badge.className).toContain("amber");
  });
});

describe("ConfidenceBadge – gap coverage", () => {
  it("confidence 0.0 shows '0%'", () => {
    render(<ConfidenceBadge passed={false} confidence={0.0} />);
    expect(screen.getByText("Unverified (0%)")).toBeInTheDocument();
  });

  it("confidence 1.0 shows '100%'", () => {
    render(<ConfidenceBadge passed confidence={1.0} />);
    expect(screen.getByText("Verified (100%)")).toBeInTheDocument();
  });

  it("confidence 0.005 rounds to '1%'", () => {
    render(<ConfidenceBadge passed confidence={0.005} />);
    expect(screen.getByText("Verified (1%)")).toBeInTheDocument();
  });
});
