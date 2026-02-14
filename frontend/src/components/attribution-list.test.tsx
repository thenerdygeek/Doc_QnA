import { render, screen } from "@testing-library/react";
import { vi } from "vitest";
import { AttributionList } from "./attribution-list";
import type { AttributionInfo } from "@/types/api";

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
    motion: {
      div: ({
        children,
        ...props
      }: React.PropsWithChildren<Record<string, unknown>>) => (
        <div {...filterDomProps(props)}>{children}</div>
      ),
    },
  };
});

// ── Test data ───────────────────────────────────────────────────────
const sampleAttributions: AttributionInfo[] = [
  { sentence: "The API returns JSON responses.", source_index: 0, similarity: 0.92 },
  { sentence: "Auth uses JWT tokens.", source_index: 2, similarity: 0.85 },
  { sentence: "Rate limiting is 100 req/min.", source_index: 1, similarity: 0.78 },
];

describe("AttributionList", () => {
  it("returns null when given an empty array", () => {
    const { container } = render(<AttributionList attributions={[]} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders header with Quote icon and 'Attributions' text", () => {
    render(<AttributionList attributions={sampleAttributions} />);
    expect(screen.getByText("Attributions")).toBeInTheDocument();
  });

  it("header has uppercase tracking-wider class", () => {
    render(<AttributionList attributions={sampleAttributions} />);
    const header = screen.getByText("Attributions").closest("h4")!;
    expect(header.className).toContain("uppercase");
    expect(header.className).toContain("tracking-wider");
  });

  it("renders the correct number of attribution cards", () => {
    render(<AttributionList attributions={sampleAttributions} />);
    const sentences = sampleAttributions.map((a) => a.sentence);
    for (const sentence of sentences) {
      expect(screen.getByText(sentence)).toBeInTheDocument();
    }
  });

  it("index badge shows 1-based number (source_index + 1)", () => {
    render(<AttributionList attributions={sampleAttributions} />);
    // source_index 0 => "1", source_index 2 => "3", source_index 1 => "2"
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("displays sentence text correctly", () => {
    render(<AttributionList attributions={[sampleAttributions[0]]} />);
    expect(
      screen.getByText("The API returns JSON responses."),
    ).toBeInTheDocument();
  });

  it("shows similarity as rounded percentage", () => {
    render(<AttributionList attributions={[sampleAttributions[0]]} />);
    // 0.92 * 100 = 92 => "92%"
    expect(screen.getByText("92%")).toBeInTheDocument();
  });

  it("similarity value has font-mono class", () => {
    render(<AttributionList attributions={[sampleAttributions[0]]} />);
    const pct = screen.getByText("92%");
    expect(pct.className).toContain("font-mono");
  });
});
