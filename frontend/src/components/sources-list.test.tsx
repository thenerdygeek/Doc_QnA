import { render, screen, fireEvent } from "@testing-library/react";
import { SourcesList } from "./sources-list";
import { vi } from "vitest";
import type { SourceInfo } from "@/types/api";

vi.mock("framer-motion", () => {
  const stub =
    (tag: string) =>
    ({ children, className, ...rest }: React.PropsWithChildren<Record<string, unknown>>) => {
      const Tag = tag as keyof JSX.IntrinsicElements;
      return <Tag className={className as string} onClick={rest.onClick as React.MouseEventHandler}>{children}</Tag>;
    };
  return {
    motion: { div: stub("div"), button: stub("button") },
    AnimatePresence: ({ children }: React.PropsWithChildren) => <>{children}</>,
    useReducedMotion: () => false,
  };
});

vi.mock("@/api/client", () => ({
  api: { files: { open: vi.fn().mockResolvedValue({ ok: true }) } },
}));

const mockSources: SourceInfo[] = [
  { file_path: "docs/auth.md", section_title: "Authentication", score: 0.92 },
  { file_path: "docs/api.md", section_title: "REST Endpoints", score: 0.75 },
  { file_path: "docs/setup.md", section_title: "Getting Started", score: 0.55 },
];

describe("SourcesList", () => {
  it("renders nothing when sources is empty", () => {
    const { container } = render(<SourcesList sources={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders source count badge", () => {
    render(<SourcesList sources={mockSources} />);
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("renders all source titles", () => {
    render(<SourcesList sources={mockSources} />);
    expect(screen.getByText("Authentication")).toBeInTheDocument();
    expect(screen.getByText("REST Endpoints")).toBeInTheDocument();
    expect(screen.getByText("Getting Started")).toBeInTheDocument();
  });

  it("renders file names", () => {
    render(<SourcesList sources={mockSources} />);
    expect(screen.getByText("auth.md")).toBeInTheDocument();
    expect(screen.getByText("api.md")).toBeInTheDocument();
    expect(screen.getByText("setup.md")).toBeInTheDocument();
  });

  it("displays relevance score as percentage", () => {
    render(<SourcesList sources={mockSources} />);
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("55%")).toBeInTheDocument();
  });

  it("calls api.files.open when a source is clicked", async () => {
    const { api } = await import("@/api/client");
    render(<SourcesList sources={mockSources} />);
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]);
    expect(api.files.open).toHaveBeenCalledWith("docs/auth.md");
  });
});

describe("SourcesList – gap coverage", () => {
  it("single source shows count badge '1'", () => {
    const single: SourceInfo[] = [
      { file_path: "docs/intro.md", section_title: "Intro", score: 0.8 },
    ];
    render(<SourcesList sources={single} />);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("score 1.0 shows '100%'", () => {
    const sources: SourceInfo[] = [
      { file_path: "docs/a.md", section_title: "A", score: 1.0 },
    ];
    render(<SourcesList sources={sources} />);
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("score 0.0 shows '0%'", () => {
    const sources: SourceInfo[] = [
      { file_path: "docs/b.md", section_title: "B", score: 0.0 },
    ];
    render(<SourcesList sources={sources} />);
    expect(screen.getByText("0%")).toBeInTheDocument();
  });
});
