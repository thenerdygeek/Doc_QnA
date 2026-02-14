import { cn } from "./utils";

describe("cn", () => {
  it("merges class names correctly", () => {
    const result = cn("px-2", "py-3");
    expect(result).toContain("px-2");
    expect(result).toContain("py-3");
  });

  it("resolves Tailwind conflicts with last-wins semantics", () => {
    const result = cn("px-2", "px-4");
    expect(result).toBe("px-4");
  });

  it("handles conditional classes via falsy values", () => {
    const result = cn("base", false && "hidden");
    expect(result).toBe("base");
  });
});
