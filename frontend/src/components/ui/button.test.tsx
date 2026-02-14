import { render } from "@testing-library/react";
import { Button } from "./button";

describe("Button", () => {
  it('renders size="icon-sm" with the correct class', () => {
    const { container } = render(<Button size="icon-sm">X</Button>);
    const button = container.firstElementChild!;
    expect(button.className).toContain("size-8");
  });

  it('renders size="xs" with the correct classes', () => {
    const { container } = render(<Button size="xs">Tiny</Button>);
    const button = container.firstElementChild!;
    expect(button.className).toContain("h-6");
    expect(button.className).toContain("text-xs");
  });
});
