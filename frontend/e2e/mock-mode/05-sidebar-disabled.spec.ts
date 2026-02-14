import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

test.describe("Sidebar disabled (DB unavailable)", () => {
  test("E30: Sidebar completely hidden when DB returns 501", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: false });
    await page.goto("/");

    // Wait for the page to settle (health check, conversations 501 response)
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // The sidebar (aside element) should not be present at all
    const sidebar = page.locator("aside");
    await expect(sidebar).toHaveCount(0);

    // The mobile hamburger button ("Open sidebar") should also be absent
    const menuButton = page.getByRole("button", { name: "Open sidebar" });
    await expect(menuButton).toHaveCount(0);
  });

  test("E31: Full-width main content area (no sidebar gap)", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: false });
    await page.goto("/");

    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // The main content wrapper is the direct child div with flex-1
    // When sidebar is hidden, the main content div should stretch to full viewport width
    const mainContent = page.locator("div.flex.h-screen > div.flex-1");
    const mainBox = await mainContent.boundingBox();
    const viewport = page.viewportSize()!;

    expect(mainBox).not.toBeNull();
    // Main content width should equal viewport width (no sidebar taking space)
    expect(mainBox!.width).toBe(viewport.width);
    expect(mainBox!.x).toBe(0);
  });
});
