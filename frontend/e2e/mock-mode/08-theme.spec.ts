import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

test.describe("Theme toggle", () => {
  test("E57: Default theme matches system preference", async ({ page }) => {
    await suppressTour(page);
    // Emulate a dark color scheme preference
    await page.emulateMedia({ colorScheme: "dark" });
    // Clear any stored theme so the app uses "system" default
    await page.addInitScript(() => {
      localStorage.removeItem("theme");
    });

    await mockApi(page);
    await page.goto("/");

    // Wait for page to settle
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // With system=dark and no stored theme, <html> should have .dark class
    const htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).toContain("dark");

    // Now test with light system preference
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => {
      localStorage.removeItem("theme");
    });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    const htmlClassLight = await page.locator("html").getAttribute("class");
    expect(htmlClassLight).not.toContain("dark");
  });

  test("E58: Toggle to dark mode adds .dark class on <html>", async ({
    page,
  }) => {
    await suppressTour(page);
    // Start in light mode explicitly
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => {
      localStorage.setItem("theme", "light");
    });

    await mockApi(page);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Verify starting in light mode (no .dark)
    let htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).not.toContain("dark");

    // The toggle button's aria-label depends on current theme
    // In light mode, it should say "Switch to dark mode"
    const toggleButton = page.getByRole("button", {
      name: /Switch to dark mode/,
    });
    await expect(toggleButton).toBeVisible();
    await toggleButton.click();

    // After toggle, <html> should have .dark class
    htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).toContain("dark");
  });

  test("E59: Toggle to light mode removes .dark class", async ({ page }) => {
    await suppressTour(page);
    // Start in dark mode explicitly
    await page.emulateMedia({ colorScheme: "dark" });
    await page.addInitScript(() => {
      localStorage.setItem("theme", "dark");
    });

    await mockApi(page);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Verify starting in dark mode
    let htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).toContain("dark");

    // In dark mode, button says "Switch to light mode"
    const toggleButton = page.getByRole("button", {
      name: /Switch to light mode/,
    });
    await expect(toggleButton).toBeVisible();
    await toggleButton.click();

    // After toggle, .dark should be removed
    htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).not.toContain("dark");
  });

  test("E60: Theme persists across page reload", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });

    await mockApi(page);
    await page.goto("/");

    // Set the initial theme to light via evaluate (NOT addInitScript,
    // which would run on every page load including reload and reset
    // the theme back before React reads it).
    await page.evaluate(() => localStorage.setItem("theme", "light"));
    await page.reload();
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Verify starting in light mode
    let htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).not.toContain("dark");

    // Toggle to dark
    await page
      .getByRole("button", { name: /Switch to dark mode/ })
      .click();

    // Verify theme is dark
    htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).toContain("dark");

    // Check localStorage was updated
    const storedTheme = await page.evaluate(() =>
      localStorage.getItem("theme"),
    );
    expect(storedTheme).toBe("dark");

    // Reload the page — the addInitScript from suppressTour still runs
    // but does NOT touch theme. The "dark" value in localStorage persists.
    await page.reload();
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // After reload, dark theme should persist
    htmlClass = await page.locator("html").getAttribute("class");
    expect(htmlClass).toContain("dark");
  });

  test("E61: Icon switches between Moon and Sun on toggle", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => {
      localStorage.setItem("theme", "light");
    });

    await mockApi(page);
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // In light mode: the Sun icon is shown, aria-label says "Switch to dark mode"
    const lightButton = page.getByRole("button", {
      name: /Switch to dark mode/,
    });
    await expect(lightButton).toBeVisible();

    // Toggle to dark
    await lightButton.click();

    // In dark mode: the Moon icon is shown, aria-label says "Switch to light mode"
    const darkButton = page.getByRole("button", {
      name: /Switch to light mode/,
    });
    await expect(darkButton).toBeVisible();

    // Toggle back to light
    await darkButton.click();

    // Should be back to "Switch to dark mode" label
    await expect(
      page.getByRole("button", { name: /Switch to dark mode/ }),
    ).toBeVisible();
  });
});
