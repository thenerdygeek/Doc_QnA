import { test, expect } from "@playwright/test";
import { mockApi, mockHealthOffline, mockConfigRestart } from "../fixtures/mock-api";
import { fullStream, errorStream } from "../fixtures/sse-streams";

// Suppress the first-visit tour by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

/** Submit a question and wait for the assistant response. */
async function submitAndWait(page: import("@playwright/test").Page, question = "How does auth work?") {
  const textarea = page.getByLabel("Question input");
  await textarea.fill(question);
  await textarea.press("Enter");
  await expect(page.getByLabel("Assistant response")).toBeVisible();
}

// ──────────────────────────────────────────────────────────────────
// V1-V3: Welcome screen
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Welcome", () => {
  test("V1: Welcome — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();
    // Let animations settle
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot("v1-welcome-desktop-light.png");
  });

  test("V2: Welcome — desktop dark", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "dark" });
    await page.addInitScript(() => localStorage.setItem("theme", "dark"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot("v2-welcome-desktop-dark.png");
  });

  test("V3: Welcome — mobile light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page, { dbEnabled: false });
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();
    await page.waitForTimeout(500);
    await expect(page).toHaveScreenshot("v3-welcome-mobile-light.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V4-V6: Streaming and complete answer
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Answer", () => {
  test("V4: Streaming in progress — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    // Use a long delay so the stream hasn't arrived yet — UI shows status indicator
    await mockApi(page, { sseDelay: 10_000 });
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    const textarea = page.getByLabel("Question input");
    await textarea.fill("How does auth work?");
    await textarea.press("Enter");

    // Wait for the stop button (streaming started) and a brief moment
    await expect(page.getByLabel("Stop generating")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v4-streaming-desktop-light.png");
  });

  test("V5: Complete answer — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await submitAndWait(page);

    // Wait for full completion (sources, attributions, badge)
    await expect(page.getByText(/Answered in/)).toBeVisible();
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot("v5-complete-answer-desktop-light.png");
  });

  test("V6: Complete answer — desktop dark", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "dark" });
    await page.addInitScript(() => localStorage.setItem("theme", "dark"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    await submitAndWait(page);

    await expect(page.getByText(/Answered in/)).toBeVisible();
    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot("v6-complete-answer-desktop-dark.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V7: Error display
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Error", () => {
  test("V7: Error display — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page, { sseBody: errorStream() });
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    const textarea = page.getByLabel("Question input");
    await textarea.fill("How does auth work?");
    await textarea.press("Enter");

    await expect(page.getByRole("alert")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v7-error-desktop-light.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V8-V9: Settings dialog
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Settings", () => {
  test("V8: Settings — Retrieval tab — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    await page.getByRole("button", { name: "Settings" }).click();
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();

    // Switch to Retrieval tab
    await page.locator("[role='tab']").filter({ hasText: "Retrieval" }).click();
    await expect(page.getByLabel("Top K")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v8-settings-retrieval-desktop-light.png");
  });

  test("V9: Settings — LLM tab with restart badge — desktop light", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await mockConfigRestart(page, ["llm"]);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    await page.getByRole("button", { name: "Settings" }).click();
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();

    // Switch to LLM tab and save to trigger restart badge
    await page.locator("[role='tab']").filter({ hasText: "LLM" }).click();
    await expect(page.getByText("Primary LLM")).toBeVisible();
    await page.getByRole("button", { name: "Save" }).click();
    await expect(page.getByText("restart required")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v9-settings-llm-restart-desktop-light.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V10-V11: Sidebar
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Sidebar", () => {
  test("V10: Sidebar with active item — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page, { dbEnabled: true });
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    const sidebar = page.locator("aside");
    await expect(sidebar.getByText("How does auth work?")).toBeVisible();

    // Click to make active
    await sidebar.getByText("How does auth work?").click();
    await expect(page.getByLabel("Your question")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v10-sidebar-active-desktop-light.png");
  });

  test("V11: Sidebar mobile overlay — mobile light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page, { dbEnabled: true });
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/");

    // Open the sidebar via hamburger menu
    const hamburger = page.getByLabel("Open sidebar");
    await expect(hamburger).toBeVisible();
    await hamburger.click();

    const sidebar = page.locator("aside");
    await expect(sidebar.getByText("How does auth work?")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v11-sidebar-mobile-light.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V12: Tour overlay
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Tour", () => {
  test("V12: Tour overlay step 2 (LLM tab) — desktop light", async ({
    page,
  }) => {
    // Do NOT suppress tour — let it auto-open
    await page.addInitScript(() => {
      localStorage.removeItem("doc-qa-tour-completed");
      sessionStorage.removeItem("doc-qa-messages");
    });
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    // Wait for tour to auto-open on first step
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();

    // Advance to step 2 (AI Backend / LLM tab)
    await page.getByRole("button", { name: "Next" }).click();
    await expect(page.getByText("AI Backend")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v12-tour-step2-desktop-light.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V13-V14: Connection status
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Connection", () => {
  test("V13: Connection status connected — desktop light", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    await expect(page.getByText("Connected")).toBeVisible();
    await page.waitForTimeout(300);

    // Screenshot just the header area for the connection badge
    const header = page.locator("header").first();
    await expect(header).toHaveScreenshot("v13-connection-connected.png");
  });

  test("V14: Connection status offline — desktop light", async ({ page }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page);
    await mockHealthOffline(page);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    await expect(page.getByText("Offline")).toBeVisible();
    await page.waitForTimeout(300);

    const header = page.locator("header").first();
    await expect(header).toHaveScreenshot("v14-connection-offline.png");
  });
});

// ──────────────────────────────────────────────────────────────────
// V15: Empty sidebar
// ──────────────────────────────────────────────────────────────────
test.describe("Visual regression: Empty sidebar", () => {
  test("V15: Empty sidebar 'No conversations yet' — desktop light", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.emulateMedia({ colorScheme: "light" });
    await page.addInitScript(() => localStorage.setItem("theme", "light"));
    await mockApi(page, { dbEnabled: true });

    // Override conversations to return empty list
    await page.route("**/api/conversations?*", (route) =>
      route.fulfill({ json: [] }),
    );
    await page.route("**/api/conversations", (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({ json: [] });
      }
      return route.continue();
    });

    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");

    const sidebar = page.locator("aside");
    await expect(sidebar.getByText("No conversations yet")).toBeVisible();
    await page.waitForTimeout(300);

    await expect(page).toHaveScreenshot("v15-empty-sidebar-desktop-light.png");
  });
});
