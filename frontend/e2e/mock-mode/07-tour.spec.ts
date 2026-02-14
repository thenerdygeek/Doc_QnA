import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";

// Suppress the first-visit tour by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

// Ensure a clean first-visit state (no tour-completed flag).
async function clearTourState(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.removeItem("doc-qa-tour-completed");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

// Mark tour as already completed (returning user).
async function markTourCompleted(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

test.describe("Tour", () => {
  test("E44: First visit auto-opens settings with tour", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // On first visit, the settings dialog opens automatically with the tour
    // The dialog title should say "Setup Guide" (not "Settings") when tour is active
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();

    // The first step title "Welcome to Doc QA" should be visible
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();
  });

  test("E45: Progress bar and step counter 1/6", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Wait for tour to open
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();

    // Step counter "1/6" should be visible
    await expect(page.getByText("1/6")).toBeVisible();

    // Progress bar should exist (the outer container with the inner fill bar)
    const progressBar = page.locator(".bg-primary.h-full.rounded-full");
    await expect(progressBar).toBeVisible();
  });

  test("E46: Required badge on required steps (1, 2, 6)", async ({
    page,
  }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Step 1 (Welcome) — required
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();
    await expect(page.getByText("Required")).toBeVisible();

    // Step 2 (AI Backend) — required
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("2/6")).toBeVisible();
    await expect(page.getByText("AI Backend")).toBeVisible();
    await expect(page.getByText("Required")).toBeVisible();

    // Navigate to step 6 (You're all set!) — required
    await page.getByRole("button", { name: /Next/ }).click(); // step 3
    await page.getByRole("button", { name: /Next/ }).click(); // step 4
    await page.getByRole("button", { name: /Next/ }).click(); // step 5
    await expect(page.getByText("5/6")).toBeVisible();
    await page.getByRole("button", { name: /Next/ }).click(); // step 6
    await expect(page.getByText("6/6")).toBeVisible();
    await expect(page.getByText("You're all set!")).toBeVisible();
    await expect(page.getByText("Required")).toBeVisible();
  });

  test("E47: Optional badge on optional steps (3, 4, 5)", async ({
    page,
  }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();

    // Navigate to step 3 (Database Setup) — optional
    await page.getByRole("button", { name: /Next/ }).click(); // step 2
    await page.getByRole("button", { name: /Next/ }).click(); // step 3
    await expect(page.getByText("3/6")).toBeVisible();
    await expect(page.getByText("Database Setup")).toBeVisible();
    await expect(page.getByText("Optional")).toBeVisible();

    // Step 4 (Search Settings) — optional
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("4/6")).toBeVisible();
    await expect(page.getByText("Search Settings")).toBeVisible();
    await expect(page.getByText("Optional")).toBeVisible();

    // Step 5 (Advanced Settings) — optional
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("5/6")).toBeVisible();
    await expect(page.getByText("Advanced Settings")).toBeVisible();
    await expect(page.getByText("Optional")).toBeVisible();
  });

  test("E48: Next button advances tour step", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    await expect(page.getByText("1/6")).toBeVisible();
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();

    // Click Next
    await page.getByRole("button", { name: /Next/ }).click();

    // Should now be on step 2
    await expect(page.getByText("2/6")).toBeVisible();
    await expect(page.getByText("AI Backend")).toBeVisible();
  });

  test("E49: Back button goes to previous step", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Start at step 1 — no Back button on first step
    await expect(page.getByText("1/6")).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Back/ }),
    ).not.toBeVisible();

    // Go to step 2
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("2/6")).toBeVisible();

    // Back button should now be visible
    await page.getByRole("button", { name: /Back/ }).click();

    // Should be back on step 1
    await expect(page.getByText("1/6")).toBeVisible();
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();
  });

  test("E50: Skip button on optional steps", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Step 1 (required) — no Skip
    await expect(page.getByText("1/6")).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Skip/ }),
    ).not.toBeVisible();

    // Go to step 2 (required) — no Skip
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("2/6")).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Skip/ }),
    ).not.toBeVisible();

    // Go to step 3 (optional) — Skip should be visible
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("3/6")).toBeVisible();
    await expect(page.getByRole("button", { name: /Skip/ })).toBeVisible();

    // Click Skip — should advance to step 4
    await page.getByRole("button", { name: /Skip/ }).click();
    await expect(page.getByText("4/6")).toBeVisible();
  });

  test("E51: Tab switching disabled during tour", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Navigate to step 2 (LLM tab step) — tabs should be visible but disabled
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("2/6")).toBeVisible();

    // The LLM tab is active (tour forces it). All tab triggers should be disabled.
    const databaseTab = page.getByRole("tab", { name: "Database" });
    await expect(databaseTab).toBeDisabled();

    const retrievalTab = page.getByRole("tab", { name: "Retrieval" });
    await expect(retrievalTab).toBeDisabled();

    const intelTab = page.getByRole("tab", { name: "Intel" });
    await expect(intelTab).toBeDisabled();

    // Clicking a disabled tab should not change the active tab
    // The LLM tab content should still be showing
    await databaseTab.click({ force: true });

    // LLM tab should still be selected (the tour step controls it)
    await expect(page.getByText("AI Backend")).toBeVisible();
  });

  test("E52: Finish (Get Started) completes tour", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Navigate through all steps to the last one
    for (let i = 0; i < 5; i++) {
      await page.getByRole("button", { name: /Next/ }).click();
    }

    // Should be on step 6
    await expect(page.getByText("6/6")).toBeVisible();
    await expect(page.getByText("You're all set!")).toBeVisible();

    // Click "Get Started" button
    await page.getByRole("button", { name: /Get Started/ }).click();

    // Tour should be finished — dialog title should change to "Settings" (no longer "Setup Guide")
    // Actually, finishing the tour closes the dialog (the handleOpenChange in SettingsDialog calls tour.finish)
    // The "Setup Guide" heading should no longer be visible
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).not.toBeVisible();

    // localStorage should have tour-completed flag set
    const completed = await page.evaluate(() =>
      localStorage.getItem("doc-qa-tour-completed"),
    );
    expect(completed).toBe("true");
  });

  test("E53: Second visit — no auto-tour", async ({ page }) => {
    await markTourCompleted(page);
    await mockApi(page);
    await page.goto("/");

    // Wait for the page to settle
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Settings dialog should NOT have auto-opened
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).not.toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).not.toBeVisible();
  });

  test("E54: 'Take a Tour' link (main footer) restarts tour", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // The bottom "Take a Tour" link in the main content footer
    // There may be two "Take a Tour" buttons — one in main footer, one in settings footer
    // The main footer one is always visible when tour is not active
    const mainTourLink = page.locator("button", { hasText: "Take a Tour" }).first();
    await expect(mainTourLink).toBeVisible();
    await mainTourLink.click();

    // Tour should start — dialog opens with "Setup Guide"
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();
    await expect(page.getByText("1/6")).toBeVisible();
  });

  test("E55: 'Take a Tour' link (settings footer) restarts tour", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await page.getByRole("button", { name: "Settings" }).click();
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).toBeVisible();

    // The "Take a Tour" link inside the settings dialog footer
    const settingsTourLink = page
      .locator("[data-slot='dialog-content']")
      .locator("button", { hasText: "Take a Tour" });
    await expect(settingsTourLink).toBeVisible();
    await settingsTourLink.click();

    // Tour should restart — dialog title changes to "Setup Guide"
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();
    await expect(page.getByText("1/6")).toBeVisible();
  });

  test("E56: Non-tab steps hide tab content area", async ({ page }) => {
    await clearTourState(page);
    await mockApi(page);
    await page.goto("/");

    // Step 1 (Welcome) — non-tab step; tabs should NOT be visible
    await expect(
      page.getByRole("heading", { name: "Setup Guide" }),
    ).toBeVisible();
    await expect(page.getByText("Welcome to Doc QA")).toBeVisible();

    // TabsList should not be present for non-tab steps
    const tabsList = page.getByRole("tablist");
    await expect(tabsList).not.toBeVisible();

    // Navigate to step 2 (LLM tab step) — tabs SHOULD be visible
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("2/6")).toBeVisible();
    await expect(tabsList).toBeVisible();

    // Navigate to step 5 (Advanced Settings) — non-tab step; tabs hidden again
    await page.getByRole("button", { name: /Next/ }).click(); // step 3
    await page.getByRole("button", { name: /Next/ }).click(); // step 4
    await page.getByRole("button", { name: /Next/ }).click(); // step 5
    await expect(page.getByText("5/6")).toBeVisible();
    await expect(page.getByText("Advanced Settings")).toBeVisible();
    await expect(tabsList).not.toBeVisible();

    // Navigate to step 6 (complete) — non-tab step; tabs hidden
    await page.getByRole("button", { name: /Next/ }).click();
    await expect(page.getByText("6/6")).toBeVisible();
    await expect(page.getByText("You're all set!")).toBeVisible();
    await expect(tabsList).not.toBeVisible();
  });
});
