import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";
import {
  indexingFullStream,
  indexingCancelledStream,
  indexingErrorStream,
} from "../fixtures/sse-streams";

async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

async function openSettings(page: import("@playwright/test").Page) {
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
}

async function goToIndexTab(page: import("@playwright/test").Page) {
  await page.locator("[role='tab']").filter({ hasText: "Index" }).click();
  await expect(
    page.getByLabel("Documentation Repository Path"),
  ).toBeVisible();
}

test.describe("Indexing tab", () => {
  test("E50: Shows repo path input and Start Indexing button", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);
    await goToIndexTab(page);

    await expect(page.getByPlaceholder("/path/to/docs")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Start Indexing" }),
    ).toBeVisible();
    // Start button disabled when no path
    await expect(
      page.getByRole("button", { name: "Start Indexing" }),
    ).toBeDisabled();
  });

  test("E51: Start Indexing shows progress bar and file log", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page, { indexingSseBody: indexingFullStream() });
    await page.goto("/");
    await openSettings(page);
    await goToIndexTab(page);

    // Enter a repo path
    await page.getByPlaceholder("/path/to/docs").fill("/docs/repo");

    // Start indexing
    await page.getByRole("button", { name: "Start Indexing" }).click();

    // Should show "Indexing complete" after the full stream finishes
    await expect(page.getByText("Indexing complete")).toBeVisible({
      timeout: 5000,
    });

    // Stats should show
    await expect(page.getByText("3 files")).toBeVisible();
    await expect(page.getByText("12 chunks")).toBeVisible();
  });

  test("E52: Cancel button appears during indexing", async ({ page }) => {
    await suppressTour(page);
    // Use a stream that takes longer by having the SSE delay
    await mockApi(page, {
      indexingSseBody: indexingCancelledStream(),
    });
    await page.goto("/");
    await openSettings(page);
    await goToIndexTab(page);

    await page.getByPlaceholder("/path/to/docs").fill("/docs/repo");
    await page.getByRole("button", { name: "Start Indexing" }).click();

    // Should show cancelled state
    await expect(page.getByText("Indexing cancelled")).toBeVisible({
      timeout: 5000,
    });
    await expect(
      page.getByText("Previous index preserved"),
    ).toBeVisible();
  });

  test("E53: Error during indexing shows error banner", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page, { indexingSseBody: indexingErrorStream() });
    await page.goto("/");
    await openSettings(page);
    await goToIndexTab(page);

    await page.getByPlaceholder("/path/to/docs").fill("/docs/repo");
    await page.getByRole("button", { name: "Start Indexing" }).click();

    // Should show error state
    await expect(page.getByText("Indexing failed")).toBeVisible({
      timeout: 5000,
    });
    await expect(
      page.getByText("Permission denied: /docs/repo"),
    ).toBeVisible();
  });

  test("E54: Chunk settings still visible below indexing controls", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);
    await goToIndexTab(page);

    // Chunk settings should be visible
    await expect(
      page.getByLabel("Chunk Size", { exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Chunk Overlap")).toBeVisible();
    await expect(page.getByLabel("Min Chunk Size")).toBeVisible();
    await expect(page.getByLabel("Embedding Model")).toBeVisible();
  });

  test("E55: Repo path input disabled during indexing", async ({ page }) => {
    await suppressTour(page);
    // Use a delayed SSE response to keep it in "running" state
    await mockApi(page);

    // Override index stream to delay
    await page.route("**/api/index/stream*", async (route) => {
      const url = route.request().url();
      if (url.includes("action=start")) {
        // Send just the scanning status, then hang (never complete)
        const partialStream = [
          `event: status\ndata: ${JSON.stringify({ state: "scanning", repo_path: "/docs/repo" })}\n\n`,
          `event: progress\ndata: ${JSON.stringify({ state: "scanning", processed: 0, total_files: 10, total_chunks: 0, percent: 0 })}\n\n`,
        ].join("");
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          body: partialStream,
        });
      } else {
        await route.fulfill({ status: 204 });
      }
    });

    await page.goto("/");
    await openSettings(page);
    await goToIndexTab(page);

    await page.getByPlaceholder("/path/to/docs").fill("/docs/repo");
    await page.getByRole("button", { name: "Start Indexing" }).click();

    // Wait for scanning state text
    await expect(page.getByText("Scanning files")).toBeVisible({ timeout: 5000 });

    // Input should be disabled during indexing
    await expect(page.getByPlaceholder("/path/to/docs")).toBeDisabled();

    // Cancel button should be visible
    await expect(
      page.getByRole("button", { name: "Cancel" }),
    ).toBeVisible();
  });
});
