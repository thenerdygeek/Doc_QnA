import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";
import * as responses from "../fixtures/api-responses";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

test.describe("Connection status", () => {
  test("E62: Backend healthy — green dot + 'Connected' text", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    // Wait for page to settle
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Connection status should show "Backend connected"
    const status = page.locator(
      "[role='status'][aria-label='Backend connected']",
    );
    await expect(status).toBeVisible();

    // The text "Connected" should be inside the status element
    await expect(status.getByText("Connected")).toBeVisible();

    // The green dot (emerald-500 bg) should be present
    const greenDot = status.locator("span.bg-emerald-500");
    await expect(greenDot).toBeVisible();
  });

  test("E63: Backend down — red dot + 'Offline' text", async ({ page }) => {
    await suppressTour(page);

    // Set up all standard routes via mockApi, then override health to fail.
    // Playwright routes match in reverse registration order (last registered wins),
    // so we register the failing health route AFTER mockApi.
    await mockApi(page);

    // Override health to return 503
    await page.route("**/api/health", (route) =>
      route.fulfill({ status: 503, json: { detail: "Service unavailable" } }),
    );

    await page.goto("/");

    // Connection status should show "Backend disconnected"
    const status = page.locator(
      "[role='status'][aria-label='Backend disconnected']",
    );
    await expect(status).toBeVisible({ timeout: 10000 });

    // "Offline" text should be present
    await expect(status.getByText("Offline")).toBeVisible();

    // Red dot (bg-destructive) should be present
    const redDot = status.locator("span.bg-destructive");
    await expect(redDot).toBeVisible();
  });

  test("E64: Recovery — offline then online after re-poll", async ({
    page,
  }) => {
    await suppressTour(page);

    // Install fake timers before navigation so we can fast-forward the 30s interval
    await page.clock.install();

    // Track health call count so we can switch behavior
    let healthCallCount = 0;

    // Set up all standard routes
    await page.route("**/api/stats", (route) =>
      route.fulfill({ json: responses.stats }),
    );
    await page.route("**/api/query/stream*", (route) =>
      route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: "",
      }),
    );
    await page.route("**/api/conversations?*", (route) =>
      route.fulfill({ json: responses.conversationsList }),
    );
    await page.route("**/api/conversations", (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({ json: responses.conversationsList });
      }
      return route.continue();
    });
    await page.route("**/api/conversations/*", (route) =>
      route.fulfill({ json: responses.conversationDetail }),
    );
    await page.route("**/api/config", (route) =>
      route.fulfill({ json: responses.configDataWithDb }),
    );
    await page.route("**/api/config/db/test", (route) =>
      route.fulfill({ json: responses.dbTestSuccess }),
    );
    await page.route("**/api/config/db/migrate", (route) =>
      route.fulfill({ json: responses.dbMigrateSuccess }),
    );

    // Health: first call returns 503, subsequent calls return 200
    await page.route("**/api/health", (route) => {
      healthCallCount++;
      if (healthCallCount <= 1) {
        return route.fulfill({
          status: 503,
          json: { detail: "Service unavailable" },
        });
      }
      return route.fulfill({ json: { status: "ok" } });
    });

    await page.goto("/");

    // Let the initial render and first health check complete
    await page.clock.runFor(500);

    // First: should show disconnected (first health call fails)
    const disconnectedStatus = page.locator(
      "[role='status'][aria-label='Backend disconnected']",
    );
    await expect(disconnectedStatus).toBeVisible({ timeout: 5000 });
    await expect(disconnectedStatus.getByText("Offline")).toBeVisible();

    // Fast-forward 30+ seconds to trigger the interval re-poll
    await page.clock.runFor(31_000);

    // After the re-poll, health should succeed and status should change to connected
    const connectedStatus = page.locator(
      "[role='status'][aria-label='Backend connected']",
    );
    await expect(connectedStatus).toBeVisible({ timeout: 5000 });
    await expect(connectedStatus.getByText("Connected")).toBeVisible();
  });
});
