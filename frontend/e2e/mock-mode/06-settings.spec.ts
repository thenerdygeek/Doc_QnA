import { test, expect } from "@playwright/test";
import {
  mockApi,
  mockConfigRestart,
  mockDbTestFailure,
} from "../fixtures/mock-api";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

/** Open the settings dialog and wait for config to load. */
async function openSettings(page: import("@playwright/test").Page) {
  await page.getByRole("button", { name: "Settings" }).click();
  // Wait for the dialog title to appear (config has loaded)
  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
}

test.describe("Settings dialog", () => {
  test("E32: Settings button opens dialog", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    // Dialog should not be visible initially
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).not.toBeVisible();

    // Click settings button
    await page.getByRole("button", { name: "Settings" }).click();

    // Dialog should now be visible
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).toBeVisible();
    await expect(
      page.getByText("Configure the Doc QA system"),
    ).toBeVisible();
  });

  test("E33: All 7 tabs visible and clickable", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);

    // All 7 tab triggers should be visible. Use regex for accessible name
    // because the tabs contain SVG icons whose content may be included
    // in the computed accessible name.
    const tabNames = [
      "Database",
      "Retrieval",
      "LLM",
      "Intel",
      "Gen",
      "Verify",
      "Index",
    ];

    for (const name of tabNames) {
      const tab = page.locator("[role='tab']").filter({ hasText: name });
      await expect(tab).toBeVisible();
    }

    // Click LLM tab — should show LLM content (Primary LLM label)
    await page.locator("[role='tab']").filter({ hasText: "LLM" }).click();
    await expect(page.getByText("Primary LLM")).toBeVisible();

    // Click Retrieval tab — should show Retrieval content (Top K label)
    await page.locator("[role='tab']").filter({ hasText: "Retrieval" }).click();
    await expect(page.getByLabel("Top K")).toBeVisible();

    // Click Intel tab — should show Intelligence content
    await page.locator("[role='tab']").filter({ hasText: "Intel" }).click();
    await expect(page.getByLabel("Intent Classification")).toBeVisible();

    // Click Gen tab — should show Generation content
    await page.locator("[role='tab']").filter({ hasText: "Gen" }).click();
    await expect(page.getByLabel("Enable Diagrams")).toBeVisible();

    // Click Verify tab — should show Verification content
    await page.locator("[role='tab']").filter({ hasText: "Verify" }).click();
    await expect(page.getByLabel("Enable Verification")).toBeVisible();

    // Click Index tab — should show Indexing content
    await page.locator("[role='tab']").filter({ hasText: "Index" }).click();
    await expect(page.getByLabel("Chunk Size", { exact: true })).toBeVisible();

    // Click Database tab — should show Database content
    await page.locator("[role='tab']").filter({ hasText: "Database" }).click();
    await expect(page.getByLabel("Database URL")).toBeVisible();
  });

  test("E34: Retrieval tab — modify top_k and save sends PATCH", async ({
    page,
  }) => {
    await suppressTour(page);

    // Capture PATCH requests
    const patchBodies: unknown[] = [];
    await mockApi(page);
    await page.route("**/api/config", async (route) => {
      if (route.request().method() === "PATCH") {
        const body = route.request().postDataJSON();
        patchBodies.push(body);
        await route.fulfill({
          json: { saved: true, restart_required: false, restart_sections: [] },
        });
      } else {
        // GET — return config
        await route.fulfill({
          json: (await import("../fixtures/api-responses")).configDataWithDb,
        });
      }
    });

    await page.goto("/");
    await openSettings(page);

    // Switch to Retrieval tab
    await page.getByRole("tab", { name: "Retrieval" }).click();
    await expect(page.getByLabel("Top K")).toBeVisible();

    // Clear and type new value for Top K
    const topKInput = page.getByLabel("Top K");
    await topKInput.fill("20");

    // Click Save
    await page.getByRole("button", { name: "Save" }).click();

    // Wait for "Saved" to appear (confirms the save completed)
    await expect(page.getByRole("button", { name: "Saved" })).toBeVisible();

    // Verify a PATCH was sent with the retrieval section
    expect(patchBodies.length).toBeGreaterThan(0);
    const patchBody = patchBodies[0] as Record<string, unknown>;
    expect(patchBody).toHaveProperty("retrieval");
    expect((patchBody.retrieval as Record<string, unknown>).top_k).toBe(20);
  });

  test("E35: LLM tab — change primary to ollama and save sends PATCH", async ({
    page,
  }) => {
    await suppressTour(page);

    const patchBodies: unknown[] = [];
    await mockApi(page);
    await page.route("**/api/config", async (route) => {
      if (route.request().method() === "PATCH") {
        const body = route.request().postDataJSON();
        patchBodies.push(body);
        await route.fulfill({
          json: { saved: true, restart_required: false, restart_sections: [] },
        });
      } else {
        await route.fulfill({
          json: (await import("../fixtures/api-responses")).configDataWithDb,
        });
      }
    });

    await page.goto("/");
    await openSettings(page);

    // Switch to LLM tab
    await page.getByRole("tab", { name: "LLM" }).click();
    await expect(page.getByText("Primary LLM")).toBeVisible();

    // Change primary LLM to Ollama via the select dropdown
    // The Primary LLM select currently shows "Cody"
    const primarySelect = page.locator("div.space-y-2").filter({ hasText: "Primary LLM" }).getByRole("combobox");
    await primarySelect.click();
    await page.getByRole("option", { name: "Ollama" }).click();

    // Click Save
    await page.getByRole("button", { name: "Save" }).click();

    // Wait for save confirmation
    await expect(page.getByRole("button", { name: "Saved" })).toBeVisible();

    // Verify PATCH was sent with llm section containing primary: "ollama"
    expect(patchBodies.length).toBeGreaterThan(0);
    const llmPatch = patchBodies.find(
      (b) => (b as Record<string, unknown>).llm,
    ) as Record<string, Record<string, unknown>> | undefined;
    expect(llmPatch).toBeTruthy();
    expect(llmPatch!.llm.primary).toBe("ollama");
  });

  test("E36: DB tab — test connection success shows green indicator", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);

    // Database tab is the default tab
    await expect(page.getByLabel("Database URL")).toBeVisible();

    // Enter a URL so the Test Connection button becomes enabled
    await page.getByLabel("Database URL").fill("postgresql://user:pass@localhost:5432/docqa");

    // Click Test Connection
    await page.getByRole("button", { name: "Test Connection" }).click();

    // Green "Connected" text should appear
    await expect(page.getByText("Connected").first()).toBeVisible();
  });

  test("E37: DB tab — test connection failure shows red error", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);

    // Database tab is the default
    await expect(page.getByLabel("Database URL")).toBeVisible();

    // Enter a URL
    await page.getByLabel("Database URL").fill("postgresql://bad@localhost:5432/nope");

    // Override the DB test route to fail BEFORE clicking
    await mockDbTestFailure(page);

    // Click Test Connection
    await page.getByRole("button", { name: "Test Connection" }).click();

    // Red error text should appear
    await expect(page.getByText("Connection refused")).toBeVisible();
  });

  test("E38: DB tab — run migrations success shows revision", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);

    // Enter URL and test connection first (Run Migrations is disabled until test succeeds)
    await page.getByLabel("Database URL").fill("postgresql://user:pass@localhost:5432/docqa");
    await page.getByRole("button", { name: "Test Connection" }).click();
    await expect(page.getByText("Connected").first()).toBeVisible();

    // Now Run Migrations should be enabled
    await page.getByRole("button", { name: "Run Migrations" }).click();

    // Success message with revision should appear
    await expect(page.getByText("Migrations applied")).toBeVisible();
    await expect(page.getByText("rev abc123def")).toBeVisible();
  });

  test("E39: Restart required badge after LLM save", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    // Override config PATCH to return restart_required BEFORE opening settings
    await mockConfigRestart(page, ["llm"]);

    await openSettings(page);

    // Switch to LLM tab
    await page.getByRole("tab", { name: "LLM" }).click();

    // Click Save
    await page.getByRole("button", { name: "Save" }).click();

    // Wait for save to complete
    await expect(page.getByRole("button", { name: "Saved" })).toBeVisible();

    // Restart badge should appear
    await expect(page.getByText("restart required")).toBeVisible();
  });

  test("E40: Indexing tab — restart badge after save", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    // Override config PATCH to return restart for indexing
    await mockConfigRestart(page, ["indexing"]);

    await openSettings(page);

    // Switch to Indexing tab
    await page.locator("[role='tab']").filter({ hasText: "Index" }).click();
    await expect(page.getByLabel("Chunk Size", { exact: true })).toBeVisible();

    // Click Save
    await page.getByRole("button", { name: "Save" }).click();

    // Wait for save with a generous timeout — the PATCH response triggers
    // a state update that re-renders the button text from "Save" to "Saved".
    await expect(page.getByRole("button", { name: "Saved" })).toBeVisible({
      timeout: 5000,
    });

    // Restart badge should appear on the indexing tab content
    await expect(page.getByText("restart required")).toBeVisible({
      timeout: 5000,
    });
  });

  test("E41: Close dialog with X button", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);

    // Dialog is visible
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).toBeVisible();

    // Click the X close button (sr-only "Close" text)
    await page.getByRole("button", { name: "Close" }).click();

    // Dialog should be gone
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).not.toBeVisible();
  });

  test("E42: Close dialog with Escape key", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");
    await openSettings(page);

    // Dialog is visible
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).toBeVisible();

    // Press Escape
    await page.keyboard.press("Escape");

    // Dialog should be gone
    await expect(
      page.getByRole("heading", { name: "Settings" }),
    ).not.toBeVisible();
  });

  test("E43: Loading spinner while config fetches", async ({ page }) => {
    await suppressTour(page);

    // Set up mockApi with default routes first
    await mockApi(page);

    // Override the config GET to add a deliberate delay
    await page.route("**/api/config", async (route) => {
      if (route.request().method() === "GET") {
        // Delay 2 seconds before responding
        await new Promise((r) => setTimeout(r, 2000));
        await route.fulfill({
          json: (await import("../fixtures/api-responses")).configDataWithDb,
        });
      } else {
        await route.fulfill({
          json: { saved: true, restart_required: false, restart_sections: [] },
        });
      }
    });

    await page.goto("/");

    // Open settings — config fetch starts
    await page.getByRole("button", { name: "Settings" }).click();

    // The Loader2 spinner should be visible while loading
    // It renders as an SVG with animate-spin class inside the dialog
    const spinner = page.locator("[data-slot='dialog-content'] .animate-spin");
    await expect(spinner).toBeVisible();

    // After config loads, spinner should disappear and tabs should be visible
    await expect(page.getByRole("tab", { name: "Database" })).toBeVisible({
      timeout: 5000,
    });
    await expect(spinner).not.toBeVisible();
  });
});
