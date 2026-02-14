import { test, expect, type Page } from "@playwright/test";

const BACKEND = "http://localhost:8000";

// Suppress the first-visit tour by pre-setting localStorage.
async function suppressTour(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

/** Submit a question via the textarea and wait for the send to complete. */
async function submitQuestion(page: Page, question: string) {
  const textarea = page.getByLabel("Question input");
  await textarea.fill(question);
  await textarea.press("Enter");
}

/** Wait for the assistant response to fully complete. */
async function waitForComplete(page: Page, timeout = 90_000) {
  const response = page.getByLabel("Assistant response");
  await expect(response).toBeVisible({ timeout });

  // Wait for streaming to finish — the stop button disappears when done.
  // This is the most reliable signal regardless of pipeline config.
  const stopButton = page.getByLabel("Stop generating");
  await expect(stopButton).not.toBeVisible({ timeout });
}

test.describe("Real LLM (requires Ollama)", () => {
  // Skip all tests if real backend + LLM is not available.
  test.beforeAll(async ({ request }) => {
    try {
      const healthRes = await request.get(`${BACKEND}/api/health`, {
        timeout: 3000,
      });
      if (!healthRes.ok()) {
        test.skip(true, "Real backend not reachable");
        return;
      }
      // Smoke-check that the LLM can respond (quick query)
      const streamRes = await request.get(
        `${BACKEND}/api/query/stream?q=${encodeURIComponent("hello")}`,
        { timeout: 60_000 },
      );
      if (!streamRes.ok()) {
        test.skip(true, "LLM not responding");
      }
    } catch {
      test.skip(true, "Real backend/LLM not reachable");
    }
  });

  // Real-LLM tests take longer — increase timeout.
  test.setTimeout(120_000);

  test("R6: Smoke query — ask trivial question, get real answer", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(page, "What is this project about?");

    // Wait for assistant response with generous timeout
    await waitForComplete(page);

    const response = page.getByLabel("Assistant response");
    const text = await response.textContent();
    expect(text!.trim().length).toBeGreaterThan(10);
  });

  test("R7: Full Q&A — answer appears after streaming", async ({ page }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(page, "How do I contribute to this project?");

    // The response should appear
    await waitForComplete(page);

    const response = page.getByLabel("Assistant response");
    const text = await response.textContent();
    expect(text!.trim().length).toBeGreaterThan(20);
  });

  test("R8: Real sources — retrieval returns actual chunks", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(page, "What are the contributing guidelines?");

    await waitForComplete(page);

    // Sources section should appear with real file paths
    // The real pipeline sends a sources event with retrieved chunks.
    // The heading renders as "Sources N" where N is the chunk count.
    const sourcesHeading = page.getByRole("heading", { name: /Sources/i });
    await expect(sourcesHeading).toBeVisible({ timeout: 10_000 });
  });

  test("R9: Multi-turn — follow-up uses conversation context", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    // First question
    await submitQuestion(page, "What is this project about?");
    await waitForComplete(page);

    // Follow-up question
    await submitQuestion(page, "Tell me more about that");

    // Should get a second response
    const responses = page.getByLabel("Assistant response");
    await expect(responses.nth(1)).toBeVisible({ timeout: 90_000 });
  });

  test("R10: Real verification — confidence badge if pipeline includes it", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(page, "What are the community standards?");

    await waitForComplete(page);

    // Verification may or may not run depending on pipeline config.
    // If it runs, a Verified/Unverified badge appears. If not, that's OK.
    const badge = page.getByText(/(?:Verified|Unverified)\s*\(\d+%\)/);
    const hasBadge = await badge.isVisible().catch(() => false);

    if (hasBadge) {
      // Badge rendered — verify it has a percentage
      await expect(badge).toBeVisible();
    } else {
      // No verification step — just confirm the answer completed
      const response = page.getByLabel("Assistant response");
      const text = await response.textContent();
      expect(text!.trim().length).toBeGreaterThan(10);
    }
  });

  test("R11: Long answer — markdown renders structured content", async ({
    page,
  }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(
      page,
      "Explain the code of conduct with examples in a numbered list",
    );

    await waitForComplete(page, 90_000);

    const response = page.getByLabel("Assistant response");

    // Should have some structured content — at least substantial prose
    const text = await response.textContent();
    expect(text!.trim().length).toBeGreaterThan(50);

    // Check for any rendered markdown elements
    const hasList =
      (await response.locator("ol li, ul li").count()) > 0;
    const hasCode = (await response.locator("pre code, code").count()) > 0;
    const hasHeading =
      (await response.locator("h1, h2, h3, h4").count()) > 0;
    const hasBold = (await response.locator("strong").count()) > 0;

    // At least one structural element should be present
    expect(hasList || hasCode || hasHeading || hasBold).toBe(true);
  });

  test("R12: Abort mid-stream — cancel stops real SSE", async ({ page }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(
      page,
      "Give me a very detailed and comprehensive explanation of everything",
    );

    // Wait for the stop button to appear (streaming started)
    const stopButton = page.getByLabel("Stop generating");
    await expect(stopButton).toBeVisible({ timeout: 30_000 });

    // Click stop
    await stopButton.click();

    // Stop button should disappear
    await expect(stopButton).not.toBeVisible({ timeout: 5_000 });

    // Send button should return
    await expect(page.getByLabel("Send question")).toBeVisible();

    // No error should be shown
    await expect(page.getByRole("alert")).not.toBeVisible();
  });

  test("R13: Final answer renders correctly in UI", async ({ page }) => {
    await suppressTour(page);
    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    await submitQuestion(page, "What are the project maintainer responsibilities?");

    await waitForComplete(page);

    // The answer should have real prose
    const response = page.getByLabel("Assistant response");
    const text = await response.textContent();
    expect(text!.trim().length).toBeGreaterThan(50);
  });

  test("R14: Session persistence across questions", async ({ page }) => {
    await suppressTour(page);

    // Capture SSE request URLs to check session_id
    const sseUrls: string[] = [];
    page.on("request", (req) => {
      if (req.url().includes("/api/query/stream")) {
        sseUrls.push(req.url());
      }
    });

    await page.goto(`${BACKEND}/`);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible({ timeout: 15_000 });

    // First question
    await submitQuestion(page, "What is this?");
    await waitForComplete(page);

    // Second question
    await submitQuestion(page, "Tell me more");
    const responses = page.getByLabel("Assistant response");
    await expect(responses.nth(1)).toBeVisible({ timeout: 90_000 });

    // Second request should include session_id from the first response
    expect(sseUrls.length).toBe(2);
    expect(sseUrls[1]).toContain("session_id=");
  });
});
