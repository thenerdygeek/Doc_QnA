import { test, expect } from "@playwright/test";
import { mockApi, overrideSse, captureSseRequests } from "../fixtures/mock-api";
import { quickStream } from "../fixtures/sse-streams";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

/** Submit a question and wait for the assistant response to appear. */
async function submitAndWait(
  page: import("@playwright/test").Page,
  question: string,
  expectedAnswerCount: number,
) {
  const textarea = page.getByLabel("Question input");
  await textarea.fill(question);
  await textarea.press("Enter");

  // Wait until the expected number of assistant responses are visible
  await expect(page.getByLabel("Assistant response")).toHaveCount(
    expectedAnswerCount,
  );
}

test.describe("Multi-turn conversation", () => {
  test("E19: Second question preserves first Q&A pair", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page, {
      sseBody: quickStream("First answer here.", "sess-turn-1"),
    });
    await page.goto("/");

    // First question
    await submitAndWait(page, "First question?", 1);
    await expect(page.getByLabel("Your question").first()).toContainText(
      "First question?",
    );
    await expect(
      page.getByLabel("Assistant response").first(),
    ).toContainText("First answer here.");

    // Override SSE for the second question
    await overrideSse(page, quickStream("Second answer here.", "sess-turn-1"));

    // Second question
    await submitAndWait(page, "Second question?", 2);

    // Both Q&A pairs should be visible
    const userMessages = page.getByLabel("Your question");
    await expect(userMessages).toHaveCount(2);
    await expect(userMessages.first()).toContainText("First question?");
    await expect(userMessages.last()).toContainText("Second question?");

    const assistantMessages = page.getByLabel("Assistant response");
    await expect(assistantMessages).toHaveCount(2);
    await expect(assistantMessages.first()).toContainText(
      "First answer here.",
    );
    await expect(assistantMessages.last()).toContainText(
      "Second answer here.",
    );
  });

  test("E20: Session ID passed on follow-up (captureSseRequests)", async ({
    page,
  }) => {
    await suppressTour(page);
    // The first stream returns a session_id in the answer event
    const sessionId = "sess-capture-test";
    await mockApi(page, {
      sseBody: quickStream("Answer one.", sessionId),
    });
    await page.goto("/");

    // Start capturing SSE requests
    const capturedUrls = captureSseRequests(page);

    // First question
    await submitAndWait(page, "Question one?", 1);

    // Override for second question (keep same session ID)
    await overrideSse(page, quickStream("Answer two.", sessionId));

    // Second question
    await submitAndWait(page, "Question two?", 2);

    // Verify captured URLs
    expect(capturedUrls.length).toBeGreaterThanOrEqual(2);

    // First request should NOT have session_id (none was known yet)
    const firstUrl = capturedUrls[0]!;
    expect(firstUrl).toContain("q=Question");

    // Second request should include the session_id from the first answer
    const secondUrl = capturedUrls[1]!;
    expect(secondUrl).toContain(`session_id=${sessionId}`);
  });

  test("E21: Auto-scroll to bottom on new message", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page, {
      sseBody: quickStream("Short reply.", "sess-scroll"),
    });
    await page.goto("/");

    // Submit a question
    await submitAndWait(page, "Scroll test?", 1);

    // The bottom ref div should be scrolled into view. We check by verifying
    // the conversation container's last child is visible.
    const conversation = page.getByRole("log", { name: "Conversation" });
    await expect(conversation).toBeVisible();

    // The assistant response at the bottom should be in view
    const lastResponse = page.getByLabel("Assistant response").last();
    await expect(lastResponse).toBeInViewport();
  });

  test("E22: Long conversation (4+ questions) still scrolls", async ({
    page,
  }) => {
    await suppressTour(page);
    const sessionId = "sess-long";
    await mockApi(page, {
      sseBody: quickStream(
        "This is a detailed first answer with plenty of text to push content down the page.",
        sessionId,
      ),
    });
    await page.goto("/");

    // Question 1
    await submitAndWait(page, "Long conversation Q1?", 1);

    // Questions 2-4
    for (let i = 2; i <= 4; i++) {
      await overrideSse(
        page,
        quickStream(`Answer number ${i} with some content.`, sessionId),
      );
      await submitAndWait(page, `Long conversation Q${i}?`, i);
    }

    // All 4 Q&A pairs should exist
    await expect(page.getByLabel("Your question")).toHaveCount(4);
    await expect(page.getByLabel("Assistant response")).toHaveCount(4);

    // The latest response should be scrolled into view
    const lastResponse = page.getByLabel("Assistant response").last();
    await expect(lastResponse).toBeInViewport();
    await expect(lastResponse).toContainText("Answer number 4");
  });
});
