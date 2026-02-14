import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

test.describe("Welcome screen", () => {
  test("E1: Welcome screen visible on fresh load", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    // Heading
    const heading = page.getByRole("heading", { name: "Ask your docs anything" });
    await expect(heading).toBeVisible();

    // 4 example question cards
    const cards = page.getByRole("button", { name: /^Ask: / });
    await expect(cards).toHaveCount(4);

    // Stats line
    await expect(page.getByText("42 files indexed")).toBeVisible();
    await expect(page.getByText("2,345 chunks")).toBeVisible();
  });

  test("E2: Click example question card submits and replaces welcome", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    // Welcome heading should be visible initially
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Click the first card ("How does the authentication flow work?")
    const firstCard = page.getByRole("button", {
      name: "Ask: How does the authentication flow work?",
    });
    await firstCard.click();

    // Welcome heading should disappear
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).not.toBeVisible();

    // User message should appear in the conversation log
    const conversation = page.getByRole("log", { name: "Conversation" });
    await expect(conversation).toBeVisible();

    // The user's question appears
    const userMsg = page.getByLabel("Your question");
    await expect(userMsg.first()).toContainText(
      "How does the authentication flow work?",
    );

    // Assistant response should eventually appear (stream completes)
    await expect(page.getByLabel("Assistant response")).toBeVisible();
  });

  test("E3: Type question and press Enter submits", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    const textarea = page.getByLabel("Question input");
    await textarea.fill("What is JWT?");
    await textarea.press("Enter");

    // Welcome is gone
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).not.toBeVisible();

    // User message visible
    await expect(page.getByLabel("Your question").first()).toContainText(
      "What is JWT?",
    );

    // Assistant response appears
    await expect(page.getByLabel("Assistant response")).toBeVisible();
  });

  test("E4: Type question and click Send submits", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    const textarea = page.getByLabel("Question input");
    await textarea.fill("Explain tokens");

    const sendButton = page.getByLabel("Send question");
    await expect(sendButton).toBeEnabled();
    await sendButton.click();

    // Welcome is gone
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).not.toBeVisible();

    // User message visible
    await expect(page.getByLabel("Your question").first()).toContainText(
      "Explain tokens",
    );

    // Assistant response appears
    await expect(page.getByLabel("Assistant response")).toBeVisible();
  });

  test("E5: Empty input keeps Send button disabled", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page);
    await page.goto("/");

    const sendButton = page.getByLabel("Send question");

    // Button should be disabled when textarea is empty
    await expect(sendButton).toBeDisabled();

    // Type only whitespace
    const textarea = page.getByLabel("Question input");
    await textarea.fill("   ");
    await expect(sendButton).toBeDisabled();

    // Welcome screen should still be visible (nothing submitted)
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();
  });
});
