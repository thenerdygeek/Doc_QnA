import { test, expect } from "@playwright/test";
import { mockApi } from "../fixtures/mock-api";
import { quickStream } from "../fixtures/sse-streams";

// Suppress the first-visit tour/settings dialog by pre-setting localStorage.
async function suppressTour(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    localStorage.setItem("doc-qa-tour-completed", "true");
    sessionStorage.removeItem("doc-qa-messages");
  });
}

test.describe("Conversation sidebar", () => {
  test("E23: Sidebar shows conversation list (with DB enabled)", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: true });
    await page.goto("/");

    // Sidebar should be visible (on desktop widths)
    const sidebar = page.locator("aside");
    await expect(sidebar).toBeVisible();

    // "History" header
    await expect(sidebar.getByText("History")).toBeVisible();

    // Conversation titles from canned data (conv-1, conv-2, conv-3)
    await expect(sidebar.getByText("How does auth work?")).toBeVisible();
    await expect(sidebar.getByText("REST API examples")).toBeVisible();
    // conv-3 has empty title, rendered as "Untitled"
    await expect(sidebar.getByText("Untitled")).toBeVisible();
  });

  test("E24: Click conversation loads its history (messages replaced)", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: true });
    await page.goto("/");

    // Click on the first conversation in the sidebar
    const sidebar = page.locator("aside");
    await sidebar.getByText("How does auth work?").click();

    // The conversation detail is loaded via GET /api/conversations/conv-1
    // which returns conversationDetail with 2 messages.
    // Welcome screen should be gone since messages are loaded.
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).not.toBeVisible();

    // Conversation log should show the loaded messages
    const conversation = page.getByRole("log", { name: "Conversation" });
    await expect(conversation).toBeVisible();

    // User message from the conversation detail
    await expect(page.getByLabel("Your question").first()).toContainText(
      "How does authentication work?",
    );

    // Assistant message from the conversation detail
    await expect(
      page.getByLabel("Assistant response").first(),
    ).toContainText("Authentication uses JWT tokens");
  });

  test("E25: New Chat button clears everything and shows welcome", async ({
    page,
  }) => {
    await suppressTour(page);
    await mockApi(page, {
      dbEnabled: true,
      sseBody: quickStream("Quick answer.", "sess-new-chat"),
    });
    await page.goto("/");

    // Submit a question to move away from welcome
    const textarea = page.getByLabel("Question input");
    await textarea.fill("Some question?");
    await textarea.press("Enter");
    await expect(page.getByLabel("Assistant response")).toBeVisible();

    // Welcome should be gone
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).not.toBeVisible();

    // Click "New conversation" button (in the header, visible when not empty)
    const newChatButton = page.getByLabel("New conversation").first();
    await newChatButton.click();

    // Welcome screen should reappear
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // Messages should be gone
    await expect(page.getByLabel("Your question")).toHaveCount(0);
    await expect(page.getByLabel("Assistant response")).toHaveCount(0);
  });

  test("E26: Delete conversation removes from list", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: true });
    await page.goto("/");

    const sidebar = page.locator("aside");

    // Initially 3 conversations
    await expect(sidebar.getByText("How does auth work?")).toBeVisible();

    // Track DELETE requests
    const deleteRequests: string[] = [];
    page.on("request", (req) => {
      if (
        req.url().includes("/api/conversations/") &&
        req.method() === "DELETE"
      ) {
        deleteRequests.push(req.url());
      }
    });

    // Hover over the first conversation to reveal the delete button
    const firstConvButton = sidebar
      .getByText("How does auth work?")
      .locator("..");
    await firstConvButton.hover();

    // Click the delete button (aria-label="Delete conversation")
    const deleteButton = sidebar.getByLabel("Delete conversation").first();
    await deleteButton.click();

    // The conversation should be removed from the list
    await expect(sidebar.getByText("How does auth work?")).not.toBeVisible();

    // Verify a DELETE request was sent
    expect(deleteRequests.length).toBe(1);
    expect(deleteRequests[0]).toContain("conv-1");
  });

  test("E27: Active conversation highlighted", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: true });
    await page.goto("/");

    const sidebar = page.locator("aside");

    // Click on the first conversation
    const firstConv = sidebar.getByText("How does auth work?");
    await firstConv.click();

    // Wait for the conversation to load
    await expect(page.getByLabel("Your question")).toHaveCount(1);

    // The motion.button that contains the active conversation text should
    // have "bg-accent" as a standalone class (not "hover:bg-accent/50").
    // Use a filter locator to find the correct button ancestor.
    const activeButton = sidebar.locator("button").filter({ hasText: "How does auth work?" });
    await expect(activeButton).toHaveClass(/(?:^|\s)bg-accent(?:\s|$)/);

    // Other conversations should NOT have the standalone "bg-accent" class
    // (they may have "hover:bg-accent/50" which is fine).
    const secondConvButton = sidebar.locator("button").filter({ hasText: "REST API examples" });
    await expect(secondConvButton).not.toHaveClass(/(?:^|\s)bg-accent(?:\s|$)/);
  });

  test("E28: Empty sidebar shows 'No conversations yet'", async ({
    page,
  }) => {
    await suppressTour(page);

    // Override the conversations list to return empty array
    await mockApi(page, { dbEnabled: true });

    // Override conversations routes to return empty list
    await page.route("**/api/conversations?*", (route) =>
      route.fulfill({ json: [] }),
    );
    await page.route("**/api/conversations", (route) => {
      if (route.request().method() === "GET") {
        return route.fulfill({ json: [] });
      }
      return route.continue();
    });

    await page.goto("/");

    const sidebar = page.locator("aside");
    await expect(sidebar).toBeVisible();
    await expect(sidebar.getByText("No conversations yet")).toBeVisible();
  });

  test("E29: Deleting active conversation clears chat", async ({ page }) => {
    await suppressTour(page);
    await mockApi(page, { dbEnabled: true });
    await page.goto("/");

    const sidebar = page.locator("aside");

    // Load a conversation by clicking it
    await sidebar.getByText("How does auth work?").click();

    // Wait for messages to load
    await expect(page.getByLabel("Your question")).toHaveCount(1);
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).not.toBeVisible();

    // Now delete that active conversation
    const firstConvButton = sidebar
      .getByText("How does auth work?")
      .locator("..");
    await firstConvButton.hover();

    const deleteButton = sidebar.getByLabel("Delete conversation").first();
    await deleteButton.click();

    // Since we deleted the active conversation, chat should clear to welcome
    await expect(
      page.getByRole("heading", { name: "Ask your docs anything" }),
    ).toBeVisible();

    // No messages should remain
    await expect(page.getByLabel("Your question")).toHaveCount(0);
    await expect(page.getByLabel("Assistant response")).toHaveCount(0);
  });
});
