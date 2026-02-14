import { type Page, type Route } from "@playwright/test";
import * as responses from "./api-responses";
import { fullStream, indexingFullStream } from "./sse-streams";

export interface MockApiOptions {
  /** Whether DB is enabled (controls conversations endpoint). Default true. */
  dbEnabled?: boolean;
  /** Custom SSE stream body. Default: fullStream(). */
  sseBody?: string;
  /** Override health response. */
  health?: object;
  /** Override stats response. */
  stats?: object;
  /** Override config response. */
  config?: object;
  /** Delay (ms) before SSE body is sent. Default 0. */
  sseDelay?: number;
  /** Custom indexing SSE stream body. Default: indexingFullStream(). */
  indexingSseBody?: string;
}

/**
 * Install route handlers that intercept all /api/* requests and return
 * canned responses. Call this early in each test.
 */
export async function mockApi(page: Page, options: MockApiOptions = {}) {
  const {
    dbEnabled = true,
    sseBody = fullStream(),
    health = responses.health,
    stats = responses.stats,
    config = dbEnabled ? responses.configDataWithDb : responses.configData,
    sseDelay = 0,
    indexingSseBody = indexingFullStream(),
  } = options;

  // Health
  await page.route("**/api/health", (route) =>
    route.fulfill({ json: health }),
  );

  // Stats
  await page.route("**/api/stats", (route) =>
    route.fulfill({ json: stats }),
  );

  // SSE stream
  await page.route("**/api/query/stream*", async (route) => {
    if (sseDelay > 0) {
      await new Promise((r) => setTimeout(r, sseDelay));
    }
    await route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody,
    });
  });

  // Conversations list
  await page.route("**/api/conversations?*", (route) => {
    if (!dbEnabled) {
      return route.fulfill({ status: 501, json: { detail: "No database" } });
    }
    return route.fulfill({ json: responses.conversationsList });
  });

  // Conversations list (no query params)
  await page.route("**/api/conversations", (route) => {
    if (route.request().method() === "GET") {
      if (!dbEnabled) {
        return route.fulfill({ status: 501, json: { detail: "No database" } });
      }
      return route.fulfill({ json: responses.conversationsList });
    }
    return route.continue();
  });

  // Single conversation
  await page.route("**/api/conversations/*", (route) => {
    const method = route.request().method();
    if (method === "GET") {
      return route.fulfill({ json: responses.conversationDetail });
    }
    if (method === "DELETE") {
      return route.fulfill({ json: { ok: true } });
    }
    if (method === "PATCH") {
      return route.fulfill({ json: { ok: true } });
    }
    return route.continue();
  });

  // Config GET
  await page.route("**/api/config", (route) => {
    const method = route.request().method();
    if (method === "GET") {
      return route.fulfill({ json: config });
    }
    if (method === "PATCH") {
      return route.fulfill({ json: responses.configUpdateSuccess });
    }
    return route.continue();
  });

  // Config DB test
  await page.route("**/api/config/db/test", (route) =>
    route.fulfill({ json: responses.dbTestSuccess }),
  );

  // Config DB migrate
  await page.route("**/api/config/db/migrate", (route) =>
    route.fulfill({ json: responses.dbMigrateSuccess }),
  );

  // Indexing SSE stream
  await page.route("**/api/index/stream*", async (route) => {
    const url = route.request().url();
    if (url.includes("action=start")) {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: indexingSseBody,
      });
    } else {
      // Reconnect — return 204 (no active job)
      await route.fulfill({ status: 204 });
    }
  });

  // Indexing cancel
  await page.route("**/api/index/cancel", (route) =>
    route.fulfill({ json: { ok: true } }),
  );

  // Indexing status
  await page.route("**/api/index/status", (route) =>
    route.fulfill({ json: { state: "idle" } }),
  );
}

/** Helper: intercept SSE stream route with a custom body mid-test. */
export async function overrideSse(page: Page, sseBody: string) {
  await page.route("**/api/query/stream*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: sseBody,
    }),
  );
}

/** Helper: make the health endpoint fail. */
export async function mockHealthOffline(page: Page) {
  await page.route("**/api/health", (route) =>
    route.fulfill({ status: 503, json: { detail: "Service unavailable" } }),
  );
}

/** Helper: make config PATCH return restart_sections. */
export async function mockConfigRestart(
  page: Page,
  sections: string[] = ["llm"],
) {
  await page.route("**/api/config", (route) => {
    if (route.request().method() === "PATCH") {
      return route.fulfill({
        json: {
          saved: true,
          restart_required: true,
          restart_sections: sections,
        },
      });
    }
    return route.fulfill({ json: responses.configDataWithDb });
  });
}

/** Helper: make DB test fail. */
export async function mockDbTestFailure(page: Page) {
  await page.route("**/api/config/db/test", (route) =>
    route.fulfill({ json: responses.dbTestFailure }),
  );
}

/**
 * Capture SSE requests for assertion (e.g., checking session_id param).
 * Returns an array that is pushed to on each matching request.
 */
export function captureSseRequests(page: Page): string[] {
  const urls: string[] = [];
  page.on("request", (req) => {
    if (req.url().includes("/api/query/stream")) {
      urls.push(req.url());
    }
  });
  return urls;
}
