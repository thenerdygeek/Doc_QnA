import { test, expect } from "@playwright/test";

const BACKEND = "http://localhost:8000";

test.describe("Real infrastructure (no LLM required)", () => {
  // Skip all tests if the real backend is not running.
  test.beforeAll(async ({ request }) => {
    try {
      const res = await request.get(`${BACKEND}/api/health`, { timeout: 3000 });
      if (!res.ok()) {
        test.skip(true, "Real backend not reachable");
      }
    } catch {
      test.skip(true, "Real backend not reachable");
    }
  });
  test("R1: GET /api/health returns {status: 'ok'}", async ({ request }) => {
    const res = await request.get(`${BACKEND}/api/health`);
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body).toHaveProperty("status", "ok");
  });

  test("R2: GET /api/stats returns real index statistics", async ({
    request,
  }) => {
    const res = await request.get(`${BACKEND}/api/stats`);
    expect(res.ok()).toBe(true);
    const body = await res.json();
    expect(body.total_files).toBeGreaterThan(0);
    expect(body.total_chunks).toBeGreaterThan(0);
    expect(typeof body.embedding_model).toBe("string");
  });

  test("R3: POST /api/retrieve returns real document chunks", async ({
    request,
  }) => {
    const res = await request.post(`${BACKEND}/api/retrieve`, {
      data: { question: "contributing", top_k: 3 },
    });
    expect(res.ok()).toBe(true);
    const body = await res.json();
    // Response has a "chunks" array
    expect(Array.isArray(body.chunks)).toBe(true);
    expect(body.chunks.length).toBeGreaterThan(0);
    const chunk = body.chunks[0];
    expect(typeof chunk.score).toBe("number");
    expect(typeof chunk.file_path).toBe("string");
    expect(typeof chunk.text).toBe("string");
  });

  test("R4: GET /api/config returns real configuration", async ({
    request,
  }) => {
    const res = await request.get(`${BACKEND}/api/config`);
    expect(res.ok()).toBe(true);
    const body = await res.json();
    // All major sections should be present
    expect(body).toHaveProperty("llm");
    expect(body).toHaveProperty("retrieval");
    expect(body).toHaveProperty("indexing");
    expect(body).toHaveProperty("ollama");
    expect(body).toHaveProperty("api");
  });

  test("R5: PATCH /api/config updates and returns restart info", async ({
    request,
  }) => {
    // Read current config first to restore later
    const getRes = await request.get(`${BACKEND}/api/config`);
    const currentConfig = await getRes.json();
    const originalTopK = currentConfig.retrieval?.top_k ?? 10;

    // Send a benign PATCH (change top_k, then change it back)
    const patchRes = await request.patch(`${BACKEND}/api/config`, {
      data: { retrieval: { top_k: originalTopK } },
    });
    expect(patchRes.ok()).toBe(true);
    const patchBody = await patchRes.json();
    expect(patchBody).toHaveProperty("saved", true);
    expect(patchBody).toHaveProperty("restart_required");
    expect(patchBody).toHaveProperty("restart_sections");
  });
});
