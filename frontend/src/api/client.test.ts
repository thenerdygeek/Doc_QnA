import { api, ApiError } from "./client";
import { vi } from "vitest";

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

function mockJsonResponse(data: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: () => Promise.resolve(data),
  };
}

describe("api.health", () => {
  it("returns health response on success", async () => {
    mockFetch.mockResolvedValue(mockJsonResponse({ status: "ok" }));

    const result = await api.health();
    expect(result).toEqual({ status: "ok" });
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/health",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  it("throws ApiError on non-ok response", async () => {
    mockFetch.mockResolvedValue(
      mockJsonResponse({ detail: "Service unavailable" }, 503),
    );

    await expect(api.health()).rejects.toThrow(ApiError);
    await expect(api.health()).rejects.toThrow("Service unavailable");
  });
});

describe("api.stats", () => {
  it("returns stats response", async () => {
    const stats = {
      total_chunks: 500,
      total_files: 25,
      db_path: "/data/db",
      embedding_model: "text-embedding-3-small",
    };
    mockFetch.mockResolvedValue(mockJsonResponse(stats));

    const result = await api.stats();
    expect(result).toEqual(stats);
  });
});

describe("api.query", () => {
  it("sends POST with question body", async () => {
    const response = {
      answer: "The answer",
      sources: [],
      chunks_retrieved: 0,
      model: "gpt-4",
      session_id: "abc",
      error: null,
      attributions: null,
      intent: "technical",
      confidence: 0.9,
      is_abstained: false,
      diagrams: null,
      detected_formats: null,
    };
    mockFetch.mockResolvedValue(mockJsonResponse(response));

    const result = await api.query({ question: "How does auth work?" });
    expect(result).toEqual(response);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/query",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ question: "How does auth work?" }),
      }),
    );
  });
});

describe("api.retrieve", () => {
  it("sends POST with retrieval request", async () => {
    const chunks = [
      {
        text: "Auth docs...",
        score: 0.9,
        file_path: "auth.md",
        section_title: "Auth",
        chunk_id: "c1",
      },
    ];
    mockFetch.mockResolvedValue(mockJsonResponse({ chunks }));

    const result = await api.retrieve({ question: "auth", top_k: 5 });
    expect(result.chunks).toEqual(chunks);
  });
});

describe("timeout handling", () => {
  it("throws on timeout", async () => {
    mockFetch.mockImplementation(
      () =>
        new Promise((_, reject) => {
          setTimeout(
            () => reject(new DOMException("Aborted", "AbortError")),
            10,
          );
        }),
    );

    await expect(api.health()).rejects.toThrow("Request timed out");
  });
});

describe("api.conversations", () => {
  it("list() returns parsed array", async () => {
    const conversations = [
      {
        id: "c1",
        user_id: null,
        title: "First chat",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ];
    mockFetch.mockResolvedValue(mockJsonResponse(conversations));

    const result = await api.conversations.list();
    expect(result).toEqual(conversations);
    expect(result).toHaveLength(1);
  });

  it("list(limit, offset) sends query params", async () => {
    mockFetch.mockResolvedValue(mockJsonResponse([]));

    await api.conversations.list(10, 20);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/conversations?limit=10&offset=20",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  it("get(id) returns ConversationDetail", async () => {
    const detail = {
      id: "c1",
      user_id: null,
      title: "Chat",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      messages: [
        {
          id: "m1",
          role: "user",
          content: "Hello",
          metadata: null,
          created_at: "2026-01-01T00:00:00Z",
        },
      ],
    };
    mockFetch.mockResolvedValue(mockJsonResponse(detail));

    const result = await api.conversations.get("c1");
    expect(result).toEqual(detail);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/conversations/c1",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  it("delete(id) sends DELETE method", async () => {
    mockFetch.mockResolvedValue(mockJsonResponse({ ok: true }));

    const result = await api.conversations.delete("c1");
    expect(result).toEqual({ ok: true });
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/conversations/c1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("update(id, title) sends PATCH", async () => {
    mockFetch.mockResolvedValue(mockJsonResponse({ ok: true }));

    const result = await api.conversations.update("c1", "New title");
    expect(result).toEqual({ ok: true });
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/conversations/c1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ title: "New title" }),
      }),
    );
  });
});

describe("api.config", () => {
  it("get() returns ConfigData", async () => {
    const config = {
      llm: { model: "gpt-4", temperature: 0.7 },
      retrieval: { top_k: 10 },
    };
    mockFetch.mockResolvedValue(mockJsonResponse(config));

    const result = await api.config.get();
    expect(result).toEqual(config);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/config",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  it("update({section: data}) returns restart info", async () => {
    const response = {
      saved: true,
      restart_required: true,
      restart_sections: ["llm"],
    };
    mockFetch.mockResolvedValue(mockJsonResponse(response));

    const result = await api.config.update({ llm: { model: "gpt-4o" } });
    expect(result).toEqual(response);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/config",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ llm: { model: "gpt-4o" } }),
      }),
    );
  });

  it("dbTest(url) returns {ok: true}", async () => {
    mockFetch.mockResolvedValue(mockJsonResponse({ ok: true }));

    const result = await api.config.dbTest("postgresql://localhost/mydb");
    expect(result).toEqual({ ok: true });
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/config/db/test",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ url: "postgresql://localhost/mydb" }),
      }),
    );
  });

  it("dbTest(url) failure returns {ok: false, error}", async () => {
    const response = { ok: false, error: "Connection refused" };
    mockFetch.mockResolvedValue(mockJsonResponse(response));

    const result = await api.config.dbTest("postgresql://bad-host/db");
    expect(result).toEqual(response);
    expect(result.ok).toBe(false);
    expect(result.error).toBe("Connection refused");
  });

  it("dbMigrate() returns {ok: true, revision}", async () => {
    const response = { ok: true, revision: "abc123" };
    mockFetch.mockResolvedValue(mockJsonResponse(response));

    const result = await api.config.dbMigrate();
    expect(result).toEqual(response);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/config/db/migrate",
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("error handling", () => {
  it("404 response throws ApiError with status === 404", async () => {
    mockFetch.mockResolvedValue(
      mockJsonResponse({ detail: "Not found" }, 404),
    );

    try {
      await api.conversations.get("nonexistent");
      expect.unreachable("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ApiError);
      expect((err as ApiError).status).toBe(404);
      expect((err as ApiError).detail).toBe("Not found");
    }
  });

  it("500 response throws ApiError", async () => {
    mockFetch.mockResolvedValue(
      mockJsonResponse({ detail: "Internal server error" }, 500),
    );

    await expect(api.config.get()).rejects.toThrow(ApiError);
    await expect(api.config.get()).rejects.toThrow("Internal server error");
  });

  it("network failure (fetch rejects) throws", async () => {
    mockFetch.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(api.stats()).rejects.toThrow(TypeError);
    await expect(api.stats()).rejects.toThrow("Failed to fetch");
  });
});
