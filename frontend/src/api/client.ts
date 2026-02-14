import type {
  HealthResponse,
  StatsResponse,
  QueryRequest,
  QueryResponse,
  RetrievalRequest,
  RetrievalResponse,
  ConversationSummary,
  ConversationDetail,
  ConfigData,
  ConfigUpdateResponse,
  DbTestResponse,
  DbMigrateResponse,
  CodyTestResponse,
  OllamaTestResponse,
} from "@/types/api";
import type { IndexingStatusResponse } from "@/types/indexing";

const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options?: RequestInit & { timeoutMs?: number },
): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options ?? {};

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...fetchOptions,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, body.detail ?? res.statusText);
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(0, "Request timed out");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export const api = {
  health: () => request<HealthResponse>("/api/health", { timeoutMs: 5_000 }),

  stats: () => request<StatsResponse>("/api/stats"),

  query: (body: QueryRequest) =>
    request<QueryResponse>("/api/query", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 120_000,
    }),

  retrieve: (body: RetrievalRequest) =>
    request<RetrievalResponse>("/api/retrieve", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  conversations: {
    list: (limit = 50, offset = 0) =>
      request<ConversationSummary[]>(
        `/api/conversations?limit=${limit}&offset=${offset}`,
      ),

    get: (id: string) =>
      request<ConversationDetail>(`/api/conversations/${id}`),

    update: (id: string, title: string) =>
      request<{ ok: boolean }>(`/api/conversations/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      }),

    delete: (id: string) =>
      request<{ ok: boolean }>(`/api/conversations/${id}`, {
        method: "DELETE",
      }),
  },

  config: {
    get: () => request<ConfigData>("/api/config"),

    update: (body: Partial<ConfigData>) =>
      request<ConfigUpdateResponse>("/api/config", {
        method: "PATCH",
        body: JSON.stringify(body),
      }),

    dbTest: (url: string) =>
      request<DbTestResponse>("/api/config/db/test", {
        method: "POST",
        body: JSON.stringify({ url }),
      }),

    dbMigrate: () =>
      request<DbMigrateResponse>("/api/config/db/migrate", {
        method: "POST",
      }),
  },

  indexing: {
    cancel: () =>
      request<{ ok: boolean }>("/api/index/cancel", { method: "POST" }),

    status: () => request<IndexingStatusResponse>("/api/index/status"),
  },

  files: {
    open: (filePath: string) =>
      request<{ ok: boolean; error?: string }>("/api/files/open", {
        method: "POST",
        body: JSON.stringify({ file_path: filePath }),
        timeoutMs: 5_000,
      }),
  },

  browse: (path = "") =>
    request<{ path: string; parent: string; dirs: { name: string; path: string }[] }>(
      `/api/browse?path=${encodeURIComponent(path)}`,
      { timeoutMs: 5_000 },
    ),

  llm: {
    testCody: (body: { endpoint: string; access_token_env: string }) =>
      request<CodyTestResponse>("/api/llm/cody/test", {
        method: "POST",
        body: JSON.stringify(body),
        timeoutMs: 30_000,
      }),

    testOllama: (body: { host: string }) =>
      request<OllamaTestResponse>("/api/llm/ollama/test", {
        method: "POST",
        body: JSON.stringify(body),
        timeoutMs: 10_000,
      }),
  },
};
