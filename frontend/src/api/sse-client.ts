import { fetchEventSource } from "@microsoft/fetch-event-source";
import type { SSEEvent } from "@/types/sse";
import type { IndexingSSEEvent } from "@/types/indexing";

interface StreamQueryOptions {
  question: string;
  sessionId?: string;
  onEvent: (event: SSEEvent) => void;
  signal: AbortSignal;
}

export async function streamQuery({
  question,
  sessionId,
  onEvent,
  signal,
}: StreamQueryOptions): Promise<void> {
  const params = new URLSearchParams({ q: question });
  if (sessionId) params.set("session_id", sessionId);

  await fetchEventSource(`/api/query/stream?${params}`, {
    signal,
    openWhenHidden: true,
    onmessage(msg) {
      // Skip keepalive pings (empty event or empty data)
      if (!msg.event || !msg.data) return;

      try {
        const data = JSON.parse(msg.data);
        onEvent({ event: msg.event, data } as SSEEvent);
      } catch {
        // Ignore malformed JSON
      }
    },
    onerror(err) {
      // Don't retry — let the user re-submit
      throw err;
    },
  });
}

// ── Indexing SSE stream ─────────────────────────────────────────

interface StreamIndexOptions {
  repoPath?: string;
  action?: "start";
  forceReindex?: boolean;
  onEvent: (event: IndexingSSEEvent) => void;
  signal: AbortSignal;
}

export async function streamIndex({
  repoPath,
  action,
  forceReindex,
  onEvent,
  signal,
}: StreamIndexOptions): Promise<void> {
  const params = new URLSearchParams();
  if (action) params.set("action", action);
  if (repoPath) params.set("repo_path", repoPath);
  if (forceReindex) params.set("force_reindex", "true");

  await fetchEventSource(`/api/index/stream?${params}`, {
    signal,
    openWhenHidden: true,
    async onopen(response) {
      // Surface HTTP error status codes in the error message so
      // callers (e.g. use-indexing) can detect 409, 400, etc.
      // Without this, fetchEventSource throws a generic content-type
      // mismatch error that hides the actual HTTP status.
      if (response.ok) return;

      let detail = "";
      try {
        const body = await response.json();
        detail = body.detail || JSON.stringify(body);
      } catch {
        try {
          detail = await response.text();
        } catch {
          detail = response.statusText;
        }
      }
      throw new Error(`${response.status}: ${detail}`);
    },
    onmessage(msg) {
      if (!msg.event || !msg.data) return;

      try {
        const data = JSON.parse(msg.data);
        onEvent({ event: msg.event, data } as IndexingSSEEvent);
      } catch {
        // Ignore malformed JSON
      }
    },
    onerror(err) {
      throw err;
    },
  });
}
