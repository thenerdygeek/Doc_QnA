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
  onEvent: (event: IndexingSSEEvent) => void;
  signal: AbortSignal;
}

export async function streamIndex({
  repoPath,
  action,
  onEvent,
  signal,
}: StreamIndexOptions): Promise<void> {
  const params = new URLSearchParams();
  if (action) params.set("action", action);
  if (repoPath) params.set("repo_path", repoPath);

  await fetchEventSource(`/api/index/stream?${params}`, {
    signal,
    openWhenHidden: true,
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
