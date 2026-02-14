import { vi } from "vitest";
import type { SSEEvent } from "@/types/sse";
import type { IndexingSSEEvent } from "@/types/indexing";

// Capture the options passed to fetchEventSource so we can invoke callbacks directly
let capturedUrl = "";
let capturedOptions: Record<string, unknown> = {};

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: vi.fn(async (url: string, opts: Record<string, unknown>) => {
    capturedUrl = url;
    capturedOptions = opts;
  }),
}));

import { streamQuery, streamIndex } from "./sse-client";
import { fetchEventSource } from "@microsoft/fetch-event-source";

beforeEach(() => {
  vi.clearAllMocks();
  capturedUrl = "";
  capturedOptions = {};
});

describe("streamQuery", () => {
  const defaults = () => ({
    question: "What is RAG?",
    onEvent: vi.fn(),
    signal: new AbortController().signal,
  });

  it("calls fetchEventSource with URL containing q= param", async () => {
    await streamQuery(defaults());

    expect(fetchEventSource).toHaveBeenCalledTimes(1);
    expect(capturedUrl).toContain("/api/query/stream?");
    expect(capturedUrl).toContain("q=What+is+RAG%3F");
  });

  it("includes session_id param when provided", async () => {
    await streamQuery({ ...defaults(), sessionId: "sess-42" });

    expect(capturedUrl).toContain("session_id=sess-42");
  });

  it("omits session_id from URL when not provided", async () => {
    await streamQuery(defaults());

    expect(capturedUrl).not.toContain("session_id");
  });

  it("parses valid JSON event data and fires onEvent", async () => {
    const onEvent = vi.fn();
    await streamQuery({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    onmessage({
      event: "status",
      data: JSON.stringify({ status: "classifying" }),
    });

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith({
      event: "status",
      data: { status: "classifying" },
    } satisfies SSEEvent);
  });

  it("handles multiple events in sequence, in order", async () => {
    const onEvent = vi.fn();
    await streamQuery({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    onmessage({
      event: "status",
      data: JSON.stringify({ status: "retrieving" }),
    });
    onmessage({
      event: "answer_token",
      data: JSON.stringify({ token: "Hello" }),
    });
    onmessage({
      event: "done",
      data: JSON.stringify({ status: "complete", elapsed: 1.5 }),
    });

    expect(onEvent).toHaveBeenCalledTimes(3);
    expect(onEvent.mock.calls[0][0].event).toBe("status");
    expect(onEvent.mock.calls[1][0].event).toBe("answer_token");
    expect(onEvent.mock.calls[2][0].event).toBe("done");
  });

  it("skips keepalive ping with empty msg.data", async () => {
    const onEvent = vi.fn();
    await streamQuery({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    onmessage({ event: "status", data: "" });

    expect(onEvent).not.toHaveBeenCalled();
  });

  it("skips event with empty msg.event", async () => {
    const onEvent = vi.fn();
    await streamQuery({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    onmessage({ event: "", data: JSON.stringify({ status: "classifying" }) });

    expect(onEvent).not.toHaveBeenCalled();
  });

  it("ignores malformed JSON gracefully (no throw)", async () => {
    const onEvent = vi.fn();
    await streamQuery({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    expect(() => {
      onmessage({ event: "status", data: "not-valid-json{{{" });
    }).not.toThrow();

    expect(onEvent).not.toHaveBeenCalled();
  });

  it("passes AbortSignal to fetchEventSource", async () => {
    const controller = new AbortController();
    await streamQuery({ ...defaults(), signal: controller.signal });

    expect(capturedOptions.signal).toBe(controller.signal);
  });

  it("sets openWhenHidden: true", async () => {
    await streamQuery(defaults());

    expect(capturedOptions.openWhenHidden).toBe(true);
  });

  it("URL-encodes question with special characters", async () => {
    await streamQuery({
      ...defaults(),
      question: "what is A & B? #test",
    });

    // URLSearchParams encodes & as %26, # as %23, spaces as +
    expect(capturedUrl).toContain("q=what+is+A+%26+B%3F+%23test");
  });

  it("onerror throws error (no retry)", async () => {
    await streamQuery(defaults());

    const onerror = capturedOptions.onerror as (err: Error) => void;
    const error = new Error("connection lost");

    expect(() => onerror(error)).toThrow("connection lost");
  });
});

describe("streamIndex", () => {
  const defaults = () => ({
    onEvent: vi.fn(),
    signal: new AbortController().signal,
  });

  it("calls fetchEventSource with /api/index/stream URL", async () => {
    await streamIndex(defaults());

    expect(fetchEventSource).toHaveBeenCalledTimes(1);
    expect(capturedUrl).toContain("/api/index/stream?");
  });

  it("includes action and repo_path params when provided", async () => {
    await streamIndex({
      ...defaults(),
      action: "start",
      repoPath: "/home/user/my-repo",
    });

    expect(capturedUrl).toContain("action=start");
    expect(capturedUrl).toContain("repo_path=%2Fhome%2Fuser%2Fmy-repo");
  });

  it("omits action and repo_path when not provided", async () => {
    await streamIndex(defaults());

    expect(capturedUrl).not.toContain("action=");
    expect(capturedUrl).not.toContain("repo_path=");
  });

  it("parses valid JSON and fires onEvent with IndexingSSEEvent", async () => {
    const onEvent = vi.fn();
    await streamIndex({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    onmessage({
      event: "status",
      data: JSON.stringify({ state: "scanning", repo_path: "/tmp/repo" }),
    });

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith({
      event: "status",
      data: { state: "scanning", repo_path: "/tmp/repo" },
    } satisfies IndexingSSEEvent);
  });

  it("skips keepalive ping with empty data", async () => {
    const onEvent = vi.fn();
    await streamIndex({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    onmessage({ event: "progress", data: "" });

    expect(onEvent).not.toHaveBeenCalled();
  });

  it("ignores malformed JSON gracefully", async () => {
    const onEvent = vi.fn();
    await streamIndex({ ...defaults(), onEvent });

    const onmessage = capturedOptions.onmessage as (msg: {
      event: string;
      data: string;
    }) => void;

    expect(() => {
      onmessage({ event: "status", data: "not-valid-json{{{" });
    }).not.toThrow();

    expect(onEvent).not.toHaveBeenCalled();
  });

  it("passes AbortSignal to fetchEventSource", async () => {
    const controller = new AbortController();
    await streamIndex({ ...defaults(), signal: controller.signal });

    expect(capturedOptions.signal).toBe(controller.signal);
  });

  it("onerror throws error (no retry)", async () => {
    await streamIndex(defaults());

    const onerror = capturedOptions.onerror as (err: Error) => void;
    const error = new Error("stream failed");

    expect(() => onerror(error)).toThrow("stream failed");
  });
});
