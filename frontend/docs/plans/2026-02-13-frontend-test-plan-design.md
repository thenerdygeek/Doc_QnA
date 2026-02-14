# Doc QA Frontend — Comprehensive Test Plan

**Date:** 2026-02-13
**Scope:** Unit tests (Vitest), E2E mock mode (Playwright), E2E real mode (Playwright + Ollama), visual regression screenshots
**Target:** 317 tests + 15 visual baselines

---

## Table of Contents

1. [Architecture & Tools](#1-architecture--tools)
2. [Tier 1 — Unit Tests (Vitest)](#2-tier-1--unit-tests-vitest)
   - [2A. SSE Client](#2a-apisse-clientts)
   - [2B. API Client — Gap Coverage](#2b-apiclientts--gap-coverage)
   - [2C. useStreamingQuery — Gap Coverage](#2c-hooksuse-streaming-queryts--gap-coverage)
   - [2D. useSettings](#2d-hooksuse-settingsts)
   - [2E. useConversations](#2e-hooksuse-conversationsts)
   - [2F. useTour](#2f-hooksuse-tourts)
   - [2G. useTheme — Gap Coverage](#2g-hooksuse-themets--gap-coverage)
   - [2H. useSession — Gap Coverage](#2h-hooksuse-sessionts--gap-coverage)
   - [2I. App.tsx](#2i-apptsx)
   - [2J. message-list](#2j-componentsmessage-listtsx)
   - [2K. streaming-answer](#2k-componentsstreaming-answertsx)
   - [2L. welcome-screen](#2l-componentswelcome-screentsx)
   - [2M. conversation-sidebar](#2m-componentsconversation-sidebartsx)
   - [2N. settings-dialog](#2n-componentssettings-dialogtsx)
   - [2O. attribution-list](#2o-componentsattribution-listtsx)
   - [2P. connection-status](#2p-componentsconnection-statustsx)
   - [2Q. theme-toggle](#2q-componentstheme-toggletsx)
   - [2R. error-boundary](#2r-componentserror-boundarytsx)
   - [2S. Existing Test Gaps](#2s-existing-test-gaps)
   - [2T. Utilities & Primitives](#2t-utilities--primitives)
3. [Tier 2 — E2E Mock Mode (Playwright)](#3-tier-2--e2e-mock-mode-playwright)
4. [Tier 3 — E2E Real Mode (Playwright + Ollama)](#4-tier-3--e2e-real-mode-playwright--ollama)
5. [Visual Regression Strategy](#5-visual-regression-strategy)
6. [Test Counts Summary](#6-test-counts-summary)
7. [Execution Plan](#7-execution-plan)

---

## 1. Architecture & Tools

### Framework Choices

| Layer | Tool | Rationale |
|-------|------|-----------|
| Unit tests | **Vitest** + Testing Library | Already set up, 58 tests passing. Fast, deterministic. |
| E2E tests | **Playwright** | Available via MCP, excellent SSE support, network interception for mock mode, screenshot APIs for visual regression. |
| Visual regression | **Playwright screenshots** | DOM assertions in Vitest for CI speed, Playwright screenshot snapshots for visual baseline. |

### Test Modes

| Mode | Backend | Speed | Use Case |
|------|---------|-------|----------|
| Mock mode (CI) | Playwright network interception, canned SSE responses | Fast (~30s) | Every PR, deterministic |
| Real mode (integration) | Real FastAPI + Ollama (qwen2.5:0.5b) | Slow (~2-5min) | Manual validation, nightly |

### LLM Availability

- **Cody:** Not available on dev machine
- **Ollama:** Installed (v0.13.5), not running by default
- **Strategy:** Real-mode tests gated by pre-flight check. If Ollama not running, tests skip gracefully.
- **FallbackBackend** means only one LLM needed — if Ollama responds, tests pass regardless of Cody state.

### Pre-flight Check (Real Mode)

```
Backend up? (GET /api/health)
  ├─ No  → skip ALL real-mode, log reason
  └─ Yes → Smoke query (POST /api/query, trivial question)
              ├─ answer.error != null → skip real-LLM tests, run infra-only tests
              └─ answer.error == null → run full real-mode suite
```

Real-mode splits into two sub-tiers:

| Sub-tier | Needs LLM? | Tests |
|----------|-----------|-------|
| real-infra | No | Health, stats, retrieval, config, conversations |
| real-llm | Yes | Full Q&A, streaming, verification, attribution |

### Directory Structure

```
frontend/
├── src/**/*.test.{ts,tsx}          # Unit tests (Vitest)
├── e2e/
│   ├── fixtures/                    # Canned API/SSE responses
│   ├── helpers/                     # Shared utilities (mock setup, SSE emitter)
│   ├── mock-mode/                   # Tests with intercepted network
│   ├── real-mode/                   # Tests against live backend
│   ├── screenshots/                 # Visual regression baselines
│   └── global-setup.ts             # Pre-flight LLM check
├── playwright.config.ts
└── vite.config.ts                   # Vitest config (existing)
```

### Fixture Files

| File | Content |
|------|---------|
| `health.json` | `{"status": "ok"}` |
| `stats.json` | `{"total_chunks": 1200, "total_files": 45, "db_path": "...", "embedding_model": "..."}` |
| `config.json` | Full config object with all 9 sections |
| `conversations.json` | Array of 3 conversation summaries |
| `conversation-detail.json` | Single conversation with 4 messages |
| `query-response.json` | Full QueryResponse with sources, attributions, verification |
| `sse-full-stream.txt` | Raw SSE event sequence (all 9 event types, complete happy path) |
| `sse-error-stream.txt` | SSE stream that ends with error event mid-generation |
| `sse-no-sources.txt` | SSE stream with empty sources (abstain scenario) |
| `sse-no-verification.txt` | SSE stream without verified event (verification disabled) |
| `db-disabled-501.json` | 501 Not Implemented response for conversations |

---

## 2. Tier 1 — Unit Tests (Vitest)

### 2A. `api/sse-client.ts`

**Currently:** 0% tested (CRITICAL)

**What it does:** Wraps `@microsoft/fetch-event-source` for SSE streaming. Sends question + optional sessionId as query params. Parses JSON event data. Skips keepalive pings. No auto-retry.

**How it should behave:**
- Calls `fetchEventSource` with URL `/api/query/stream?q={question}&session_id={id}`
- Parses each SSE event's JSON `data` field and calls `onEvent({event, data})`
- Silently skips keepalive pings (empty `msg.event` or empty `msg.data`)
- Silently ignores malformed JSON (no crash)
- Passes `AbortSignal` through for cancellation
- Sets `openWhenHidden: true` so streams survive tab-hide
- On `onerror`, throws the error (no retry)

| # | Test Case | Assert |
|---|-----------|--------|
| 1 | Calls fetchEventSource with correct URL containing `q=` param | URL verified |
| 2 | Includes `session_id` param when provided | URL contains `session_id=abc` |
| 3 | Omits `session_id` from URL when not provided | No `session_id` in URL |
| 4 | Parses valid JSON event data and fires onEvent | `onEvent({event:"status", data:{status:"retrieving"}})` called |
| 5 | Handles multiple events in sequence, in order | onEvent called N times, order preserved |
| 6 | Skips keepalive ping — empty `msg.data` | onEvent NOT called |
| 7 | Skips event with empty `msg.event` | onEvent NOT called |
| 8 | Ignores malformed JSON gracefully (no throw) | No error, onEvent not called for bad event |
| 9 | Passes AbortSignal to fetchEventSource | `signal` option matches provided signal |
| 10 | Sets `openWhenHidden: true` | Option verified |
| 11 | URL-encodes question with special characters (`&`, `?`, `#`, spaces) | Encoded via URLSearchParams |
| 12 | `onerror` throws error (no retry) | Error propagated to caller |

---

### 2B. `api/client.ts` — Gap Coverage

**Currently:** 11 tests (health, stats, query, retrieve, timeout). Missing: conversations.*, config.*, error variations.

| # | Test Case | Assert |
|---|-----------|--------|
| 13 | `conversations.list()` — GET /api/conversations | Returns parsed array |
| 14 | `conversations.list(limit, offset)` — sends query params | URL has `?limit=10&offset=5` |
| 15 | `conversations.get(id)` — GET /api/conversations/{id} | Returns ConversationDetail with messages array |
| 16 | `conversations.delete(id)` — sends DELETE method | Method=DELETE, URL correct |
| 17 | `conversations.update(id, {title})` — sends PATCH | Method=PATCH, body has title |
| 18 | `config.get()` — GET /api/config | Returns ConfigData object |
| 19 | `config.update({section: data})` — PATCH /api/config | Returns `{saved, restart_required, restart_sections}` |
| 20 | `config.dbTest(url)` — POST /api/config/db/test | Returns `{ok: true}` |
| 21 | `config.dbTest(url)` failure — returns error | Returns `{ok: false, error: "..."}` |
| 22 | `config.dbMigrate()` — POST /api/config/db/migrate | Returns `{ok: true, revision: "abc"}` |
| 23 | 404 response throws ApiError with `status === 404` | Error class + status verified |
| 24 | 500 response throws ApiError with `status === 500` | Error class + status verified |
| 25 | Network failure (fetch rejects) throws | Error propagated |

---

### 2C. `hooks/use-streaming-query.ts` — Gap Coverage

**Currently:** 11 tests. Missing: concurrent submit, session/diagram extraction, reset completeness.

| # | Test Case | Assert |
|---|-----------|--------|
| 26 | Submit while already streaming aborts previous stream | First AbortController.abort() called |
| 27 | Session ID extracted from `answer` event | `sessionId` matches event data |
| 28 | Diagrams extracted from `answer` event | `diagrams` array populated correctly |
| 29 | Intent + confidence from `intent` event | Both fields set |
| 30 | Attribution event populates `attributions` array | Array matches payload |
| 31 | Multiple `answer_token` events concatenate in order | `tokens === "Hello world"` |
| 32 | Error event sets `phase="error"` + error message | Both fields correct |
| 33 | Cancel while idle is no-op (no crash) | No state change |
| 34 | Reset clears ALL accumulated state to initial values | Every field back to default |
| 35 | Stream error (non-abort) sets error phase with message | `phase === "error"`, `error` has message |

---

### 2D. `hooks/use-settings.ts`

**Currently:** 0% tested.

**How it should behave:**
- Returns `config: null` + `loading: false` before dialog opens
- Lazy-fetches config only when `open` transitions to `true` (not on mount)
- `fetched` ref prevents re-fetch on subsequent opens
- `updateSection()` sends PATCH, refetches GET, accumulates `restart_sections` via Set deduplication
- `testDbConnection()` clears previous result, calls POST, stores new result
- `runMigrations()` clears previous result, calls POST, stores new result
- `saving` flag true during `updateSection` in-flight

| # | Test Case | Assert |
|---|-----------|--------|
| 36 | Config is null before dialog opens | `config === null`, `loading === false` |
| 37 | Opening dialog (`setOpen(true)`) triggers GET /api/config | fetch called |
| 38 | `loading` is true during fetch, false after | Flag transitions correctly |
| 39 | Config populated after fetch resolves | `config` has section keys |
| 40 | Second `setOpen(true)` doesn't re-fetch (hasFetched ref) | fetch called only once total |
| 41 | `updateSection("retrieval", data)` sends PATCH with `{retrieval: data}` | Request body correct |
| 42 | After updateSection, config re-fetched via GET | GET called again |
| 43 | `restart_sections` accumulated across multiple saves via Set | No duplicates, array grows |
| 44 | `saving` is true during updateSection, false after | Flag transitions |
| 45 | `testDbConnection(url)` calls POST /api/config/db/test with url | Endpoint + body correct |
| 46 | DB test success: `dbTestResult` set to `{ok: true}` | State updated |
| 47 | DB test failure: `dbTestResult` set to `{ok: false, error: "..."}` | Error message present |
| 48 | `runMigrations()` calls POST /api/config/db/migrate | Endpoint hit |
| 49 | Migrate result stored in `migrateResult` | State updated |
| 50 | Previous `dbTestResult` cleared before new test | Null then result |
| 51 | Config fetch failure doesn't crash (console.error, config stays null) | No throw |

---

### 2E. `hooks/use-conversations.ts`

**Currently:** 0% tested.

**How it should behave:**
- Fetches conversation list on mount via GET /api/conversations
- `hasFetched` ref prevents double-fetch on StrictMode re-mount
- 501 response → `dbEnabled: false`, conversations stay empty
- Successful response → `dbEnabled: true`, conversations populated
- `deleteConversation(id)` removes from local list optimistically + calls DELETE API
- `refresh()` re-fetches full list (resets `hasFetched`)
- Delete failure: console.error, list unchanged

| # | Test Case | Assert |
|---|-----------|--------|
| 52 | Fetches conversations on mount | GET /api/conversations called |
| 53 | Returns parsed conversation array | `conversations.length > 0` |
| 54 | `loading` true during fetch, false after | Flag transitions |
| 55 | 501 response sets `dbEnabled: false` | Flag set, conversations empty |
| 56 | Successful response sets `dbEnabled: true` | Flag set |
| 57 | `deleteConversation(id)` removes from local list | Array shrinks by 1 |
| 58 | `deleteConversation(id)` calls DELETE /api/conversations/{id} | Correct endpoint |
| 59 | Delete API failure: list unchanged, console.error | No crash, array same |
| 60 | `refresh()` re-fetches full list | GET called again |
| 61 | Double-mount doesn't double-fetch (hasFetched ref) | Single GET call |

---

### 2F. `hooks/use-tour.ts`

**Currently:** 0% tested.

**How it should behave:**
- 6 steps defined in `TOUR_STEPS`: welcome(null), llm(llm), database(database), retrieval(retrieval), advanced(null), complete(null)
- Reads `TOUR_COMPLETED_KEY` from localStorage on init
- Auto-starts (`active: true`, `currentStep: 0`) on first visit (localStorage key absent)
- Return visit (key present): `active: false`, `isFirstVisit: false`
- `next()` increments step, returns false at end
- `back()` decrements step, clamps at 0
- `skip()` same as `next()` (advances one step)
- `finish()` sets `active: false`, resets to step 0, writes localStorage
- `start()` sets `active: true`, resets to step 0

| # | Test Case | Assert |
|---|-----------|--------|
| 62 | First visit: `isFirstVisit === true`, tour auto-starts | `active: true`, `currentStep: 0` |
| 63 | Return visit (localStorage set): tour doesn't auto-start | `active: false`, `isFirstVisit: false` |
| 64 | `next()` advances step index by 1 | `currentStep` increments |
| 65 | `next()` at last step (index 5) returns false | Returns false, step unchanged |
| 66 | `back()` decrements step index | `currentStep` decrements |
| 67 | `back()` at step 0 stays at 0 (clamp) | `currentStep === 0` |
| 68 | `skip()` advances step (same as next) | `currentStep` increments |
| 69 | `finish()` deactivates, resets to 0, writes localStorage | `active: false`, storage set |
| 70 | `start()` reactivates tour from step 0 | `active: true`, `currentStep: 0` |
| 71 | Each step has correct `tab` reference | welcome→null, llm→"llm", database→"database", etc. |
| 72 | `totalSteps === 6` | Constant correct |
| 73 | Step object matches TOUR_STEPS definition | title, description, required fields correct |

---

### 2G. `hooks/use-theme.ts` — Gap Coverage

**Currently:** 5 tests. Missing: system preference detection, invalid values.

| # | Test Case | Assert |
|---|-----------|--------|
| 74 | System preference "dark" via matchMedia resolves correctly | `resolvedTheme === "dark"` |
| 75 | System preference change listener fires and updates theme | Theme auto-updates |
| 76 | Invalid localStorage value falls back to "system" | No crash, default applied |

---

### 2H. `hooks/use-session.ts` — Gap Coverage

**Currently:** 4 tests. Missing: storage unavailable.

| # | Test Case | Assert |
|---|-----------|--------|
| 77 | sessionStorage unavailable (throws) doesn't crash | Graceful fallback |

---

### 2I. `App.tsx`

**Currently:** 0% tested (CRITICAL — primary orchestration component, 273 lines).

**What it renders:**
- Full-screen flex layout: optional sidebar (left) + main content (right)
- Header: logo ("Doc QA"), New Chat button (when messages exist), Settings button, ConnectionStatus, ThemeToggle
- Main area: WelcomeScreen (empty state) OR MessageList (conversation state)
- Footer: ChatInput + "Take a Tour" link (when tour inactive)
- SettingsDialog overlay

**How it should behave:**
- Messages persisted to/restored from sessionStorage (`MESSAGES_KEY`)
- Corrupt sessionStorage silently falls back to `[]`
- Assistant messages with empty content filtered out on restore
- `submitQuestion` adds user msg + empty assistant msg pair
- `handleRetry` removes last msg, adds new assistant msg, re-submits
- `handleNewChat` resets stream + messages + session
- Session ID synced from stream.sessionId to useSession (deduped via ref)
- Final answer content persisted back into message for sessionStorage survival
- Sidebar refreshes after stream completes (if `dbEnabled`)
- `handleSelectConversation` loads from API, resets stream, sets messages
- `handleDeleteConversation` clears chat if deleting active conversation
- `displayMessages` injects `streaming` property into last assistant msg
- New Chat button conditionally shown (only when `!isEmpty`)
- Sidebar conditionally shown (only when `dbEnabled === true`)
- "Take a Tour" link hidden during active tour
- Tour auto-opens settings on first visit

| # | Test Case | Source Lines | Assert |
|---|-----------|-------------|--------|
| 78 | Messages restored from sessionStorage on mount | 21-34 | Messages array populated |
| 79 | Corrupt sessionStorage JSON doesn't crash | 22-33 | Falls back to `[]` |
| 80 | Empty content assistant messages filtered on restore | 26-28 | Only user + non-empty assistant kept |
| 81 | `submitQuestion` adds user + empty assistant pair | 67-78 | messages.length grows by 2 |
| 82 | `submitQuestion` calls `stream.submit` with question + sessionId | 74 | Hook called correctly |
| 83 | `handleRetry` removes last message, adds new assistant | 80-91 | Last msg replaced |
| 84 | `handleRetry` calls `stream.submit` with original question | 87 | Re-submits same q |
| 85 | `handleNewChat` resets stream, messages, and session | 93-98 | All three cleared |
| 86 | Session ID synced from stream to useSession | 101-106 | setSessionId called |
| 87 | Session ID not re-set if same value (ref guard) | 102 | setSessionId NOT called twice |
| 88 | Final answer persisted back to last assistant message | 109-119 | msg.content updated |
| 89 | Sidebar refreshes after stream completes (dbEnabled=true) | 122-127 | refresh() called |
| 90 | No sidebar refresh when dbEnabled=false | 123 | refresh() NOT called |
| 91 | `handleSelectConversation` loads messages from API | 129-148 | Messages replaced, stream reset |
| 92 | `handleSelectConversation` API failure: console.error, no crash | 142-144 | Try/catch works |
| 93 | `handleDeleteConversation` on active conversation clears chat | 150-159 | handleNewChat called |
| 94 | `handleDeleteConversation` on non-active: no clear | 154 | handleNewChat NOT called |
| 95 | `displayMessages` injects streaming state into last assistant msg | 161-170 | `streaming` prop present |
| 96 | Non-streaming assistant msgs don't get streaming prop | 169 | `streaming` undefined |
| 97 | New Chat button hidden when isEmpty | 209 | Not in DOM |
| 98 | New Chat button visible when messages exist | 209 | In DOM |
| 99 | Sidebar hidden when `dbEnabled !== true` | 173, 178 | Not rendered |
| 100 | Sidebar shown when `dbEnabled === true` | 173 | Rendered |
| 101 | "Take a Tour" link hidden when tour active | 247 | Not in DOM |
| 102 | "Take a Tour" click starts tour + opens settings | 251-254 | Both actions fire |
| 103 | Tour auto-opens settings on first visit | 51-56 | settings.setOpen(true) called |
| 104 | Messages persisted to sessionStorage on change | 59-65 | sessionStorage.setItem called |
| 105 | Empty messages removes sessionStorage key | 60-61 | sessionStorage.removeItem called |

---

### 2J. `components/message-list.tsx`

**Currently:** 0% tested (CRITICAL).

**What it renders:**
- ScrollArea container with `role="log"`, `aria-label="Conversation"`, `aria-live="polite"`
- User messages: right-aligned (`flex justify-end`), gradient bg (from-primary to-primary/80), rounded-2xl rounded-br-sm, white text
- Assistant messages: left-aligned, robot icon avatar (Sparkles, hidden on mobile via `hidden sm:flex`), response area
- For streaming assistant (last msg): StatusIndicator, then either ErrorDisplay (error phase) OR StreamingAnswer
- Sources shown when `streaming.sources.length > 0`
- Attributions shown when `streaming.attributions.length > 0`
- ConfidenceBadge shown when `streaming.verification` exists
- Elapsed time "Answered in X.Xs" shown when phase=complete and elapsed != null
- Auto-scrolls to bottom ref on `messages.length` change AND `latestTokens` change
- onRetry only available when `i > 0` (not for first assistant message)

| # | Test Case | Assert |
|---|-----------|--------|
| 106 | Renders user messages with `justify-end` class | Right-aligned |
| 107 | User message shows question text in `<p>` | Text content matches |
| 108 | User message has `aria-label="Your question"` | ARIA present |
| 109 | Assistant message has robot avatar icon (Sparkles) | Icon rendered |
| 110 | Avatar hidden on mobile (`hidden sm:flex` class) | Class present |
| 111 | Assistant message has `aria-label="Assistant response"` | ARIA present |
| 112 | Streaming msg: StatusIndicator rendered | Component visible |
| 113 | Streaming msg: StreamingAnswer rendered with tokens | Tokens passed |
| 114 | Error phase: ErrorDisplay rendered INSTEAD OF StreamingAnswer | ErrorDisplay visible, StreamingAnswer absent |
| 115 | Error phase: onRetry uses previous user message content | Callback arg = messages[i-1].content |
| 116 | onRetry not provided for first assistant msg (i=0 guard) | onRetry undefined |
| 117 | Sources shown when `streaming.sources.length > 0` | SourcesList rendered |
| 118 | Sources hidden when empty array | SourcesList not rendered |
| 119 | Attributions shown when `streaming.attributions.length > 0` | AttributionList rendered |
| 120 | ConfidenceBadge shown when verification exists | Badge rendered |
| 121 | Elapsed time "Answered in X.Xs" shown on complete | Text present, formatted to 1 decimal |
| 122 | Elapsed time hidden when phase != complete | Text absent |
| 123 | Empty messages array renders no messages (just scroll anchor) | Container has only ref div |
| 124 | Multiple messages render in DOM order matching array | Order verified |
| 125 | Auto-scroll ref div exists at bottom | Div with ref present |
| 126 | Container has `role="log"` and `aria-live="polite"` | ARIA attributes present |
| 127 | Non-streaming assistant uses `msg.content` for answer | Content displayed |

---

### 2K. `components/streaming-answer.tsx`

**Currently:** 0% tested.

**What it renders:**
- Streamdown component with plugins (code, mermaid) inside `div.answer-prose`
- During streaming: displays `tokens` prop with `isAnimating={true}`
- After streaming: displays `finalAnswer` prop with `isAnimating={false}`
- Copy buttons injected into `<pre>` elements after streaming ends (useEffect)
- Copy button: `className="copy-btn"`, `aria-label="Copy code"`, text "Copy"
- Click: clipboard write → "Copied!" + `.copied` class → reset after 2s
- Clipboard failure: "Failed" → reset after 2s
- Cleanup: removes `.copy-btn` elements on unmount/re-render
- Returns null if no content

| # | Test Case | Assert |
|---|-----------|--------|
| 128 | Renders tokens while streaming (`isStreaming=true`) | Streamdown receives tokens, isAnimating=true |
| 129 | Renders finalAnswer when not streaming | Streamdown receives finalAnswer |
| 130 | Returns null when both tokens empty and finalAnswer null/empty | Nothing rendered |
| 131 | Content precedence: streaming uses tokens, complete uses finalAnswer | Correct content selected |
| 132 | Copy buttons injected into `<pre>` blocks after streaming ends | `.copy-btn` elements present |
| 133 | Copy buttons NOT injected while still streaming | No `.copy-btn` |
| 134 | Duplicate copy buttons prevented (`.copy-btn` already exists check) | Single button per pre |
| 135 | Copy click: `navigator.clipboard.writeText` called with code text | Clipboard API called |
| 136 | Copy success: button text changes to "Copied!", `.copied` class added | Text + class updated |
| 137 | Copy success: resets to "Copy" after 2 seconds | setTimeout revert |
| 138 | Copy failure: button text changes to "Failed" | Error text shown |
| 139 | Copy failure: resets to "Copy" after 2 seconds | setTimeout revert |
| 140 | Multiple pre blocks each get independent copy buttons | Each has own button |
| 141 | Cleanup removes copy buttons on effect re-run | `.copy-btn` removed |
| 142 | Copy button has `aria-label="Copy code"` | ARIA present |

---

### 2L. `components/welcome-screen.tsx`

**Currently:** 0% tested.

**What it renders:**
- Centered flex container with decorative glow orb (`aria-hidden="true"`)
- `<h2>` heading: "Ask your docs anything" with `gradient-text` class
- Subtitle paragraph with description text
- Stats line (conditional): Database icon + "X files indexed" + separator + "Y chunks"
- 4 example question buttons in `grid sm:grid-cols-2`
- Each button: icon div + label `<p>` + question preview `<p>` with `line-clamp-2`
- Buttons have `aria-label="Ask: {question}"`
- Stats fetched on mount via `api.stats()`, mounted guard prevents state update on unmount
- Stats fetch failure silently ignored

| # | Test Case | Assert |
|---|-----------|--------|
| 143 | Renders heading "Ask your docs anything" | Text present in h2 |
| 144 | Renders subtitle description text | Paragraph present |
| 145 | Fetches stats on mount via `api.stats()` | GET /api/stats called |
| 146 | Shows stats when available: "X files indexed" with Database icon | Text + icon present |
| 147 | Shows chunk count with locale formatting | `toLocaleString()` applied |
| 148 | Stats fetch failure doesn't crash (silent catch) | Heading still visible, no error |
| 149 | Unmounted component doesn't setState (mounted guard) | No React warning |
| 150 | Renders exactly 4 example question buttons | 4 buttons in DOM |
| 151 | Clicking example button calls `onSelectQuestion` with question text | Callback with correct string |
| 152 | Each button has icon + label + question preview | All 3 elements present |
| 153 | Question preview has `line-clamp-2` class | Truncation class present |
| 154 | Each button has `aria-label="Ask: {question}"` | ARIA correct |
| 155 | Decorative orb div has `aria-hidden="true"` | Hidden from AT |

---

### 2M. `components/conversation-sidebar.tsx`

**Currently:** 0% tested.

**What it renders:**
- Mobile backdrop: fixed overlay, `bg-black/40`, `md:hidden`, click calls `onClose`
- Aside panel: 280px width, fixed on mobile (z-50, slide transition), relative on desktop (md:relative, md:translate-x-0)
- Hidden when `open=false` via `-translate-x-full` (but desktop always visible: `md:translate-x-0`)
- Header: "History" text + New Chat button (`aria-label="New conversation"`) + Close button (`aria-label="Close sidebar"`, `md:hidden`)
- Conversation list in ScrollArea, each as `<motion.button>`:
  - Title with truncate (falls back to "Untitled" for empty title)
  - Relative timestamp via `formatTimeAgo()`
  - Active conversation: `bg-accent text-accent-foreground`
  - Inactive: `hover:bg-accent/50`
  - Delete button: `role="button"`, `tabIndex={0}`, `aria-label="Delete conversation"`, opacity-0 → group-hover:opacity-100
  - Delete click stops propagation (no parent onSelect)
  - Delete keyboard: Enter/Space handling
- Empty state: "No conversations yet" paragraph

| # | Test Case | Assert |
|---|-----------|--------|
| 156 | Renders conversation titles | Text matches data |
| 157 | Empty title falls back to "Untitled" | "Untitled" displayed |
| 158 | Active conversation has highlight class (`bg-accent`) | CSS class present |
| 159 | Inactive conversation has hover class (`hover:bg-accent/50`) | CSS class present |
| 160 | Click conversation calls `onSelect(conv.id)` | Correct ID passed |
| 161 | Click New Chat button calls `onNew` | Callback fired |
| 162 | Delete button calls `onDelete(conv.id)` on click | Correct ID passed |
| 163 | Delete click stops propagation (onSelect NOT called) | Only onDelete fires |
| 164 | Delete keyboard: Enter key calls onDelete | Keyboard accessible |
| 165 | Delete keyboard: Space key calls onDelete | Keyboard accessible |
| 166 | Empty list shows "No conversations yet" | Text visible |
| 167 | Close button calls `onClose` | Callback fired |
| 168 | Close button has `md:hidden` class (mobile only) | Class present |
| 169 | `formatTimeAgo`: < 1 min → "just now" | Correct string |
| 170 | `formatTimeAgo`: 5 min → "5m ago" | Correct string |
| 171 | `formatTimeAgo`: 2 hours → "2h ago" | Correct string |
| 172 | `formatTimeAgo`: 3 days → "3d ago" | Correct string |
| 173 | `formatTimeAgo`: 8+ days → locale date string | Formatted date |
| 174 | Sidebar hidden when `open=false` (`-translate-x-full` class) | Transform class present |
| 175 | Mobile backdrop visible when open | Overlay element in DOM |
| 176 | Backdrop click calls `onClose` | Callback fired |
| 177 | Sidebar has semantic `<aside>` element | aside tag |
| 178 | Delete button has `aria-label="Delete conversation"` | ARIA present |

---

### 2N. `components/settings-dialog.tsx`

**Currently:** 0% tested. 771 lines, 8 internal sub-components.

#### 2N-1. Helper functions

| # | Test Case | Assert |
|---|-----------|--------|
| 179 | `field()` returns value from config | Correct value extracted |
| 180 | `field()` returns fallback when config is null | Fallback returned |
| 181 | `field()` returns fallback when section missing | Fallback returned |
| 182 | `field()` returns fallback when key missing | Fallback returned |
| 183 | `field()` returns fallback when value is null/undefined | Fallback returned |

#### 2N-2. RestartBadge sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 184 | Renders yellow AlertTriangle icon | Icon present |
| 185 | Renders "restart required" text | Text present |
| 186 | Has yellow-500 border styling | Class present |

#### 2N-3. SaveButton sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 187 | Shows "Save" text by default | Text present |
| 188 | Shows "Saved" text after save | Text changes |
| 189 | Disabled when `saving=true` | Button disabled |
| 190 | Shows Loader2 spinner when saving | Spinner visible |
| 191 | Calls onClick on click | Callback fired |

#### 2N-4. TourOverlay sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 192 | Progress bar width = `((currentStep + 1) / totalSteps) * 100%` | Width correct |
| 193 | Step counter shows "N/total" | Text correct (e.g., "2/6") |
| 194 | Step title and description displayed | Text matches step data |
| 195 | Required badge shown for required steps | "Required" badge present |
| 196 | Optional badge shown for optional steps | "Optional" badge present |
| 197 | Rocket icon on first step | Rocket icon present |
| 198 | PartyPopper icon on last step | PartyPopper icon present |
| 199 | Back button hidden on first step | Not in DOM |
| 200 | Back button visible on non-first steps | Button present |
| 201 | Skip button hidden on required steps | Not in DOM |
| 202 | Skip button hidden on last step | Not in DOM |
| 203 | Skip button visible on optional, non-last steps | Button present |
| 204 | "Get Started" button shown on last step | Text = "Get Started" |
| 205 | "Next" button shown on non-last steps | Text = "Next" |
| 206 | Next click calls `next()` | Callback fired |
| 207 | Back click calls `back()` | Callback fired |
| 208 | Skip click calls `skip()` | Callback fired |
| 209 | Finish click calls `finish()` | Callback fired |

#### 2N-5. DatabaseTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 210 | URL input synced from config on load | Value matches config |
| 211 | URL input editable | onChange updates state |
| 212 | Test button disabled when URL empty | Button disabled |
| 213 | Test button disabled while testing (spinner shown) | Disabled + Loader2 |
| 214 | Test success: green "Connected" text with CheckCircle | Green text + icon |
| 215 | Test failure: red error message with XCircle | Red text + error + icon |
| 216 | Migrate button disabled until test passes (`!dbTestResult?.ok`) | Button disabled |
| 217 | Migrate success: green text with revision code | Revision shown |
| 218 | Migrate failure: red error text | Error shown |
| 219 | Save sends `{url: null}` when input is empty string | Correct payload |
| 220 | Save calls `updateSection("database", {url})` | Correct section + data |
| 221 | "Saved" feedback shown for 2 seconds after save | Timeout behavior |

#### 2N-6. RetrievalTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 222 | All 6 form fields sync from config | Values match config |
| 223 | Search mode select has 3 options: hybrid, vector, fts | Options present |
| 224 | Top K number input with min=1, max=50 | Attributes correct |
| 225 | Candidate Pool input with min=1, max=200 | Attributes correct |
| 226 | Min Score input with step=0.05, min=0, max=1 | Attributes correct |
| 227 | Max Chunks/File input with min=1, max=20 | Attributes correct |
| 228 | Rerank switch toggles on/off | Checked state changes |
| 229 | Save calls `updateSection("retrieval", form)` | Correct payload |

#### 2N-7. LLMTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 230 | Primary select has cody/ollama options | Options present |
| 231 | Fallback select has cody/ollama options | Options present |
| 232 | Cody fieldset: model + endpoint inputs | Both inputs present |
| 233 | Ollama fieldset: host + model inputs | Both inputs present |
| 234 | All fields sync from config on load | Values match |
| 235 | RestartBadge shown when restartRequired includes llm/cody/ollama | Badge visible |
| 236 | RestartBadge hidden when no restart needed | Badge absent |
| 237 | Save calls updateSection 3 times: llm, cody, ollama | 3 PATCH calls |

#### 2N-8. IntelligenceTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 238 | Intent classification switch toggles | State changes |
| 239 | Multi-intent switch toggles | State changes |
| 240 | High/Medium confidence inputs present | Inputs rendered |
| 241 | Max sub-queries input with w-24 class | Width class present |
| 242 | Save calls `updateSection("intelligence", form)` | Correct payload |

#### 2N-9. GenerationTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 243 | Enable diagrams switch toggles | State changes |
| 244 | Mermaid validation select has 4 options: auto, node, regex, none | Options present |
| 245 | Max diagram retries input present | Input rendered |
| 246 | Save calls `updateSection("generation", form)` | Correct payload |

#### 2N-10. VerificationTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 247 | Enable verification switch toggles | State changes |
| 248 | Enable CRAG switch toggles | State changes |
| 249 | Confidence threshold + Max CRAG rewrites inputs present | Inputs rendered |
| 250 | Abstain on low confidence switch toggles | State changes |
| 251 | Save calls `updateSection("verification", form)` | Correct payload |

#### 2N-11. IndexingTab sub-component

| # | Test Case | Assert |
|---|-----------|--------|
| 252 | Chunk size/overlap/min inputs present with correct min/max | Attributes correct |
| 253 | Embedding model text input present | Input rendered |
| 254 | RestartBadge shown when restartRequired includes "indexing" | Badge visible |
| 255 | Restart warning text displayed | Text present |
| 256 | Save calls `updateSection("indexing", form)` | Correct payload |

#### 2N-12. SettingsDialog main component

| # | Test Case | Assert |
|---|-----------|--------|
| 257 | Dialog renders when `open=true` | Dialog in DOM |
| 258 | Dialog hidden when `open=false` | Not in DOM |
| 259 | Title is "Settings" when tour inactive | Correct title |
| 260 | Title is "Setup Guide" when tour active | Title changes |
| 261 | Description changes during tour | Text changes |
| 262 | All 7 tab triggers render with icons | Triggers present |
| 263 | Clicking tab switches content | Content changes |
| 264 | Tab switching disabled during tour | Click ignored |
| 265 | Tour step auto-switches to correct tab | `activeTab` matches `step.tab` |
| 266 | TourOverlay shown when tour active + step has tab | Overlay present |
| 267 | Non-tab tour steps: tabs hidden, overlay shown directly | Tabs absent, overlay present |
| 268 | `tabBadge()` shows yellow dot for restart-required tabs | Dot visible |
| 269 | Loading spinner while config fetches | Loader2 + centering present |
| 270 | Closing dialog during tour calls `tour.finish()` | finish() invoked |
| 271 | `wrappedSettings` calls `onDbSaved` on database section save | Callback fired |
| 272 | "Take a Tour" footer button visible when tour inactive | Button present |
| 273 | "Take a Tour" footer button hidden when tour active | Button absent |
| 274 | "Take a Tour" click calls `tour.start()` | start() invoked |

---

### 2O. `components/attribution-list.tsx`

**Currently:** 0% tested.

**What it renders:**
- Returns null if `attributions.length === 0`
- Header: Quote icon + `<h4>` "Attributions" (uppercase, `tracking-wider`)
- Cards: motion.div with stagger animation (delay: i * 0.04)
  - Index badge: `source_index + 1` (1-based), circle with `bg-primary/10`
  - Sentence text: `<p>` with `text-foreground/80`
  - Similarity: `Math.round(attr.similarity * 100)%` in mono font

| # | Test Case | Assert |
|---|-----------|--------|
| 275 | Returns null when empty array | Nothing in DOM |
| 276 | Renders header with Quote icon + "Attributions" text | Both present |
| 277 | Header has uppercase tracking-wider class | CSS classes present |
| 278 | Renders correct number of attribution cards | Count matches input |
| 279 | Index badge shows 1-based number (source_index + 1) | "1" for index 0, "3" for index 2 |
| 280 | Sentence text displayed correctly | Text matches data |
| 281 | Similarity shown as rounded percentage | "92%" for 0.92, "88%" for 0.876 |
| 282 | Similarity in mono font class | `font-mono` class present |

---

### 2P. `components/connection-status.tsx`

**Currently:** 0% tested.

**What it renders:**
- Returns null while status is "checking" (initial state)
- Connected: animated ping (green, `animate-ping`) + solid green dot (`bg-emerald-500`) + "Connected" text
- Disconnected: static red dot (`bg-destructive`) + "Offline" text (red, `text-destructive`)
- Text hidden on mobile: `hidden sm:inline` class
- Container: `role="status"`, `aria-label="Backend {status}"`
- Polls `api.health()` on mount then every 30 seconds
- Mounted guard prevents setState on unmounted component
- Cleanup clears interval

| # | Test Case | Assert |
|---|-----------|--------|
| 283 | Returns null while status is "checking" | Nothing rendered |
| 284 | Shows "Connected" + green dot when health succeeds | Text + emerald class present |
| 285 | Shows animated ping on connected (`animate-ping` class) | Class present |
| 286 | Shows "Offline" + red dot when health fails | Text + destructive class present |
| 287 | No animated ping on disconnected | `animate-ping` absent |
| 288 | `role="status"` on container | ARIA present |
| 289 | `aria-label="Backend connected"` when connected | Label correct |
| 290 | `aria-label="Backend disconnected"` when disconnected | Label correct |
| 291 | Text hidden on mobile (`hidden sm:inline` class) | Class present |
| 292 | Polls health endpoint on mount | `api.health()` called |
| 293 | Re-polls every 30 seconds | setInterval with 30_000 |
| 294 | Cleanup cancels polling interval on unmount | clearInterval called |

---

### 2Q. `components/theme-toggle.tsx`

**Currently:** 0% tested.

**What it renders:**
- Button (ghost variant, icon-sm size) with `overflow-hidden`
- Dark mode: Moon icon (shows what you're IN)
- Light mode: Sun icon
- AnimatePresence with mode="wait": slide up/down with opacity + rotation (30deg)
- `aria-label` changes: dark → "Switch to light mode", light → "Switch to dark mode"
- Click calls `toggleTheme` from useTheme hook

| # | Test Case | Assert |
|---|-----------|--------|
| 295 | Renders Moon icon when `resolvedTheme === "dark"` | Moon SVG present |
| 296 | Renders Sun icon when `resolvedTheme === "light"` | Sun SVG present |
| 297 | `aria-label="Switch to light mode"` in dark mode | Label correct |
| 298 | `aria-label="Switch to dark mode"` in light mode | Label correct |
| 299 | Click calls `toggleTheme` | Hook function invoked |
| 300 | AnimatePresence wraps icon with motion.span | Motion element present |

---

### 2R. `components/error-boundary.tsx`

**Currently:** 0% tested.

**What it renders:**
- Normal: renders children
- Error: full-screen centered layout with:
  - Decorative glow (`bg-destructive/5 blur-3xl`, `aria-hidden`)
  - AlertTriangle icon in destructive box (h-16 w-16)
  - `<h2>` "Something went wrong"
  - Error message: `this.state.error?.message ?? "An unexpected error occurred."`
  - "Try again" button (variant="outline", size="lg")
  - Click resets state: `{hasError: false, error: null}`
- `componentDidCatch` logs to console.error

| # | Test Case | Assert |
|---|-----------|--------|
| 301 | Renders children when no error | Children visible |
| 302 | Catches render error and shows fallback UI | Error UI visible |
| 303 | Shows "Something went wrong" heading | h2 text present |
| 304 | Shows `error.message` in description | Error text matches |
| 305 | Shows fallback "An unexpected error occurred." when no message | Fallback text |
| 306 | "Try again" button resets error state | Children render again |
| 307 | Logs error + componentStack to console.error | console.error called |
| 308 | Decorative elements have `aria-hidden="true"` | Hidden from AT |

---

### 2S. Existing Test Gaps

Additional tests for components that already have test files but are missing edge cases.

#### chat-input.tsx (existing: 12 tests)

| # | Test Case | Assert |
|---|-----------|--------|
| 309 | Escape key calls `onStop` when streaming | onStop invoked |
| 310 | Escape key does nothing when not streaming | onStop NOT called |
| 311 | Submit blocked during streaming (`isStreaming=true`) | onSubmit NOT called |
| 312 | Whitespace-only input doesn't submit | onSubmit NOT called |
| 313 | Textarea height resets to "auto" after submit | Style reset |
| 314 | Form has `role="search"` attribute | Role present |
| 315 | Form has `aria-label="Ask a question"` | ARIA present |
| 316 | Helper text "Enter to send" hidden on mobile (`hidden ... sm:block`) | Class present |

#### confidence-badge.tsx (existing: 5 tests)

| # | Test Case | Assert |
|---|-----------|--------|
| 317 | Confidence 0.0 shows "0%" | Text correct |
| 318 | Confidence 1.0 shows "100%" | Text correct |
| 319 | Confidence 0.005 rounds to "1%" | Correct rounding |

#### sources-list.tsx (existing: 5 tests)

| # | Test Case | Assert |
|---|-----------|--------|
| 320 | Single source: count badge shows "1" | Correct count |
| 321 | Score 1.0 shows "100%" | Correct |
| 322 | Score 0.0 shows "0%" | Correct |

#### status-indicator.tsx (existing: 6 tests)

| # | Test Case | Assert |
|---|-----------|--------|
| 323 | "grading" pipeline status renders correct text | Status text for grading shown |

---

### 2T. Utilities & Primitives

#### `lib/utils.ts` — `cn()` function

| # | Test Case | Assert |
|---|-----------|--------|
| 324 | Merges class names correctly | `cn("px-2", "py-3") === "px-2 py-3"` |
| 325 | Resolves Tailwind conflicts (last wins) | `cn("px-2", "px-4") === "px-4"` |
| 326 | Handles conditional classes | `cn("base", false && "hidden") === "base"` |

#### shadcn/ui Button — project-specific variants

| # | Test Case | Assert |
|---|-----------|--------|
| 327 | `size="icon-sm"` renders with correct size class | Custom size applied |
| 328 | `size="xs"` renders with correct size class | Custom size applied |

---

## 3. Tier 2 — E2E Mock Mode (Playwright)

**Setup:** Playwright intercepts all `/api/*` requests using `page.route()`. No backend needed. Canned responses from `e2e/fixtures/`.

### Suite 1: Welcome & First Interaction

**Expected appearance:** Centered welcome screen with gradient heading, decorative glow orb, 4 example cards in 2x2 grid, stats line showing file/chunk counts.

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E1 | Welcome screen visible on fresh load | Screenshot: heading, orb, 4 cards | Stats fetched and shown |
| E2 | Click example question card | — | Input populated, submitted, welcome replaced by messages |
| E3 | Type question and press Enter | — | User message right-aligned, streaming starts |
| E4 | Type question and click Send button | — | Same as E3 |
| E5 | Empty input: Send button disabled | Screenshot: grayed-out button | Click does nothing |

### Suite 2: Streaming Pipeline (CRITICAL)

**Expected appearance:** Status indicator animates through phases with spinner. Tokens appear incrementally in answer area. Final answer renders as formatted markdown (headers, code blocks, lists). Source cards expand below answer. Attribution cards slide in. Confidence badge appears with color coding.

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E6 | Status shows "Classifying..." | Screenshot: spinner + text | Status indicator visible |
| E7 | Status transitions: classifying → retrieving → generating | — | Each status text appears in sequence |
| E8 | Sources appear after retrieving phase | Screenshot: source cards with scores | Correct count badge number |
| E9 | Tokens stream incrementally | — | Answer text grows as tokens arrive |
| E10 | Final answer renders markdown (headers, code, lists) | Screenshot: formatted prose | Markdown elements present in DOM |
| E11 | Code blocks have copy buttons after streaming | Screenshot: copy btn in code block | Click copies to clipboard |
| E12 | Attribution cards with sentence + similarity % | Screenshot: attribution list | Correct count and values |
| E13 | Confidence badge "Verified" (passed=true) | Screenshot: green/emerald badge | Correct text + percentage |
| E14 | Confidence badge "Unverified" (passed=false) | Screenshot: amber badge | Different styling from verified |
| E15 | "Answered in X.Xs" elapsed time shown | — | Text present after done event |
| E16 | Error during stream shows error display | Screenshot: red error card | Error message + retry button visible |
| E17 | Click retry re-submits same question | — | New stream starts with same question |
| E18 | Cancel button stops streaming mid-flow | — | Phase → complete, partial tokens preserved |

### Suite 3: Multi-turn Conversation

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E19 | Second question preserves first Q&A pair | — | Both message pairs visible |
| E20 | Session ID passed on follow-up request | — | Network request includes `session_id` param |
| E21 | Scroll to bottom on new message arrival | — | Latest message in viewport |
| E22 | Long conversation with 10+ messages scrolls | Screenshot: scrollable area | ScrollArea functional |

### Suite 4: Conversation Sidebar (DB enabled)

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E23 | Sidebar shows conversation list | Screenshot: sidebar with items | Titles + timestamps visible |
| E24 | Click conversation loads its history | — | Messages replaced with loaded conversation |
| E25 | New Chat button clears everything | — | Welcome screen reappears |
| E26 | Delete conversation removes from list | — | Item disappears with animation |
| E27 | Active conversation highlighted | Screenshot: highlight styling | `bg-accent` class applied |
| E28 | Empty sidebar shows "No conversations yet" | Screenshot: placeholder text | Text visible |
| E29 | Deleting active conversation clears chat | — | Welcome screen + session cleared |

### Suite 5: Conversation Sidebar (DB disabled — 501)

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E30 | Sidebar completely hidden | Screenshot: no sidebar element | 501 response → sidebar not rendered |
| E31 | Full-width main content area | Screenshot: content fills width | No sidebar gap |

### Suite 6: Settings Dialog

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E32 | Settings button opens dialog | Screenshot: modal overlay | Dialog visible |
| E33 | All 7 tabs visible and clickable | Screenshot: tab bar with icons | Content switches on click |
| E34 | Retrieval tab: modify top_k and save | — | PATCH request sent with updated value |
| E35 | LLM tab: change primary to ollama and save | — | 3 PATCH requests sent (llm, cody, ollama) |
| E36 | DB tab: test connection success | Screenshot: green "Connected" | Result indicator shown |
| E37 | DB tab: test connection failure | Screenshot: red error message | Error text visible |
| E38 | DB tab: run migrations success | — | Revision code shown |
| E39 | Restart required badge after LLM save | Screenshot: yellow dot on LLM tab | Badge visible |
| E40 | Indexing tab: restart badge after save | Screenshot: RestartBadge component | Yellow alert shown |
| E41 | Close dialog with X button | — | Dialog hidden |
| E42 | Close dialog with Escape key | — | Dialog hidden |
| E43 | Loading spinner while config fetches | Screenshot: centered spinner | Loader visible |

### Suite 7: Tour / Onboarding

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E44 | First visit auto-opens settings with tour | Screenshot: tour overlay | Step 1 "Welcome" visible |
| E45 | Progress bar + step counter ("1/6") | Screenshot: progress UI | Width and text correct |
| E46 | Required badge on required steps | Screenshot: "Required" badge | Badge visible |
| E47 | Optional badge on optional steps | Screenshot: "Optional" badge | Badge visible |
| E48 | Next button advances tour step | — | Counter increments, tab switches |
| E49 | Back button goes to previous step | — | Counter decrements |
| E50 | Skip button on optional steps | — | Advances past step |
| E51 | Tab switching disabled during tour | — | Click on other tabs ignored |
| E52 | Finish (last step "Get Started") completes tour | — | Tour overlay removed |
| E53 | Second visit: no auto-tour | — | Settings opens normally |
| E54 | "Take a Tour" link (main footer) restarts tour | — | Tour reactivates |
| E55 | "Take a Tour" link (settings footer) restarts tour | — | Tour reactivates |
| E56 | Non-tab steps (welcome, advanced, complete) hide tab UI | Screenshot: no tabs visible | Only overlay shown |

### Suite 8: Theme Toggle

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E57 | Default theme matches system preference | Screenshot: correct color scheme | `dark` class matches preference |
| E58 | Toggle to dark mode | Screenshot: dark bg, light text | `.dark` class on `<html>` |
| E59 | Toggle to light mode | Screenshot: light bg, dark text | No `.dark` class |
| E60 | Theme persists across page reload | — | localStorage read, correct theme applied |
| E61 | Icon animation during toggle | — | AnimatePresence transition |

### Suite 9: Connection Status

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E62 | Backend healthy: green dot + "Connected" | Screenshot: green indicator | `role="status"` present |
| E63 | Backend down: red dot + "Offline" | Screenshot: red indicator | Text changes |
| E64 | Recovery: offline → online after re-poll | — | Dot turns green, text updates |

### Suite 10: Error Boundary

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E65 | Render crash shows error boundary | Screenshot: full-screen error page | "Something went wrong" + error msg |
| E66 | "Try again" button recovers app | — | App renders normally again |

### Suite 11: Mobile Responsiveness

| # | Test Case | Visual Check | Behavior Check |
|---|-----------|-------------|----------------|
| E67 | Mobile (375px): single column welcome grid | Screenshot: mobile welcome | No horizontal overflow |
| E68 | Mobile: hamburger shows sidebar overlay | Screenshot: slide-in sidebar + backdrop | Backdrop visible |
| E69 | Mobile: backdrop click closes sidebar | — | Sidebar hides |
| E70 | Mobile: close button visible in sidebar | — | md:hidden close button shown |
| E71 | Mobile: connection status text hidden (dot only) | Screenshot: dot only | `hidden sm:inline` text hidden |
| E72 | Mobile: touch targets >= 44px | — | Button dimensions verified |
| E73 | Mobile: avatar hidden in messages | — | `hidden sm:flex` hides avatar |
| E74 | Tablet (768px): sidebar inline, no overlay | Screenshot: side-by-side | Sidebar relative positioned |
| E75 | Desktop (1280px): full layout, all elements | Screenshot: complete UI | Everything visible |

### Suite 12: Accessibility

| # | Test Case | Assert |
|---|-----------|--------|
| E76 | All interactive elements keyboard-focusable | Tab cycle through all buttons/inputs |
| E77 | Enter/Space activates buttons | Action fires |
| E78 | Dialog traps focus (Tab stays within) | Focus cycle within dialog |
| E79 | Escape closes dialog | Dialog hidden |
| E80 | `role="log"` on message list | ARIA present |
| E81 | `role="status"` on connection indicator | ARIA present |
| E82 | `role="search"` on chat form | ARIA present |
| E83 | All icon buttons have `aria-label` | No unlabeled interactive elements |
| E84 | Delete button keyboard accessible (Enter/Space) | Action fires |
| E85 | Color contrast meets WCAG AA (4.5:1 for text) | Axe accessibility audit passes |

---

## 4. Tier 3 — E2E Real Mode (Playwright + Ollama)

**Setup:** Real FastAPI server running at `localhost:8000` + Ollama with `qwen2.5:0.5b` model.

**Pre-flight:** Global setup checks health + runs smoke query. All tests skip gracefully if unavailable.

### Sub-tier: real-infra (no LLM required)

| # | Test Case | Assert |
|---|-----------|--------|
| R1 | Pre-flight: `GET /api/health` returns `{status: "ok"}` | Response verified |
| R2 | `GET /api/stats` returns real index statistics | `total_files > 0`, `total_chunks > 0` |
| R3 | `POST /api/retrieve` returns real document chunks | Chunks with scores, file paths |
| R4 | `GET /api/config` returns real configuration | All sections present |
| R5 | `PATCH /api/config` updates and returns restart info | `saved: true` |

### Sub-tier: real-llm (requires Ollama running)

| # | Test Case | Assert |
|---|-----------|--------|
| R6 | Smoke query: ask trivial question, get real answer | `answer.error === null` |
| R7 | Full Q&A: ask real question, tokens stream incrementally | Multiple `answer_token` events received |
| R8 | Real sources: retrieval returns actual chunks | Source cards with real file paths |
| R9 | Multi-turn: follow-up uses conversation context | Answer references previous Q&A |
| R10 | Real verification: confidence badge with real score | Badge renders with percentage |
| R11 | Long answer: markdown renders tables, code blocks, lists | Visual verification |
| R12 | Abort mid-stream: cancel stops real SSE connection | Partial answer preserved, no error |
| R13 | Final answer renders correctly in UI | Prose formatting visible |
| R14 | Session persistence across questions | Same session_id used |

---

## 5. Visual Regression Strategy

### Screenshot Baselines (Playwright `toHaveScreenshot`)

| ID | Page State | Viewport | Theme |
|----|-----------|----------|-------|
| V1 | Welcome screen | Desktop 1280x800 | Light |
| V2 | Welcome screen | Desktop 1280x800 | Dark |
| V3 | Welcome screen | Mobile 375x812 | Light |
| V4 | Streaming in progress (status indicator + partial tokens) | Desktop 1280x800 | Light |
| V5 | Complete answer with sources + attributions + badge | Desktop 1280x800 | Light |
| V6 | Complete answer | Desktop 1280x800 | Dark |
| V7 | Error display (stream error) | Desktop 1280x800 | Light |
| V8 | Settings dialog — Retrieval tab | Desktop 1280x800 | Light |
| V9 | Settings dialog — LLM tab with restart badge | Desktop 1280x800 | Light |
| V10 | Conversation sidebar with active item | Desktop 1280x800 | Light |
| V11 | Conversation sidebar mobile overlay | Mobile 375x812 | Light |
| V12 | Tour overlay (step 2 — LLM tab) | Desktop 1280x800 | Light |
| V13 | Connection status: connected | Desktop 1280x800 | Light |
| V14 | Connection status: offline | Desktop 1280x800 | Light |
| V15 | Empty sidebar "No conversations yet" | Desktop 1280x800 | Light |

**Threshold:** 0.2% pixel diff tolerance (handles anti-aliasing and sub-pixel rendering differences).

**Update command:** `npx playwright test --update-snapshots` (first run creates baselines, subsequent runs compare).

---

## 6. Test Counts Summary

| Tier | Category | Tests |
|------|----------|-------|
| **Tier 1: Unit** | SSE client (2A) | 12 |
| | API client gaps (2B) | 13 |
| | useStreamingQuery gaps (2C) | 10 |
| | useSettings (2D) | 16 |
| | useConversations (2E) | 10 |
| | useTour (2F) | 12 |
| | useTheme gaps (2G) | 3 |
| | useSession gaps (2H) | 1 |
| | App.tsx (2I) | 28 |
| | message-list (2J) | 22 |
| | streaming-answer (2K) | 15 |
| | welcome-screen (2L) | 13 |
| | conversation-sidebar (2M) | 23 |
| | settings-dialog (2N) | 96 |
| | attribution-list (2O) | 8 |
| | connection-status (2P) | 12 |
| | theme-toggle (2Q) | 6 |
| | error-boundary (2R) | 8 |
| | Existing test gaps (2S) | 15 |
| | Utilities & primitives (2T) | 5 |
| | **Tier 1 Subtotal** | **328** |
| **Tier 2: E2E Mock** | 12 suites (E1-E85) | **85** |
| **Tier 3: E2E Real** | 2 sub-tiers (R1-R14) | **14** |
| **Visual Regression** | Screenshots (V1-V15) | **15 baselines** |
| | | |
| | **GRAND TOTAL** | **427 tests + 15 screenshots** |

### Existing Tests (Not Re-counted Above)

| File | Existing Tests |
|------|---------------|
| api/client.test.ts | 11 |
| hooks/use-session.test.ts | 4 |
| hooks/use-streaming-query.test.ts | 11 |
| hooks/use-theme.test.ts | 5 |
| components/chat-input.test.tsx | 12 |
| components/status-indicator.test.tsx | 6 |
| components/confidence-badge.test.tsx | 5 |
| components/error-display.test.tsx | 5 |
| components/sources-list.test.tsx | 5 |
| **Existing Subtotal** | **64** |
| | |
| **OVERALL TOTAL** | **491 tests + 15 screenshots** |

---

## 7. Execution Plan

| Phase | What | Command | Dependencies |
|-------|------|---------|-------------|
| 1 | Install Playwright | `npx playwright install chromium` | None |
| 2 | Run existing unit tests | `npm test` | None |
| 3 | Write new unit tests (Tier 1) | Add `*.test.{ts,tsx}` files | None |
| 4 | Run all unit tests | `npm test` | Phase 3 |
| 5 | Write E2E fixtures + helpers | Create `e2e/` directory structure | None |
| 6 | Write E2E mock-mode tests | `e2e/mock-mode/*.spec.ts` | Phase 5 |
| 7 | Run E2E mock mode | `npx playwright test e2e/mock-mode/` | Built frontend |
| 8 | Write E2E real-mode tests | `e2e/real-mode/*.spec.ts` | Phase 5 |
| 9 | Start Ollama + pull model | `ollama serve` + `ollama pull qwen2.5:0.5b` | Ollama installed |
| 10 | Start backend | `doc-qa serve --repo <path>` | Index exists |
| 11 | Run E2E real mode | `npx playwright test e2e/real-mode/` | Phases 9-10 |
| 12 | Create visual baselines | `npx playwright test --update-snapshots` | Phase 7 |

### NPM Scripts (to add)

```json
{
  "test": "vitest run",
  "test:watch": "vitest",
  "test:e2e": "playwright test e2e/mock-mode/",
  "test:e2e:real": "playwright test e2e/real-mode/",
  "test:e2e:update-snapshots": "playwright test --update-snapshots",
  "test:all": "vitest run && playwright test e2e/mock-mode/"
}
```

---

## Appendix: File → Test Mapping

Every source file in `frontend/src/` and its test coverage:

| Source File | Test File(s) | Status |
|-------------|-------------|--------|
| `App.tsx` | `App.test.tsx` | NEW — 28 tests |
| `main.tsx` | — | No logic (render only) |
| `api/client.ts` | `api/client.test.ts` | EXISTING 11 + NEW 13 |
| `api/sse-client.ts` | `api/sse-client.test.ts` | NEW — 12 tests |
| `hooks/use-session.ts` | `hooks/use-session.test.ts` | EXISTING 4 + NEW 1 |
| `hooks/use-streaming-query.ts` | `hooks/use-streaming-query.test.ts` | EXISTING 11 + NEW 10 |
| `hooks/use-theme.ts` | `hooks/use-theme.test.ts` | EXISTING 5 + NEW 3 |
| `hooks/use-settings.ts` | `hooks/use-settings.test.ts` | NEW — 16 tests |
| `hooks/use-conversations.ts` | `hooks/use-conversations.test.ts` | NEW — 10 tests |
| `hooks/use-tour.ts` | `hooks/use-tour.test.ts` | NEW — 12 tests |
| `components/chat-input.tsx` | `components/chat-input.test.tsx` | EXISTING 12 + NEW 8 |
| `components/message-list.tsx` | `components/message-list.test.tsx` | NEW — 22 tests |
| `components/streaming-answer.tsx` | `components/streaming-answer.test.tsx` | NEW — 15 tests |
| `components/status-indicator.tsx` | `components/status-indicator.test.tsx` | EXISTING 6 + NEW 1 |
| `components/sources-list.tsx` | `components/sources-list.test.tsx` | EXISTING 5 + NEW 3 |
| `components/confidence-badge.tsx` | `components/confidence-badge.test.tsx` | EXISTING 5 + NEW 3 |
| `components/error-display.tsx` | `components/error-display.test.tsx` | EXISTING 5 |
| `components/welcome-screen.tsx` | `components/welcome-screen.test.tsx` | NEW — 13 tests |
| `components/connection-status.tsx` | `components/connection-status.test.tsx` | NEW — 12 tests |
| `components/theme-toggle.tsx` | `components/theme-toggle.test.tsx` | NEW — 6 tests |
| `components/attribution-list.tsx` | `components/attribution-list.test.tsx` | NEW — 8 tests |
| `components/conversation-sidebar.tsx` | `components/conversation-sidebar.test.tsx` | NEW — 23 tests |
| `components/settings-dialog.tsx` | `components/settings-dialog.test.tsx` | NEW — 96 tests |
| `components/error-boundary.tsx` | `components/error-boundary.test.tsx` | NEW — 8 tests |
| `lib/utils.ts` | `lib/utils.test.ts` | NEW — 3 tests |
| `components/ui/button.tsx` | `components/ui/button.test.tsx` | NEW — 2 tests |
| `types/api.ts` | — | Type-only (no runtime logic) |
| `types/sse.ts` | — | Type-only (no runtime logic) |
| `components/ui/badge.tsx` | — | shadcn upstream tested |
| `components/ui/card.tsx` | — | shadcn upstream tested |
| `components/ui/input.tsx` | — | shadcn upstream tested |
| `components/ui/label.tsx` | — | shadcn upstream tested |
| `components/ui/dialog.tsx` | — | shadcn upstream tested |
| `components/ui/tabs.tsx` | — | shadcn upstream tested |
| `components/ui/switch.tsx` | — | shadcn upstream tested |
| `components/ui/select.tsx` | — | shadcn upstream tested |
| `components/ui/scroll-area.tsx` | — | shadcn upstream tested |
