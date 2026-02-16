"""LLM backend abstraction with Cody and Ollama implementations."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Directory for temp context files passed to Cody as @mentions
_CONTEXT_TEMP_DIR = os.path.join(tempfile.gettempdir(), "doc_qa_cody_context")

# Cody agent version for auto-download (matches cody_agentic_tool)
_CODY_AGENT_VERSION = "5.5.14"
_CODY_BINARY_DIR = os.path.join(os.path.expanduser("~"), ".cody-agent")


@dataclass
class Answer:
    """Structured answer from the LLM."""

    text: str
    sources: list[str]
    model: str
    error: str | None = None


# Import the canonical system prompt (single source of truth)
from doc_qa.llm.prompt_templates import SYSTEM_PROMPT as _SYSTEM_PROMPT


class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends."""

    @abc.abstractmethod
    async def ask(self, question: str, context: str, history: list[dict] | None = None) -> Answer:
        """Send a question with retrieved context to the LLM.

        Args:
            question: User's question.
            context: Formatted context from retrieved chunks.
            history: Optional conversation history for multi-turn.

        Returns:
            Structured Answer with text and source references.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


# ---------------------------------------------------------------------------
# JSON-RPC helpers (minimal, inlined — no need for a full module)
# ---------------------------------------------------------------------------

class _JSONRPCError(Exception):
    def __init__(self, method: str, error: dict) -> None:
        self.method = method
        self.error = error
        super().__init__(f"RPC error in {method}: {error}")


_RPC_READ_TIMEOUT = 120  # seconds — matches cody_agentic_tool


class _JSONRPCHandler:
    """Minimal JSON-RPC 2.0 over stdio (Content-Length framing)."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._msg_id = 0

    async def request(self, method: str, params: Any = None) -> Any:
        """Send a request and wait for the matching response."""
        self._msg_id += 1
        req_id = self._msg_id

        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        await self._send(payload)

        # Read messages until we get our response
        while True:
            msg = await self._read()
            if msg is None:
                raise ConnectionError("Agent process ended unexpectedly.")
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise _JSONRPCError(method, msg["error"])
                return msg.get("result")
            # Server→client request — auto-respond with null
            if "method" in msg and "id" in msg:
                await self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})

    async def notify(self, method: str, params: Any = None) -> None:
        """Send a notification (no response expected)."""
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        await self._send(payload)

    async def _send(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self._writer.write(header + data)
        await self._writer.drain()

    async def _read(self) -> dict | None:
        try:
            # Read headers until \r\n\r\n (with timeout to prevent hangs)
            header = await asyncio.wait_for(
                self._reader.readuntil(b"\r\n\r\n"),
                timeout=_RPC_READ_TIMEOUT,
            )
            length_line = header.split(b"\r\n")[0]
            content_length = int(length_line.split(b":")[1].strip())

            # Read exact body
            body = await asyncio.wait_for(
                self._reader.readexactly(content_length),
                timeout=_RPC_READ_TIMEOUT,
            )
            return json.loads(body)
        except asyncio.TimeoutError:
            logger.error("JSON-RPC read timed out after %ds", _RPC_READ_TIMEOUT)
            return None
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionResetError):
            return None

    async def request_streaming(
        self,
        method: str,
        params: Any = None,
        on_token: Callable[[str], None] | None = None,
        on_token_usage: Callable[[dict], None] | None = None,
    ) -> Any:
        """Send request and call *on_token* with cumulative text for each in-progress notification.

        Args:
            on_token: Called with partial assistant text as it streams.
            on_token_usage: Called with token usage dict containing
                completionTokens, promptTokens, percentUsed.
        """
        self._msg_id += 1
        req_id = self._msg_id
        payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        await self._send(payload)

        while True:
            msg = await self._read()
            if msg is None:
                raise ConnectionError("Agent process ended unexpectedly.")
            if msg.get("id") == req_id:
                if "error" in msg:
                    raise _JSONRPCError(method, msg["error"])
                return msg.get("result")
            # In-progress streaming notification
            msg_params = msg.get("params", {})
            if isinstance(msg_params, dict):
                message_data = msg_params.get("message", msg_params)
                if isinstance(message_data, dict) and message_data.get("isMessageInProgress"):
                    # Extract token usage
                    if on_token_usage:
                        token_usage = message_data.get("tokenUsage")
                        if token_usage and isinstance(token_usage, dict):
                            try:
                                on_token_usage(token_usage)
                            except Exception:
                                pass
                    # Extract partial text
                    if on_token:
                        messages = message_data.get("messages", [])
                        if messages:
                            last = messages[-1]
                            if last.get("speaker") == "assistant":
                                on_token(last.get("text", ""))
                    continue
            # Server-to-client request -- auto-respond
            if "method" in msg and "id" in msg:
                await self._send({"jsonrpc": "2.0", "id": msg["id"], "result": None})


# ---------------------------------------------------------------------------
# Cody binary resolution + auto-download
# ---------------------------------------------------------------------------


def _find_cody_binary(custom_path: str | None = None) -> str | None:
    """Locate the Cody agent binary (sync).

    Returns the path if found, or *None* if not available locally.
    Raises :class:`FileNotFoundError` only when *custom_path* is given but
    does not exist (explicit user error).  Auto-download is handled
    separately in :func:`_auto_download_cody_binary`.
    """
    # 1. Explicit custom path — hard error if missing
    if custom_path:
        if Path(custom_path).exists():
            return custom_path
        raise FileNotFoundError(
            f"Specified Cody binary not found: {custom_path}\n"
            "Ensure the path is correct and the file exists."
        )

    # 2. CODY_AGENT_BINARY env var
    env_binary = os.environ.get("CODY_AGENT_BINARY")
    if env_binary and Path(env_binary).exists():
        logger.info("Using CODY_AGENT_BINARY from env: %s", env_binary)
        return env_binary

    # 3. `cody` on PATH (npm install -g @sourcegraph/cody)
    which = shutil.which("cody")
    if which:
        logger.info("Found cody on PATH: %s", which)
        return which

    # 4. Previously-downloaded binary in ~/.cody-agent
    binary_name = f"cody-agent-{_CODY_AGENT_VERSION}{'.cmd' if os.name == 'nt' else ''}"
    cached = os.path.join(_CODY_BINARY_DIR, binary_name)
    if os.path.isfile(cached):
        logger.info("Using cached cody binary: %s", cached)
        return cached

    return None


def _check_node_installed() -> bool:
    """Check if Node.js is available on PATH."""
    return shutil.which("node") is not None


async def _auto_download_cody_binary() -> str:
    """Download the Cody agent binary from npm and return its path.

    Uses httpx (already a project dependency) for the download.
    Ported from cody_agentic_tool/utils.py.
    """
    import httpx

    os.makedirs(_CODY_BINARY_DIR, exist_ok=True)

    version = _CODY_AGENT_VERSION
    url = f"https://registry.npmjs.org/@sourcegraph/cody/-/cody-{version}.tgz"
    tar_path = os.path.join(_CODY_BINARY_DIR, f"cody-{version}.tgz")

    logger.info("Downloading Cody Agent %s from %s", version, url)

    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(tar_path, "wb") as f:
                f.write(resp.content)
    except Exception as exc:
        raise FileNotFoundError(
            f"Failed to download Cody Agent binary: {exc}\n\n"
            "Possible fixes:\n"
            "  1. Check your internet connection\n"
            "  2. Install cody globally: npm install -g @sourcegraph/cody\n"
            "  3. Set CODY_AGENT_BINARY env var to an existing binary path"
        ) from exc

    # Extract tarball
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            def _safe(members):
                for m in members:
                    if m.name.startswith("/") or ".." in m.name:
                        continue
                    yield m
            tar.extractall(path=_CODY_BINARY_DIR, members=_safe(tar))
        os.remove(tar_path)
    except Exception as exc:
        raise FileNotFoundError(f"Failed to extract Cody Agent: {exc}") from exc

    # Create platform-appropriate wrapper script
    index_js = os.path.join(_CODY_BINARY_DIR, "package", "dist", "index.js")
    if not os.path.exists(index_js):
        raise FileNotFoundError(f"Downloaded package missing index.js at {index_js}")

    binary_name = f"cody-agent-{version}{'.cmd' if os.name == 'nt' else ''}"
    binary_path = os.path.join(_CODY_BINARY_DIR, binary_name)

    if os.name == "nt":
        script = f'@echo off\r\nnode "{index_js}" %*'
    else:
        script = f'#!/bin/sh\nnode "{index_js}" "$@"'

    with open(binary_path, "w", encoding="utf-8") as f:
        f.write(script)
    if os.name != "nt":
        os.chmod(binary_path, 0o755)

    logger.info("Cody Agent %s downloaded to %s", version, binary_path)
    return binary_path


# ---------------------------------------------------------------------------
# Cody Backend
# ---------------------------------------------------------------------------


class CodyBackend(LLMBackend):
    """LLM backend using Cody agent via JSON-RPC over stdio."""

    _MAX_RESTARTS = 3

    def __init__(
        self,
        access_token: str | None = None,
        endpoint: str = "https://sourcegraph.com",
        model: str = "anthropic::2025-01-01::claude-3.5-sonnet",
        agent_binary: str | None = None,
        workspace_root: str | None = None,
        access_token_env: str = "SRC_ACCESS_TOKEN",
    ) -> None:
        self._access_token = access_token or os.environ.get(access_token_env, "")
        self._endpoint = endpoint
        self._model = model
        self._workspace_root = workspace_root or os.getcwd()
        self._process: asyncio.subprocess.Process | None = None
        self._rpc: _JSONRPCHandler | None = None
        self._chat_id: str | None = None
        self._initialized = False
        self._rpc_lock = asyncio.Lock()
        # Restart resilience (ported from cody_agentic_tool)
        self._restart_count: int = 0
        self._restart_lock = asyncio.Lock()
        # Binary: resolved lazily (may need async download)
        self._binary: str | None = _find_cody_binary(agent_binary)
        self._auth_status: dict | None = None

    # -- Fix #3: Restart protection with counter + lock --------------------

    async def _ensure_initialized(self) -> None:
        """Start agent process and initialize if needed, with restart protection."""
        if self._initialized and self._process and self._process.returncode is None:
            return

        async with self._restart_lock:
            # Re-check after acquiring lock (another coroutine may have restarted)
            if self._initialized and self._process and self._process.returncode is None:
                return

            if self._initialized:
                # Was initialized before → this is a restart
                self._restart_count += 1
                if self._restart_count > self._MAX_RESTARTS:
                    raise ConnectionError(
                        f"Cody agent failed to stay alive after {self._MAX_RESTARTS} restarts. "
                        "Check logs for details."
                    )
                logger.warning(
                    "Cody agent died (restart #%d/%d). Restarting...",
                    self._restart_count, self._MAX_RESTARTS,
                )

            await self._start_agent()

    async def _start_agent(self) -> None:
        """Spawn the Cody agent subprocess and initialize."""
        # -- Fix #5: Early token validation --------------------------------
        if not self._access_token:
            raise ConnectionError(
                "SRC_ACCESS_TOKEN is not set.\n\n"
                "You must provide a Sourcegraph access token:\n"
                "  1. Go to https://sourcegraph.com/user/settings/tokens\n"
                "  2. Create a new token\n"
                "  3. export SRC_ACCESS_TOKEN=<your-token>"
            )

        # -- Fix #2: Auto-download binary if not found locally -------------
        if not self._binary:
            if not _check_node_installed():
                raise FileNotFoundError(
                    "Cody agent binary not found and Node.js is not installed.\n\n"
                    "To fix this, either:\n"
                    "  1. Install Node.js (https://nodejs.org/) and re-run\n"
                    "  2. Install cody globally: npm install -g @sourcegraph/cody\n"
                    "  3. Set CODY_AGENT_BINARY env var to point to an existing binary"
                )
            self._binary = await _auto_download_cody_binary()

        # Kill existing process if any
        if self._process and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()

        env = os.environ.copy()
        env["CODY_AGENT_DEBUG_REMOTE"] = "false"
        trace_path = os.getenv("CODY_AGENT_TRACE_PATH")
        if trace_path:
            env["CODY_AGENT_TRACE_PATH"] = trace_path

        args = [self._binary, "api", "jsonrpc-stdio"]
        if sys.platform == "win32" and self._binary.endswith(".cmd"):
            args = ["cmd.exe", "/c"] + args

        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Quick health check — ensure process didn't crash on startup
        await asyncio.sleep(0.5)
        if self._process.returncode is not None:
            stderr_data = await self._process.stderr.read()
            error_msg = (
                stderr_data.decode(errors="replace").strip()
                if stderr_data else "No error output"
            )
            raise RuntimeError(
                f"Cody agent exited immediately (code {self._process.returncode}).\n"
                f"Error: {error_msg}\n\n"
                "Common causes:\n"
                "  - Node.js not installed or not on PATH\n"
                "  - Incompatible Node.js version (requires 18+)\n"
                f"  - Corrupted download (delete {_CODY_BINARY_DIR} and retry)"
            )

        self._rpc = _JSONRPCHandler(self._process.stdout, self._process.stdin)

        # -- Fix #1: Declare "chat": "streaming" capability ----------------
        ws_uri = Path(self._workspace_root).as_uri()
        init_params = {
            "name": "doc-qa-tool",
            "version": "0.1",
            "workspaceRootUri": ws_uri,
            "extensionConfiguration": {
                "accessToken": self._access_token,
                "serverEndpoint": self._endpoint,
                "codebase": None,
                "customConfiguration": {},
            },
            "capabilities": {
                "chat": "streaming",
                "completions": "none",
                "edit": "none",
                "editWorkspace": "none",
                "codeLenses": "none",
                "showDocument": "none",
                "ignore": "none",
                "untitledDocuments": "none",
                "showWindowMessage": "notification",
            },
        }

        result = await self._rpc.request("initialize", init_params)
        await self._rpc.notify("initialized", None)

        auth = result.get("authStatus", {}) if result else {}
        if not auth.get("authenticated", False):
            raise ConnectionError(
                f"Cody authentication failed: {auth.get('error', 'unknown error')}\n\n"
                "Check that your SRC_ACCESS_TOKEN is valid and not expired."
            )

        self._auth_status = auth
        self._initialized = True
        logger.info("Cody agent initialized (model=%s).", self._model)

    async def _new_chat(self) -> str:
        """Create a fresh chat session with the configured model."""
        chat_id = await self._rpc.request("chat/new", None)
        await self._rpc.request(
            "chat/setModel",
            {"id": chat_id, "model": self._model},
        )
        return chat_id

    @staticmethod
    def _write_context_files(context: str) -> tuple[list[dict], list[str]]:
        """Write context chunks to temp files and build contextFiles.

        Returns (context_items, temp_file_paths) where context_items is a list
        of Cody contextItem dicts and temp_file_paths tracks files for cleanup.
        """
        if not context:
            return [], []

        os.makedirs(_CONTEXT_TEMP_DIR, exist_ok=True)
        context_items: list[dict] = []
        temp_paths: list[str] = []

        # Split context into individual source chunks.
        # Format: "[Source N: filename ...] (score: X.XXX)\n<text>\n\n[Source ..."
        chunks = context.split("\n[Source ")
        for i, chunk_text in enumerate(chunks):
            raw = chunk_text.strip()
            if not raw:
                continue
            # Re-add the "[Source " prefix that was consumed by split (except first)
            if i > 0:
                raw = "[Source " + raw

            # Write to a temp file
            filename = f"chunk_{i}.txt"
            filepath = os.path.join(_CONTEXT_TEMP_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(raw)
            temp_paths.append(filepath)

            context_items.append({
                "type": "file",
                "uri": {
                    "fsPath": filepath,
                    "path": filepath,
                },
            })

        return context_items, temp_paths

    @staticmethod
    def _cleanup_context_files(temp_paths: list[str]) -> None:
        """Remove temporary context files."""
        for p in temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    async def _notify_context_files(self, context_items: list[dict]) -> None:
        """Send textDocument/didOpen for each context file so the agent is aware."""
        for item in context_items:
            fs_path = item.get("uri", {}).get("fsPath")
            if not fs_path:
                continue
            try:
                with open(fs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                await self._rpc.notify("textDocument/didOpen", {
                    "uri": Path(fs_path).as_uri(),
                    "content": content,
                })
            except Exception as exc:
                logger.debug("Failed to notify didOpen for %s: %s", fs_path, exc)

    def _build_request_data(
        self,
        chat_id: str,
        prompt: str,
        context_files: list[dict],
    ) -> dict:
        """Build the chat/submitMessage request payload."""
        return {
            "id": chat_id,
            "message": {
                "command": "submit",
                "text": prompt,
                "submitType": "user",
                "addEnhancedContext": False,
                "contextFiles": context_files,
            },
        }

    async def ask(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> Answer:
        """Send question + context to Cody and get the answer.

        Each call creates a fresh chat session to prevent transcript bloat.
        Context chunks are written to temp files and passed as contextFiles
        (file @mentions) to leverage Cody's higher context budget.
        """
        context_items, temp_paths = self._write_context_files(context)
        logger.info(
            "Cody ask: %d context files, question=%r",
            len(context_items), question[:80],
        )
        for ci in context_items:
            logger.debug("  contextFile: %s", ci.get("uri", {}).get("fsPath", "?"))
        try:
            async with self._rpc_lock:
                await self._ensure_initialized()
                chat_id = await self._new_chat()

                # Notify agent about context files for better awareness
                await self._notify_context_files(context_items)

                prompt = self._build_prompt(question, context="", history=history)
                request_data = self._build_request_data(chat_id, prompt, context_items)

                try:
                    response = await self._rpc.request("chat/submitMessage", request_data)
                except _JSONRPCError as e:
                    logger.error("Cody RPC error: %s", e.error)
                    return Answer(text="", sources=[], model=self._model, error=str(e.error))
                except ConnectionError as e:
                    logger.error("Cody connection lost: %s", e)
                    self._initialized = False
                    return Answer(text="", sources=[], model=self._model, error=str(e))
        finally:
            self._cleanup_context_files(temp_paths)

        # Reset restart count on successful round-trip
        self._restart_count = 0

        text = self._extract_response(response)
        return Answer(text=text, sources=[], model=self._model)

    async def ask_streaming(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_token_usage: Callable[[dict], None] | None = None,
    ) -> Answer:
        """Send question + context to Cody with streaming token callbacks.

        Works like :meth:`ask` but invokes *on_token* with the cumulative
        assistant text each time an ``isMessageInProgress`` notification
        arrives from the agent.

        Args:
            on_token_usage: Called with token usage dict containing
                completionTokens, promptTokens, percentUsed.
        """
        context_items, temp_paths = self._write_context_files(context)
        logger.info(
            "Cody ask_streaming: %d context files, question=%r",
            len(context_items), question[:80],
        )
        for ci in context_items:
            logger.debug("  contextFile: %s", ci.get("uri", {}).get("fsPath", "?"))
        try:
            async with self._rpc_lock:
                await self._ensure_initialized()
                chat_id = await self._new_chat()

                await self._notify_context_files(context_items)

                prompt = self._build_prompt(question, context="", history=history)
                request_data = self._build_request_data(chat_id, prompt, context_items)

                try:
                    response = await self._rpc.request_streaming(
                        "chat/submitMessage", request_data,
                        on_token=on_token, on_token_usage=on_token_usage,
                    )
                except _JSONRPCError as e:
                    logger.error("Cody RPC error: %s", e.error)
                    return Answer(text="", sources=[], model=self._model, error=str(e.error))
                except ConnectionError as e:
                    logger.error("Cody connection lost: %s", e)
                    self._initialized = False
                    return Answer(text="", sources=[], model=self._model, error=str(e))
        finally:
            self._cleanup_context_files(temp_paths)

        # Reset restart count on successful round-trip
        self._restart_count = 0

        text = self._extract_response(response)
        return Answer(text=text, sources=[], model=self._model)

    def _build_prompt(
        self,
        question: str,
        context: str = "",
        history: list[dict] | None = None,
    ) -> str:
        """Build the prompt text sent in the message field.

        When context is passed as contextFiles (file @mentions), the *context*
        parameter should be empty — the system prompt + history + question is
        all that goes in the text field, keeping it within the ~10K text budget.
        Context is still accepted for backward compatibility (e.g. Ollama).
        """
        parts: list[str] = []

        parts.append(_SYSTEM_PROMPT + "\n")

        if context:
            parts.append("## Context\n")
            parts.append(context)
            parts.append("")

        if history:
            parts.append("## Previous conversation\n")
            for msg in history:
                role = msg.get("role", "user")
                text = msg.get("text", "")
                parts.append(f"**{role}**: {text}\n")

        parts.append(f"## Question\n\n{question}")

        return "\n".join(parts)

    # -- Fix #4: Response type validation ----------------------------------

    @staticmethod
    def _extract_response(response: Any) -> str:
        """Extract the assistant's text from a chat transcript response."""
        if not response:
            return ""

        # Validate response type
        if isinstance(response, dict):
            resp_type = response.get("type")
            if resp_type and resp_type != "transcript":
                logger.warning("Unexpected Cody response type: %s", resp_type)
                if resp_type == "error":
                    return ""

        messages = response.get("messages", [])
        # Walk backwards to find last assistant message
        for msg in reversed(messages):
            if msg.get("speaker") == "assistant":
                return msg.get("text", "")

        return ""

    async def new_conversation(self) -> None:
        """Reset conversation state.

        Each ask() already creates a fresh chat session, so this just checks
        agent health and clears any stale state.
        """
        if self._initialized and self._process and self._process.returncode is not None:
            self._initialized = False

    async def close(self) -> None:
        """Shut down the Cody agent process."""
        if self._process and self._process.returncode is None:
            # Try graceful JSON-RPC shutdown first (ported from cody_agentic_tool)
            if self._rpc:
                try:
                    await asyncio.wait_for(
                        self._rpc.request("shutdown", None),
                        timeout=3.0,
                    )
                except Exception:
                    pass
                try:
                    await self._rpc.notify("exit", None)
                except Exception:
                    pass
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
        self._initialized = False
        self._chat_id = None
        logger.info("Cody agent shut down.")

    @staticmethod
    async def test_connection(
        endpoint: str,
        access_token: str,
        workspace_root: str | None = None,
    ) -> dict:
        """Test Cody connection: spawn agent, authenticate, list models, close.

        Returns:
            ``{"ok": True, "user": {...}, "models": [...]}`` on success,
            ``{"ok": False, "error": "..."}`` on failure.
        """
        from doc_qa.llm.models import format_cody_model

        backend = CodyBackend(
            access_token=access_token,
            endpoint=endpoint,
            workspace_root=workspace_root,
        )
        try:
            await backend._start_agent()

            # Extract user info from auth status
            auth = backend._auth_status or {}
            user = {
                "username": auth.get("username", ""),
                "email": auth.get("primaryEmail", ""),
                "displayName": auth.get("displayName", auth.get("username", "")),
            }

            # List available models via RPC
            models: list[dict] = []
            try:
                raw_models = await backend._rpc.request("chat/models", {"modelUsage": "chat"})
                model_list = raw_models.get("models", []) if isinstance(raw_models, dict) else []
                for m in model_list:
                    model_id = m.get("id", "") if isinstance(m, dict) else str(m)
                    caps = m if isinstance(m, dict) else None
                    models.append(format_cody_model(model_id, caps))
            except Exception as exc:
                logger.warning("Failed to list Cody models: %s", exc)

            return {"ok": True, "user": user, "models": models}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            await backend.close()


# ---------------------------------------------------------------------------
# Ollama Backend
# ---------------------------------------------------------------------------


class OllamaBackend(LLMBackend):
    """LLM backend using a local Ollama instance via its REST API.

    Ollama must be running at the configured host (default: http://localhost:11434).
    The model must be pulled beforehand (e.g., `ollama pull qwen2.5:7b`).
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: float = 120.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def ask(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> Answer:
        """Send question + context to Ollama and get the answer."""
        import httpx

        messages = self._build_messages(question, context, history)

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._host}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            return Answer(
                text="",
                sources=[],
                model=self._model,
                error=f"Cannot connect to Ollama at {self._host}. Is it running?",
            )
        except httpx.HTTPStatusError as e:
            return Answer(
                text="",
                sources=[],
                model=self._model,
                error=f"Ollama HTTP error: {e.response.status_code}",
            )
        except Exception as e:
            return Answer(
                text="",
                sources=[],
                model=self._model,
                error=f"Ollama error: {e}",
            )

        text = data.get("message", {}).get("content", "")
        return Answer(text=text, sources=[], model=self._model)

    def _build_messages(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> list[dict]:
        """Build the Ollama chat messages list."""
        messages: list[dict] = []

        # System prompt
        system = _SYSTEM_PROMPT
        messages.append({"role": "system", "content": system})

        # Conversation history
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("text", "")})

        # Current question with context
        user_msg = ""
        if context:
            user_msg += f"## Context\n\n{context}\n\n"
        user_msg += f"## Question\n\n{question}"
        messages.append({"role": "user", "content": user_msg})

        return messages

    async def close(self) -> None:
        """No cleanup needed for Ollama (stateless HTTP)."""
        pass


# -- Fix #6: FallbackBackend exposes ask_streaming -------------------------


class FallbackBackend(LLMBackend):
    """Wraps a primary and fallback backend -- retries on the fallback if primary fails."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend) -> None:
        self._primary = primary
        self._fallback = fallback

    async def ask(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> Answer:
        answer = await self._primary.ask(question, context, history)
        if answer.error is not None:
            logger.warning(
                "Primary backend failed (%s), trying fallback.", answer.error
            )
            return await self._fallback.ask(question, context, history)
        return answer

    async def ask_streaming(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> Answer:
        """Delegate streaming to primary if supported, else fall back."""
        if hasattr(self._primary, "ask_streaming"):
            try:
                answer = await self._primary.ask_streaming(
                    question, context, history, on_token=on_token,
                )
                if answer.error is None:
                    return answer
                logger.warning(
                    "Primary streaming failed (%s), trying fallback.", answer.error
                )
            except Exception as exc:
                logger.warning("Primary streaming error: %s", exc)
        # Fallback: try streaming on fallback, else plain ask
        if hasattr(self._fallback, "ask_streaming"):
            return await self._fallback.ask_streaming(
                question, context, history, on_token=on_token,
            )
        return await self._fallback.ask(question, context, history)

    async def close(self) -> None:
        await self._primary.close()
        await self._fallback.close()


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def create_backend(
    primary: str = "cody",
    fallback: str | None = None,
    cody_endpoint: str = "https://sourcegraph.com",
    cody_model: str = "anthropic::2025-01-01::claude-3.5-sonnet",
    cody_binary: str | None = None,
    workspace_root: str | None = None,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:7b",
    cody_access_token_env: str = "SRC_ACCESS_TOKEN",
) -> LLMBackend:
    """Create the configured LLM backend.

    Args:
        primary: "cody" or "ollama".
        fallback: Optional fallback backend type ("cody" or "ollama").
        Other args: backend-specific configuration.

    Returns:
        An LLMBackend instance.
    """
    def _make(backend_type: str) -> LLMBackend:
        if backend_type == "ollama":
            return OllamaBackend(host=ollama_host, model=ollama_model)
        return CodyBackend(
            endpoint=cody_endpoint,
            model=cody_model,
            agent_binary=cody_binary,
            workspace_root=workspace_root,
            access_token_env=cody_access_token_env,
        )

    primary_backend = _make(primary)

    if fallback and fallback != primary:
        fallback_backend = _make(fallback)
        return FallbackBackend(primary_backend, fallback_backend)

    return primary_backend
